### This script was built based on HYTREL implementation.
### Ref: https://github.com/awslabs/hypergraph-tabular-lm 
import logging
import numpy as np
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import transformers
from transformers.optimization import AdamW, get_scheduler
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from typing import Optional
# from model import Encoder ## file is a direct copy from the original file found in the repo https://github.com/awslabs/hypergraph-tabular-lm 
# from data import BipartiteData ## file is a direct copy from the original file found in the repo https://github.com/awslabs/hypergraph-tabular-lm 
from dataclasses import dataclass, field, fields
from torchmetrics import Precision, Recall, F1Score, AveragePrecision
####
import re
import os
import sys
import json
import logging
import pandas as pd
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import  CrossEntropyLoss
from torchmetrics import Precision, Recall, F1Score, Accuracy

import transformers
from transformers.optimization import AdamW, get_scheduler
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from typing import Optional
from dataclasses import dataclass, field
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from hytrel_modules.model import Encoder
from hytrel_modules.data import BipartiteData
from hytrel_modules.parallel_clean import clean_cell_value
import time

### scripts utils 
import argparse
import gzip
import configs.common as common_configs
import configs.hytrel_model as hytrel_model
from collections import OrderedDict

#********************************* set up arguments *********************************

@dataclass
class DataArguments:
    """
    Arguments pertaining to which config/tokenizer we are going use.
    """
    tokenizer_config_type: str = field(
        default='bert-base-uncased',
        metadata={
            "help": "bert-base-cased, bert-base-uncased etc"
        },
    )
    data_path: str = field(default='../table_graph/data/col_rel/', metadata={"help": "data path"})
    max_token_length: int = field(
        default=64,
        metadata={
            "help": "The maximum total input token length for cell/caption/header after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    max_row_length: int = field(
        default=30,
        metadata={
            "help": "The maximum total input rows for a table"
        },
    )
    max_column_length: int = field(
        default=20,
        metadata={
            "help": "The maximum total input columns for a table"

        },
    )
    label_type_num: int = field(
        default=121,
        metadata={
            "help": "The total label types"

        },
    )

    num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of workers for dataloader"},
    )

    valid_ratio: float = field(
        default=0.3,
        metadata={"help": "Number of workers for dataloader"},
    )


    def __post_init__(self):
        if self.tokenizer_config_type not in ["bert-base-cased", "bert-base-uncased"]:
            raise ValueError(
                f"The model type should be bert-base-(un)cased. The current value is {self.tokenizer_config_type}."
            )

@dataclass
class OptimizerConfig:
    batch_size: int = 256
    base_learning_rate: float = 1e-3
    weight_decay: float = 0.02
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-5
    lr_scheduler_type: transformers.SchedulerType = "linear"
    warmup_step_ratio: float = 0.1
    seed: int = 42
    optimizer: str = "Adam"
    adam_w_mode: bool = True
    save_every_n_epochs: int=1
    save_top_k: int=1
    checkpoint_path: str=''


    def __post_init__(self):
        if self.optimizer.lower() not in {
            "adam",
            "fusedadam",
            "fusedlamb",
            "fusednovograd",
        }:
            raise KeyError(
                f"The optimizer type should be one of: Adam, FusedAdam, FusedLAMB, FusedNovoGrad. The current value is {self.optimizer}."
            )

    def get_optimizer(self, optim_groups, learning_rate):
        optimizer = self.optimizer.lower()
        optim_cls = {
            "adam": AdamW if self.adam_w_mode else Adam,
        }[optimizer]

        args = [optim_groups]
        kwargs = {
            "lr": learning_rate,
            "eps": self.adam_epsilon,
            "betas": (self.adam_beta1, self.adam_beta2),
        }
        if optimizer in {"fusedadam", "fusedlamb"}:
            kwargs["adam_w_mode"] = self.adam_w_mode

        optimizer = optim_cls(*args, **kwargs)
        return optimizer
    
def get_model():
    py_logger = logging.getLogger(__name__)
    py_logger.setLevel(logging.INFO)
    # ********************************* parse arguments *********************************
    parser = HfArgumentParser((DataArguments, OptimizerConfig))
    parser = pl.Trainer.add_argparse_args(parser)

    (
        data_args,
        optimizer_cfg,
        trainer_args,
    ) = parser.parse_args_into_dataclasses()
    py_logger.info(f"data_args: {data_args}\n")
    py_logger.info(f"optimizer_cfg: {optimizer_cfg}\n")
    py_logger.info(f"trainer_args: {trainer_args}\n")

    # ********************************* set up tokenizer and model config*********************************
    # custom BERT tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_config_type)  
    new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
    py_logger.info(f"new tokens added: {new_tokens}\n")
    tokenizer.add_tokens(new_tokens)
    model_config = AutoConfig.from_pretrained(data_args.tokenizer_config_type)
    model_config.update({'vocab_size': len(tokenizer), "pre_norm": False, "activation_dropout":0.1, "gated_proj": False})
    py_logger.info(f"model config: {model_config}\n")
    encoder_model = Encoder(model_config)
    return encoder_model, tokenizer
