"""
HyTrel Model Configuration Module

This module sets up the HyTrel model configuration, including data arguments,
optimizer configuration, and model initialization.

References:
- HyTrel Implementation: https://github.com/awslabs/hypergraph-tabular-lm
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from transformers.optimization import AdamW, get_scheduler

from hytrel_modules.model import Encoder

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class DataArguments:
    """Arguments pertaining to data configuration."""
    tokenizer_config_type: str = field(
        default='bert-base-uncased',
        metadata={"help": "bert-base-cased, bert-base-uncased etc"}
    )
    data_path: str = field(
        default='../table_graph/data/col_rel/',
        metadata={"help": "data path"}
    )
    max_token_length: int = field(
        default=64,
        metadata={"help": "The maximum total input token length after tokenization."}
    )
    max_row_length: int = field(
        default=30,
        metadata={"help": "The maximum total input rows for a table"}
    )
    max_column_length: int = field(
        default=20,
        metadata={"help": "The maximum total input columns for a table"}
    )
    label_type_num: int = field(
        default=121,
        metadata={"help": "The total label types"}
    )
    num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of workers for dataloader"}
    )
    valid_ratio: float = field(
        default=0.3,
        metadata={"help": "Ratio of validation data"}
    )

    def __post_init__(self):
        if self.tokenizer_config_type not in ["bert-base-cased", "bert-base-uncased"]:
            raise ValueError(
                f"The model type should be bert-base-(un)cased. The current value is {self.tokenizer_config_type}."
            )

@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""
    batch_size: int = 256
    base_learning_rate: float = 1e-3
    weight_decay: float = 0.02
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-5
    lr_scheduler_type: str = "linear"
    warmup_step_ratio: float = 0.1
    seed: int = 42
    optimizer: str = "Adam"
    adam_w_mode: bool = True
    save_every_n_epochs: int = 1
    save_top_k: int = 1
    checkpoint_path: str = ''

    def __post_init__(self):
        valid_optimizers = {"adam", "fusedadam", "fusedlamb", "fusednovograd"}
        if self.optimizer.lower() not in valid_optimizers:
            raise ValueError(
                f"The optimizer type should be one of: {', '.join(valid_optimizers)}. "
                f"The current value is {self.optimizer}."
            )

    def get_optimizer(self, optim_groups, learning_rate: float) -> torch.optim.Optimizer:
        """Get the optimizer based on the configuration."""
        optimizer_class = AdamW if self.adam_w_mode else torch.optim.Adam
        optimizer = optimizer_class(
            optim_groups,
            lr=learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon
        )
        return optimizer

# def get_model() -> Tuple[Encoder, AutoTokenizer]:
#     """
#     Initialize and return the HyTrel model and tokenizer.

#     Returns:
#         Tuple[Encoder, AutoTokenizer]: The initialized model and tokenizer.
#     """
#     parser = HfArgumentParser((DataArguments, OptimizerConfig))
#     parser = pl.Trainer.add_argparse_args(parser)

#     data_args, optimizer_cfg, trainer_args = parser.parse_args_into_dataclasses()

#     logger.info(f"Data arguments: {data_args}")
#     logger.info(f"Optimizer configuration: {optimizer_cfg}")
#     logger.info(f"Trainer arguments: {trainer_args}")

#     # Set up tokenizer and model config
#     tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_config_type)
#     new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
#     tokenizer.add_tokens(new_tokens)
#     logger.info(f"New tokens added: {new_tokens}")

#     model_config = AutoConfig.from_pretrained(data_args.tokenizer_config_type)
#     model_config.update({
#         'vocab_size': len(tokenizer),
#         "pre_norm": False,
#         "activation_dropout": 0.1,
#         "gated_proj": False
#     })
#     logger.info(f"Model config: {model_config}")

#     encoder_model = Encoder(model_config)
#     return encoder_model, tokenizer

def get_model() -> Tuple[Encoder, AutoTokenizer]:
    """
    Initialize and return the HyTrel model and tokenizer.

    Returns:
        Tuple[Encoder, AutoTokenizer]: The initialized model and tokenizer.
    """
    parser = HfArgumentParser((DataArguments, OptimizerConfig))

    # Parse arguments into data arguments and optimizer configuration
    data_args, optimizer_cfg = parser.parse_args_into_dataclasses()

    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Optimizer configuration: {optimizer_cfg}")

    # Set up tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_config_type)
    new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
    tokenizer.add_tokens(new_tokens)
    logger.info(f"New tokens added: {new_tokens}")

    model_config = AutoConfig.from_pretrained(data_args.tokenizer_config_type)
    model_config.update({
        'vocab_size': len(tokenizer),
        "pre_norm": False,
        "activation_dropout": 0.1,
        "gated_proj": False
    })
    logger.info(f"Model config: {model_config}")

    encoder_model = Encoder(model_config)
    return encoder_model, tokenizer

if __name__ == "__main__":
    get_model()