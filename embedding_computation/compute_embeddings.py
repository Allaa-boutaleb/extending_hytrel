### This script was built based on HYTREL implementation.
### Ref: https://github.com/awslabs/hypergraph-tabular-lm 
### table processing to prepare table for the expected format for the model are exactly as the one provided in the implementatin. 
### We add preprocessing steps and sampling steps for our experiements 
import sys
import re
import json
import logging
import numpy as np
import os.path as osp
from tqdm import tqdm
import csv
import json
import pickle 

import torch
# import torch.nn as nn
# from torch.optim import Adam
# from torch.nn import BCEWithLogitsLoss
from configs.hytrel_model import OptimizerConfig

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# import transformers
from transformers.optimization import AdamW, get_scheduler
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

# from typing import Optional
# from model import Encoder
from hytrel_modules.data import BipartiteData ## file is a direct copy from the original file found in the repo https://github.com/awslabs/hypergraph-tabular-lm 
# from dataclasses import dataclass, field, fields
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
# from torch.optim import Adam
# from torch.nn import  CrossEntropyLoss
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

# from model import Encoder
# from data import BipartiteData
from hytrel_modules.parallel_clean import clean_cell_value ## file is a direct copy from the original file found in the repo https://github.com/awslabs/hypergraph-tabular-lm 
import time

### scripts utils 
from utils.table_sample import rows_remove_nulls, row_shuffle, rows_remove_nulls_max, priortize_rows_with_values, row_sort_by_tfidf, value_sort_columns, col_tfidf_sort, get_top_rows, pandas_sample, pandas_rate_sample, sample_columns_distinct
from utils.table_preprocess import make_headers_null,determine_delimiter
# import argparse
# import gzip
import configs.common as common_configs
import configs.hytrel_model as hytrel_model
from collections import OrderedDict



# #********************************* set up arguments *********************************

# @dataclass
# class DataArguments:
#     """
#     Arguments pertaining to which config/tokenizer we are going use.
#     """
#     tokenizer_config_type: str = field(
#         default='bert-base-uncased',
#         metadata={
#             "help": "bert-base-cased, bert-base-uncased etc"
#         },
#     )
#     data_path: str = field(default='../table_graph/data/col_rel/', metadata={"help": "data path"})
#     max_token_length: int = field(
#         default=64,
#         metadata={
#             "help": "The maximum total input token length for cell/caption/header after tokenization. Sequences longer "
#                     "than this will be truncated."
#         },
#     )
#     max_row_length: int = field(
#         default=30,
#         metadata={
#             "help": "The maximum total input rows for a table"
#         },
#     )
#     max_column_length: int = field(
#         default=20,
#         metadata={
#             "help": "The maximum total input columns for a table"

#         },
#     )
#     label_type_num: int = field(
#         default=121,
#         metadata={
#             "help": "The total label types"

#         },
#     )

#     num_workers: Optional[int] = field(
#         default=8,
#         metadata={"help": "Number of workers for dataloader"},
#     )

#     valid_ratio: float = field(
#         default=0.3,
#         metadata={"help": "Number of workers for dataloader"},
#     )


#     def __post_init__(self):
#         if self.tokenizer_config_type not in ["bert-base-cased", "bert-base-uncased"]:
#             raise ValueError(
#                 f"The model type should be bert-base-(un)cased. The current value is {self.tokenizer_config_type}."
#             )

# @dataclass
# class OptimizerConfig:
#     batch_size: int = 256
#     base_learning_rate: float = 1e-3
#     weight_decay: float = 0.02
#     adam_beta1: float = 0.9
#     adam_beta2: float = 0.98
#     adam_epsilon: float = 1e-5
#     lr_scheduler_type: transformers.SchedulerType = "linear"
#     warmup_step_ratio: float = 0.1
#     seed: int = 42
#     optimizer: str = "Adam"
#     adam_w_mode: bool = True
#     save_every_n_epochs: int=1
#     save_top_k: int=1
#     checkpoint_path: str=''


#     def __post_init__(self):
#         if self.optimizer.lower() not in {
#             "adam",
#             "fusedadam",
#             "fusedlamb",
#             "fusednovograd",
#         }:
#             raise KeyError(
#                 f"The optimizer type should be one of: Adam, FusedAdam, FusedLAMB, FusedNovoGrad. The current value is {self.optimizer}."
#             )

#     def get_optimizer(self, optim_groups, learning_rate):
#         optimizer = self.optimizer.lower()
#         optim_cls = {
#             "adam": AdamW if self.adam_w_mode else Adam,
#         }[optimizer]

#         args = [optim_groups]
#         kwargs = {
#             "lr": learning_rate,
#             "eps": self.adam_epsilon,
#             "betas": (self.adam_beta1, self.adam_beta2),
#         }
#         if optimizer in {"fusedadam", "fusedlamb"}:
#             kwargs["adam_w_mode"] = self.adam_w_mode

#         optimizer = optim_cls(*args, **kwargs)
#         return optimizer
  
# @dataclass
# class ScriptArguments:
#     input: str
#     output: str
#     col_output: str
#     run_id: int = field(default=0)
#     model_type: str = field(default='contrast')
#     table_process: str = field(default=None)
#     nrows: int = field(default=30) ## take 30 rows. 
#     embedding_only: bool = False
#     logs: str = field(default='/home/almutawa/hypergraph-tabular-lm/logs')
#     column_names: str = field(default=None)
#     pandas_sample: bool = False
#     pandas_rate_sample: bool = False
        
def load_model(ckpt, model):
    state_dict = torch.load(ckpt)
    new_state_dict = OrderedDict()
    for k, v in state_dict['module'].items():
        if 'model' in k:
            name = k[13:] # remove `module.model.`
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)

def extract_vectors(model,input_data):
    start_time = time.time()
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        embeddings = model(input_data[0])
    end_time = time.time()
    duration = end_time - start_time 
    return embeddings, duration

# added after a bug with \x7f
def remove_special_characters(text):
    return ''.join(char for char in text if ord(char) != 0x7f)

def extract_columns(embeddings,num_cols):
    return embeddings[1][1:num_cols+1]

def convert_csv_to_jsonl(df):
    json_data = {
        "id": None,
        "table": {
            "caption": None, ## will stay none 
            "header": None,
            "data": []
        }
    }
    json_data['id'] = "0" ## default value .. not tokenized 
    if common_configs.computation["handle_null_column_names"]:
        json_data['table']['header'] = [{"name": column.strip()} if "unnamed" not in column.strip().lower() else {"name":""} for column in df.columns]
    else: 
        json_data['table']['header'] = [{"name": column.strip()} for column in df.columns]
    
    for row in df.values:
        json_data['table']['data'].append(row.tolist())  
    return json_data

## same tokenization scheme as the one provided in data.py from github: https://github.com/awslabs/hypergraph-tabular-lm 
def _tokenize_word(word):

        # refer to numBERT: https://github.com/google-research/google-research/tree/master/numbert
        number_pattern = re.compile(
            r"(\d+)\.?(\d*)")  # Matches numbers in decimal form.
        def number_repl(matchobj):
            """Given a matchobj from number_pattern, it returns a string writing the corresponding number in scientific notation."""
            pre = matchobj.group(1).lstrip("0")
            post = matchobj.group(2)
            if pre and int(pre):
                # number is >= 1
                exponent = len(pre) - 1
            else:
                # find number of leading zeros to offset.
                exponent = -re.search("(?!0)", post).start() - 1
                post = post.lstrip("0")
            return (pre + post).rstrip("0") + " scinotexp " + str(exponent)
        
        def apply_scientific_notation(line):
            """Convert all numbers in a line to scientific notation."""
            res = re.sub(number_pattern, number_repl, line)
            return res
        
        word = clean_cell_value(word)
        word = apply_scientific_notation(word)        
        wordpieces = tokenizer.tokenize(word)[:64]

        mask = [1 for _ in range(len(wordpieces))]
        while len(wordpieces)<64:
            wordpieces.append('[PAD]')
            mask.append(0)
        return wordpieces, mask

### adjusted function to transform a given table to a graph format that hytrel would accept
### similar original function found in repository https://github.com/awslabs/hypergraph-tabular-lm 
def _table2graph(examples):
        data_list = []
        for exm in examples:
            try:
                tb = exm['table']
            except:
                tb = exm
            cap = '' if  tb['caption'] is None else tb['caption']
            cap = ' '.join(cap.split()[:64]) # filter too long caption
            header = [' '.join(h['name'].split()[:64]) for h in tb['header']][:]
            data = [row[:] for row in tb['data'][:]]
            assert len(data[0]) == len(header)
            
            wordpieces_xs_all, mask_xs_all = [], []
            wordpieces_xt_all, mask_xt_all = [], []
            nodes, edge_index = [], []
            # caption to hyper-edge (t node)
            if not cap:
                wordpieces = ['[TAB]'] + ['[PAD]' for _ in range(64 - 1)]
                mask = [1] + [0 for _ in range(64 - 1)]
                wordpieces_xt_all.append(wordpieces)
                mask_xt_all.append(mask)
            else:
                wordpieces, mask = _tokenize_word(cap)
                wordpieces_xt_all.append(wordpieces)
                mask_xt_all.append(mask)


            # header to hyper-edge (t node)
            for head in header:
                if not head:
                    wordpieces = ['[HEAD]'] + ['[PAD]' for _ in range(64 - 1)]
                    mask = [1] + [0 for _ in range(64 - 1)]
                    wordpieces_xt_all.append(wordpieces)
                    mask_xt_all.append(mask)
                else:
                    wordpieces, mask = _tokenize_word(head)
                    wordpieces_xt_all.append(wordpieces)
                    mask_xt_all.append(mask)


            # row to hyper edge (t node)
            for i in range(len(data)):
                wordpieces = ['[ROW]'] + ['[PAD]' for _ in range(64 - 1)]
                mask = [1] + [0 for _ in range(64 - 1)]
                wordpieces_xt_all.append(wordpieces)
                mask_xt_all.append(mask)
            for row_i, row in enumerate(data):
                for col_i, word in enumerate(row):
                    if not word:
                        wordpieces = ['[CELL]'] + ['[PAD]' for _ in range(64 - 1)]
                        mask = [1] + [0 for _ in range(64 - 1)]
                    else:
                        org_word = word
                        word = ' '.join(str(word).split()[:64])
                        word = remove_special_characters(word)
                        wordpieces, mask = _tokenize_word(word)
                        if sum(tokenizer.convert_tokens_to_ids(wordpieces)) == 0:
                            print(org_word)
                            print(row)
                            print(col_i)
                            print(row_i)
                    wordpieces_xs_all.append(wordpieces)
                    mask_xs_all.append(mask)
                    node_id = len(nodes)
                    nodes.append(node_id)
                    edge_index.append([node_id, 0]) # connect to table-level hyper-edge
                    edge_index.append([node_id, col_i+1]) # # connect to col-level hyper-edge
                    edge_index.append([node_id, row_i + 1 + len(header)])  # connect to row-level hyper-edge
            tab_mask = torch.zeros(len(wordpieces_xt_all), dtype=torch.long)
            tab_mask[0] = 1
                     
            xs_ids = torch.tensor([tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xs_all], dtype=torch.long)
            xt_ids = torch.tensor([tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xt_all], dtype=torch.long)
            

            # check all 0 input 
            xs_tem = torch.count_nonzero(xs_ids, dim =1)
            xt_tem = torch.count_nonzero(xt_ids, dim=1)
            

    
            assert torch.count_nonzero(xs_tem) == len(xs_tem)
            assert torch.count_nonzero(xt_tem) == len(xt_tem)
            edge_index = torch.tensor(edge_index, dtype=torch.long).T
            bigraph = BipartiteData(edge_index=edge_index, x_s=xs_ids, x_t=xt_ids)
            data_list.append(bigraph)
        return data_list
    
def get_headers_from_df(df):
    return df.columns.tolist()

def process_directory(ckpt,input,output,table_process,annotation,nrows,column_names):
    task = common_configs.global_params['downstream_task']
    run_id = common_configs.global_params['run_id']
    save_auxiliary = common_configs.computation['save_auxiliary']

     # ********************************* setup *********************************
    # Load pre-trained encoder model
    # py_logger = logging.getLogger(__name__)
    # py_logger.setLevel(logging.INFO)
    # # ********************************* parse arguments *********************************
    # parser = HfArgumentParser((DataArguments, OptimizerConfig))
    # parser = pl.Trainer.add_argparse_args(parser)

    # (
    #     data_args,
    #     optimizer_cfg,
    #     trainer_args,
    # ) = parser.parse_args_into_dataclasses()
    # py_logger.info(f"data_args: {data_args}\n")
    # py_logger.info(f"optimizer_cfg: {optimizer_cfg}\n")
    # py_logger.info(f"trainer_args: {trainer_args}\n")

    # # ********************************* set up tokenizer and model config*********************************
    # # custom BERT tokenizer and model config
    # global tokenizer 
    # tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_config_type)  
    # new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
    # py_logger.info(f"new tokens added: {new_tokens}\n")
    # tokenizer.add_tokens(new_tokens)
    # model_config = AutoConfig.from_pretrained(data_args.tokenizer_config_type)
    # model_config.update({'vocab_size': len(tokenizer), "pre_norm": False, "activation_dropout":0.1, "gated_proj": False})
    # py_logger.info(f"model config: {model_config}\n")
    # ********************************* setup ********************************* #
    global encoder_model
    # encoder_model = Encoder(model_config)

    encoder_model, tokenizer_ = hytrel_model.get_model()
    global tokenizer 
    tokenizer = tokenizer_
    load_model(ckpt, encoder_model)

    # ********************************* set directory ************************* #
    if save_auxiliary:
        aux = common_configs.output['auxiliary']
        mapping_directory = os.path.join(aux, 'mapping')
        jsonl_directory = os.path.join(aux,'jsonl')
        embedding_directory = os.path.join(aux,'embedding')
        # Create directories if it doesn't exist
        os.makedirs(mapping_directory, exist_ok=True)
        os.makedirs(jsonl_directory, exist_ok=True)
        os.makedirs(embedding_directory, exist_ok=True)

    print(f"--- process_directory for {input}: started ---")
    dataEmbeds = []
    ttl_time_to_get_embeddings = 0
    count_skipped_files = 0
    for filename in tqdm(os.listdir(input)):
        if filename.endswith('.csv') or filename.endswith('.CSV'):
            file_path = os.path.join(input, filename)
            print(f'processing: {file_path}')
            try:
                df = pd.read_csv(file_path)
            except pd.errors.ParserError as e:
                delimiter = determine_delimiter(file_path)
                print(f'delimiter: {delimiter}')
                df = pd.read_csv(file_path,delimiter=delimiter)
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError: Failed to read {file_path} with encoding. Trying a different encoding.")
                df = pd.read_csv(file_path, delimiter=delimiter, encoding='latin1')
            except pd.errors.ParserError as e:
                df = pd.read_csv(file_path, delimiter=delimiter, on_bad_lines='skip')

  
            headers = get_headers_from_df(df)
            dataset_name = filename.replace(".csv", "")
            if save_auxiliary: ## save the mapping
                mapping = {str(i+1): {'name': header, 'type':'column_embedding'} 
                        for i, header in enumerate(headers)}
                mapping.update({'0':{'name': 'table', 'type':'table_embedding'}})
                with open(os.path.join(mapping_directory, f'{dataset_name}.json'), 'w') as f:
                    json.dump(mapping, f, indent=4)
    ### applying sampling/data augmentation
            df_org = df.copy()
            if table_process == 'remove_rows_with_nulls':
                df = rows_remove_nulls_max(df)
            if table_process == 'prioritize_non_null_rows':
                df = priortize_rows_with_values(df)
            if table_process == 'remove_rows_with_nulls_and_shuffle': 
                df = rows_remove_nulls_max(df)
                df = row_shuffle(df)
            if table_process == 'sort_by_tfidf':
                df = row_sort_by_tfidf(df)
            if table_process == 'sort_by_tfidf_dropna':
                df = row_sort_by_tfidf(df,dropna=True)
            if table_process == 'value_based_sort_column':
                df = value_sort_columns(df)
            if table_process == 'sort_col_indepedent_tfidf_dropna':
                df = col_tfidf_sort(df)
            if table_process == 'sample_columns_distinct':
                df = sample_columns_distinct(df)
            if column_names == 'make_headers_null':
                df = make_headers_null(df)

            pandas_sample = common_configs.computation['pandas_sample']
            pandas_rate_sample = common_configs.computation['pandas_rate_sample']
            if pandas_sample or pandas_rate_sample: 
                if pandas_sample:
                    df = pandas_sample(df,nrows)
                elif pandas_rate_sample: 
                    df = pandas_rate_sample(df,nrows)
            else:
                df = get_top_rows(df,nrows)

            data = convert_csv_to_jsonl(df)
            if save_auxiliary: ## save the transformation from csv to jsonl 
                with open(os.path.join(jsonl_directory,f'{dataset_name}.jsonl'), 'w', encoding='utf-8') as jsonlfile:
                    jsonlfile.write(json.dumps(data) + '\n') 

            if len(df) != 0:
                input_data = _table2graph([data])
                embeddings, duration = extract_vectors(encoder_model,input_data)
                if save_auxiliary: ## save all embeddings
                    model_output = os.path.join(embedding_directory, f'{dataset_name}.pkl') 
                    with open(model_output, 'wb') as f:
                        pickle.dump(embeddings, f)
                ttl_time_to_get_embeddings = duration + ttl_time_to_get_embeddings
                cl_features_file = extract_columns(embeddings,len(headers))

                if task == 'union': ## save the embeddings of the directory in a single file depending on the task 
                    print('saving embeddings in union format')
                    dataEmbeds.append((filename,np.array(cl_features_file))) 
                elif task == 'join':
                    print('saving embeddings in join format')
                    for idx, column_name in enumerate(df_org.columns):
                        dataEmbeds.append(((filename, column_name), np.array(cl_features_file[idx])))

            else: ## for debugging purposes 
                count_skipped_files += 1
                logging.info(f'{file_path} is skipped due having at least 1 null cell in every row')
            
    os.makedirs(output,exist_ok=True)
    print(len(dataEmbeds))
    pkl_col_output = os.path.join(output,f'hytrel_{annotation}_columns_{run_id}.pkl')
    pickle.dump(dataEmbeds, open(pkl_col_output, "wb"))
    print(f"--- process_directory for {input}: c'est fini ---")
    print(f"--- time to retrieve embeddings from model for the {input} directory is {ttl_time_to_get_embeddings} seconds")
    logging.info(f'number of skipped files: {count_skipped_files}')
    logging.info(f'tome to retrieve embeddings from Hytrel of input {input}: {ttl_time_to_get_embeddings} seconds')
    logging.info(f'column vectors saved in {pkl_col_output}')
    
        
def main(): 

    # parser = HfArgumentParser(ScriptArguments)
    # hp = parser.parse_args_into_dataclasses()[0]
    ckpt = common_configs.global_params['hytrel_model']
    input = common_configs.input['source']
    input_type = common_configs.input['type']
    output = common_configs.output['vectors']
    table_process = common_configs.computation['table_process']
    nrows = common_configs.computation['nrows']
    log_directory = common_configs.computation['logs']
    log_file = common_configs.computation['log_file_name']
    column_names = common_configs.computation['column_names']

    os.makedirs(log_directory, exist_ok=True)
    log_file_path = os.path.join(log_directory, log_file)
    
    global logging 
    logging.basicConfig(
        filename=log_file_path, 
        filemode='a',  
        level=logging.DEBUG,  
        format='%(asctime)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S' 
    )
    logging.info(f'script configurations:\ninput: {input},\nvector output: {output}, \ntable_process: {table_process},\nnrows {nrows},\ncolumn_names: {column_names}')

    print(f'processing {input}')
    process_directory(ckpt,input,output,table_process,input_type,nrows,column_names)
    print(f'complete processing {input}')
    print(f'output saved in {output}')


if __name__ == '__main__':
    import warnings
    from pytorch_lightning import  seed_everything

    warnings.filterwarnings("ignore")
    seed = 42
    seed_everything(seed, workers=True)
    main()