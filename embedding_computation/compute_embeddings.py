"""
HyTrel Embedding Computation Module

This script computes embeddings using the HyTrel pre-trained model for tabular data.
It processes input data, applies various sampling and preprocessing techniques,
and generates embeddings for downstream tasks such as union and join operations.

References:
- HyTrel Implementation: https://github.com/awslabs/hypergraph-tabular-lm
"""

import sys
import re
import json
from loguru import logger
import os.path as osp
from typing import List, Tuple, Dict, Any
import csv
import pickle
import time
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from transformers import AutoTokenizer
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

import configs.common as common_configs
import configs.hytrel_model as hytrel_model
from hytrel_modules.data import BipartiteData
from utils.table_sample import (
    rows_remove_nulls, row_shuffle, rows_remove_nulls_max,
    priortize_rows_with_values, row_sort_by_tfidf, value_sort_columns,
    col_tfidf_sort, get_top_rows, pandas_sample, pandas_rate_sample,
    sample_columns_distinct
)
from utils.table_preprocess import make_headers_null, determine_delimiter


def load_model(ckpt: str, model: torch.nn.Module) -> None:
    """
    Load the pre-trained model weights.

    Args:
        ckpt (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the weights into.
    """
    state_dict = torch.load(ckpt)
    new_state_dict = OrderedDict()
    for k, v in state_dict['module'].items():
        if 'model' in k:
            name = k[13:]  # remove `module.model.`
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)

def extract_vectors(model: torch.nn.Module, input_data: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, float]:
    """
    Extract embeddings from the model.

    Args:
        model (torch.nn.Module): The pre-trained model.
        input_data (Tuple[torch.Tensor, ...]): Input data for the model.

    Returns:
        Tuple[torch.Tensor, float]: Extracted embeddings and computation time.
    """
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        embeddings = model(input_data[0])
    end_time = time.time()
    duration = end_time - start_time
    return embeddings, duration

def remove_special_characters(text: str) -> str:
    """
    Remove special characters from the text.

    Args:
        text (str): Input text.

    Returns:
        str: Text with special characters removed.
    """
    return ''.join(char for char in text if ord(char) != 0x7f)

def extract_columns(embeddings: torch.Tensor, num_cols: int) -> torch.Tensor:
    """
    Extract column embeddings from the model output.

    Args:
        embeddings (torch.Tensor): Model output embeddings.
        num_cols (int): Number of columns.

    Returns:
        torch.Tensor: Extracted column embeddings.
    """
    return embeddings[1][1:num_cols+1]

def convert_csv_to_jsonl(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert a pandas DataFrame to JSONL format.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Dict[str, Any]: JSONL formatted data.
    """
    json_data = {
        "id": "0",
        "table": {
            "caption": None,
            "header": [],
            "data": []
        }
    }
    
    if common_configs.computation["handle_null_column_names"]:
        json_data['table']['header'] = [{"name": column.strip()} if "unnamed" not in column.strip().lower() else {"name":""} for column in df.columns]
    else:
        json_data['table']['header'] = [{"name": column.strip()} for column in df.columns]
    
    json_data['table']['data'] = df.values.tolist()
    return json_data

def _tokenize_word(word: str, tokenizer: AutoTokenizer) -> Tuple[List[str], List[int]]:
    """
    Tokenize a word using the provided tokenizer.

    Args:
        word (str): Input word.
        tokenizer (AutoTokenizer): Tokenizer instance.

    Returns:
        Tuple[List[str], List[int]]: Tokenized word and mask.
    """
    number_pattern = re.compile(r"(\d+)\.?(\d*)")

    def number_repl(matchobj):
        pre = matchobj.group(1).lstrip("0")
        post = matchobj.group(2)
        if pre and int(pre):
            exponent = len(pre) - 1
        else:
            exponent = -re.search("(?!0)", post).start() - 1
            post = post.lstrip("0")
        return f"{pre}{post.rstrip('0')} scinotexp {exponent}"

    def apply_scientific_notation(line):
        return re.sub(number_pattern, number_repl, line)

    word = apply_scientific_notation(word)
    wordpieces = tokenizer.tokenize(word)[:64]
    mask = [1] * len(wordpieces) + [0] * (64 - len(wordpieces))
    wordpieces += ['[PAD]'] * (64 - len(wordpieces))
    return wordpieces, mask

def _table2graph(examples: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> List[BipartiteData]:
    """
    Convert table data to graph format.

    Args:
        examples (List[Dict[str, Any]]): List of table data.
        tokenizer (AutoTokenizer): Tokenizer instance.

    Returns:
        List[BipartiteData]: List of graph data.
    """
    data_list = []
    for exm in examples:
        tb = exm.get('table', exm)
        cap = tb.get('caption', '') or ''
        cap = ' '.join(cap.split()[:64])
        header = [' '.join(h['name'].split()[:64]) for h in tb['header']][:64]
        data = [row[:64] for row in tb['data'][:64]]
        
        wordpieces_xs_all, mask_xs_all = [], []
        wordpieces_xt_all, mask_xt_all = [], []
        nodes, edge_index = [], []
        
        # Process caption
        wordpieces, mask = _tokenize_word(cap, tokenizer) if cap else (['[TAB]'] + ['[PAD]'] * 63, [1] + [0] * 63)
        wordpieces_xt_all.append(wordpieces)
        mask_xt_all.append(mask)
        
        # Process header
        for head in header:
            wordpieces, mask = _tokenize_word(head, tokenizer) if head else (['[HEAD]'] + ['[PAD]'] * 63, [1] + [0] * 63)
            wordpieces_xt_all.append(wordpieces)
            mask_xt_all.append(mask)
        
        # Process rows
        for i in range(len(data)):
            wordpieces_xt_all.append(['[ROW]'] + ['[PAD]'] * 63)
            mask_xt_all.append([1] + [0] * 63)
        
        for row_i, row in enumerate(data):
            for col_i, word in enumerate(row):
                if not word:
                    wordpieces, mask = ['[CELL]'] + ['[PAD]'] * 63, [1] + [0] * 63
                else:
                    word = remove_special_characters(' '.join(str(word).split()[:64]))
                    wordpieces, mask = _tokenize_word(word, tokenizer)
                
                wordpieces_xs_all.append(wordpieces)
                mask_xs_all.append(mask)
                node_id = len(nodes)
                nodes.append(node_id)
                edge_index.extend([[node_id, 0], [node_id, col_i+1], [node_id, row_i + 1 + len(header)]])
        
        xs_ids = torch.tensor([tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xs_all], dtype=torch.long)
        xt_ids = torch.tensor([tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xt_all], dtype=torch.long)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        bigraph = BipartiteData(edge_index=edge_index, x_s=xs_ids, x_t=xt_ids)
        data_list.append(bigraph)
    
    return data_list

def get_headers_from_df(df: pd.DataFrame) -> List[str]:
    """
    Get column headers from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        List[str]: List of column headers.
    """
    return df.columns.tolist()

def process_directory(ckpt: str, input_path: str, output_path: str, table_process: str,
                      annotation: str, nrows: int, column_names: str) -> None:
    """
    Process the input directory and compute embeddings.

    Args:
        ckpt (str): Path to the checkpoint file.
        input_path (str): Path to the input directory.
        output_path (str): Path to the output directory.
        table_process (str): Type of table processing to apply.
        annotation (str): Annotation type (e.g., 'query' or 'datalake').
        nrows (int): Number of rows to process.
        column_names (str): Column names processing option.
    """
    task = common_configs.global_params['downstream_task']
    run_id = common_configs.global_params['run_id']
    save_auxiliary = common_configs.computation['save_auxiliary']

    encoder_model, tokenizer = hytrel_model.get_model()
    load_model(ckpt, encoder_model)

    if save_auxiliary:
        aux = common_configs.output['auxiliary']
        mapping_directory = osp.join(aux, 'mapping')
        jsonl_directory = osp.join(aux, 'jsonl')
        embedding_directory = osp.join(aux, 'embedding')
        os.makedirs(mapping_directory, exist_ok=True)
        os.makedirs(jsonl_directory, exist_ok=True)
        os.makedirs(embedding_directory, exist_ok=True)

    logger.info(f"Processing directory: {input_path}")
    data_embeds = []
    total_embedding_time = 0
    skipped_files_count = 0

    for filename in tqdm(os.listdir(input_path)):
        if filename.lower().endswith('.csv'):
            file_path = osp.join(input_path, filename)
            logger.info(f'Processing: {file_path}')

            try:
                df = pd.read_csv(file_path)
            except pd.errors.ParserError:
                delimiter = determine_delimiter(file_path)
                logger.info(f'Delimiter: {delimiter}')
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter)
                except UnicodeDecodeError:
                    logger.warning(f"UnicodeDecodeError: Failed to read {file_path}. Trying a different encoding.")
                    df = pd.read_csv(file_path, delimiter=delimiter, encoding='latin1')
                except pd.errors.ParserError:
                    df = pd.read_csv(file_path, delimiter=delimiter, on_bad_lines='skip')

            headers = get_headers_from_df(df)
            dataset_name = filename.replace(".csv", "")

            if save_auxiliary:
                mapping = {str(i+1): {'name': header, 'type':'column_embedding'} 
                           for i, header in enumerate(headers)}
                mapping['0'] = {'name': 'table', 'type':'table_embedding'}
                with open(osp.join(mapping_directory, f'{dataset_name}.json'), 'w') as f:
                    json.dump(mapping, f, indent=4)

            df_org = df.copy()
            df = apply_table_process(df, table_process)
            df = apply_sampling(df, nrows)

            if column_names == 'make_headers_null':
                df = make_headers_null(df)

            data = convert_csv_to_jsonl(df)

            if save_auxiliary:
                with open(osp.join(jsonl_directory, f'{dataset_name}.jsonl'), 'w', encoding='utf-8') as jsonlfile:
                    json.dump(data, jsonlfile)

            if len(df) != 0:
                input_data = _table2graph([data], tokenizer)
                embeddings, duration = extract_vectors(encoder_model, input_data)
                
                if save_auxiliary:
                    model_output = osp.join(embedding_directory, f'{dataset_name}.pkl')
                    with open(model_output, 'wb') as f:
                        pickle.dump(embeddings, f)
                
                total_embedding_time += duration
                cl_features_file = extract_columns(embeddings, len(headers))

                if task == 'union':
                    logger.info('Saving embeddings in union format')
                    data_embeds.append((filename, np.array(cl_features_file)))
                elif task == 'join':
                    logger.info('Saving embeddings in join format')
                    for idx, column_name in enumerate(df_org.columns):
                        data_embeds.append(((filename, column_name), np.array(cl_features_file[idx])))
            else:
                skipped_files_count += 1
                logger.info(f'{file_path} is skipped due to having at least 1 null cell in every row')

    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Number of processed files: {len(data_embeds)}")
    pkl_col_output = osp.join(output_path, f'hytrel_{annotation}_columns_{run_id}.pkl')
    with open(pkl_col_output, "wb") as f:
        pickle.dump(data_embeds, f)

    logger.info(f"Processing completed for {input_path}")
    logger.info(f"Total time to retrieve embeddings: {total_embedding_time:.2f} seconds")
    logger.info(f"Number of skipped files: {skipped_files_count}")
    logger.info(f"Column vectors saved in {pkl_col_output}")

def apply_table_process(df: pd.DataFrame, table_process: str) -> pd.DataFrame:
    """
    Apply the specified table processing technique to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        table_process (str): Type of table processing to apply.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    if table_process == 'remove_rows_with_nulls':
        return rows_remove_nulls_max(df)
    elif table_process == 'prioritize_non_null_rows':
        return priortize_rows_with_values(df)
    elif table_process == 'remove_rows_with_nulls_and_shuffle':
        return row_shuffle(rows_remove_nulls_max(df))
    elif table_process == 'sort_by_tfidf':
        return row_sort_by_tfidf(df)
    elif table_process == 'sort_by_tfidf_dropna':
        return row_sort_by_tfidf(df, dropna=True)
    elif table_process == 'value_based_sort_column':
        return value_sort_columns(df)
    elif table_process == 'sort_col_indepedent_tfidf_dropna':
        return col_tfidf_sort(df)
    elif table_process == 'sample_columns_distinct':
        return sample_columns_distinct(df)
    else:
        return df

def apply_sampling(df: pd.DataFrame, nrows: int) -> pd.DataFrame:
    """
    Apply sampling to the DataFrame based on configuration.

    Args:
        df (pd.DataFrame): Input DataFrame.
        nrows (int): Number of rows to sample.

    Returns:
        pd.DataFrame: Sampled DataFrame.
    """
    if common_configs.computation['pandas_sample']:
        return pandas_sample(df, nrows)
    elif common_configs.computation['pandas_rate_sample']:
        return pandas_rate_sample(df, nrows)
    else:
        return get_top_rows(df, nrows)

def main():
    """
    Main function to run the embedding computation process.
    """
    ckpt = common_configs.global_params['hytrel_model']
    input_path = common_configs.input['source']
    input_type = common_configs.input['type']
    output_path = common_configs.output['vectors']
    table_process = common_configs.computation['table_process']
    nrows = common_configs.computation['nrows']
    log_directory = common_configs.computation['logs']
    log_file = common_configs.computation['log_file_name']
    column_names = common_configs.computation['column_names']

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(osp.join(log_directory, log_file), rotation="10 MB")
    logger.info(f"Script configurations:")
    logger.info(f"Input: {input_path}")
    logger.info(f"Vector output: {output_path}")
    logger.info(f"Table process: {table_process}")
    logger.info(f"Number of rows: {nrows}")
    logger.info(f"Column names: {column_names}")

    logger.info(f"Processing {input_path}")
    process_directory(ckpt, input_path, output_path, table_process, input_type, nrows, column_names)
    logger.info(f"Processing completed for {input_path}")
    logger.info(f"Output saved in {output_path}")

if __name__ == '__main__':
    main()