import os
import sys
import re
import pickle
import unicodedata
import os.path as osp
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional

import ujson
import pyarrow as pa
from pyarrow import feather
import multiprocessing as mp
from multiprocessing import Pool
from collections import Counter

from hytrel_modules.data import CAP_TAG, HEADER_TAG, ROW_TAG, MISSING_CAP_TAG, MISSING_CELL_TAG, MISSING_HEADER_TAG 

# Constants for maximum lengths
MAX_ROW_LEN = 100  # Maximum number of rows to process
MAX_COL_LEN = 100  # Maximum number of columns to process
MAX_WORD_LEN = 128  # Maximum word length for tokenization

def clean_wiki_template(text: str) -> str:
    """
    Clean Wikipedia-style templates from text.

    Args:
        text (str): Input text potentially containing Wikipedia templates.

    Returns:
        str: Cleaned text with templates removed or simplified.
    """
    # If the entire text is a template, extract the last part after '|'
    if re.match(r'^{{.*}}$', text):
        text = text[2:-2].split('|')[-1]  
    else:
        # Otherwise, remove all templates
        text = re.sub(r'{{.*}}', '', text)

    return text

def sanitize_text(text: str, entity: str = "cell", replace_missing: bool = True) -> str:
    """
    Clean up text in a table to ensure that it doesn't accidentally
    contain special table tokens or tags.

    Args:
        text (str): Raw string for one cell in the table.
        entity (str): Type of entity (cell, header, or caption). Defaults to "cell".
        replace_missing (bool): Whether to replace missing values with special tokens. Defaults to True.

    Returns:
        str: The sanitized cell string.
    """
    # Replace multiple '|' characters with a single space
    rval = re.sub(r"\|+", " ", text).strip()
    # Replace multiple whitespace characters with a single space
    rval = re.sub(r'\s+', ' ', rval).strip()
    
    # Remove 'td' or 'th' from the beginning of the text
    if rval and rval.split()[0] in ['td', 'th', 'TD', 'TH']:
        rval = ' '.join(rval.split()[1:])

    # Remove any special tags
    rval = rval.replace(CAP_TAG, "").replace(HEADER_TAG, "").replace(ROW_TAG, "")
    rval = rval.replace(MISSING_CAP_TAG , "").replace(MISSING_CELL_TAG, "").replace(MISSING_HEADER_TAG, "")
    
    # Truncate to maximum word length
    rval = ' '.join(rval.strip().split()[:MAX_WORD_LEN])

    # Replace empty or "missing" values with appropriate tokens
    if (rval == "" or rval.lower() == "<missing>" or rval.lower() == "missing") and replace_missing:
        if entity == "cell":
            rval = MISSING_CELL_TAG
        elif entity == "header":
            rval = MISSING_HEADER_TAG
        else:
            rval = MISSING_CAP_TAG

    return rval

def clean_cell_value(cell_val: Any) -> str:
    """
    Clean and normalize a cell value.

    Args:
        cell_val (Any): The input cell value, which can be a string, list, or other type.

    Returns:
        str: The cleaned and normalized cell value as a string.
    """
    # Convert list to string if necessary
    if isinstance(cell_val, list):
        val = ' '.join(cell_val)
    else:
        val = str(cell_val)
    
    # Normalize Unicode characters
    val = unicodedata.normalize('NFKD', val)
    # Convert to ASCII, ignoring non-ASCII characters
    val = val.encode('ascii', errors='ignore')
    val = str(val, encoding='ascii')
    
    # Clean Wikipedia templates
    val = clean_wiki_template(val)
    # Remove extra whitespace
    val = re.sub(r'\s+', ' ', val).strip()
    # Sanitize the text
    val = sanitize_text(val)
    
    return val

def read_json(name: str, all_texts: List[str], all_lower_heads: List[str], all_lower_cells: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Read and process JSON data from a file.

    Args:
        name (str): Path to the JSON file.
        all_texts (List[str]): List to store processed text data.
        all_lower_heads (List[str]): List to store lowercase headers.
        all_lower_cells (List[str]): List to store lowercase cell values.

    Returns:
        Tuple[List[str], List[str], List[str]]: Updated lists of texts, headers, and cells.
    """
    with open(name, 'r') as f:
        for line in tqdm(f, desc='Loading Tables...', unit=' entries', file=sys.stdout):
            smpl = ujson.loads(line)
            result = json2string(smpl)
            if result is not None:
                text, lower_heads, lower_cells = result
                all_texts.append(text)
                all_lower_heads.extend(lower_heads)
                all_lower_cells.extend(lower_cells)
    return all_texts, all_lower_heads, all_lower_cells

def json2string(exm: Dict[str, Any]) -> Optional[Tuple[str, List[str], List[str]]]:
    """
    Convert a JSON table representation to a string format.

    Args:
        exm (Dict[str, Any]): JSON representation of a table.

    Returns:
        Optional[Tuple[str, List[str], List[str]]]: 
        A tuple containing the processed text, lowercase headers, and lowercase cells,
        or None if processing fails.
    """
    lower_cells, lower_heads = [], []
    
    # Extract table data
    try:
        tb = exm['table']
    except KeyError:
        tb = exm
    
    # Process caption
    cap = '' if tb['caption'] is None else tb['caption']
    
    # Process headers (limit to MAX_COL_LEN)
    header = [h['name'] for h in tb['header']][:MAX_COL_LEN]
    data = tb['data']

    # Remove rows that duplicate header information
    while len(data):
        if (header[0] and isinstance(data[0][0], list) and (header[0] in ' '.join(data[0][0]))):
            data = data[1:]
        else:
            break
    if not len(data):
        return None

    # Limit data to MAX_ROW_LEN rows and MAX_COL_LEN columns
    data = [row[:MAX_COL_LEN] for row in data[:MAX_ROW_LEN]]

    # Sanitize text
    cap = sanitize_text(cap, entity='cap')
    header = [sanitize_text(h, entity='header') for h in header]
    lower_heads.extend([h.lower() if h!=MISSING_HEADER_TAG else MISSING_HEADER_TAG for h in header])
    header = ' | '.join(header)
    
    # Clean and process cell values
    cells = [list(map(clean_cell_value, row)) for row in data]
    lower_cells.extend([cell.lower() if cell!=MISSING_CELL_TAG else MISSING_CELL_TAG for row in cells for cell in row])
    cells = [' | '.join(row) for row in cells]
    
    # Construct final text representation
    text = ' '.join([CAP_TAG, cap, HEADER_TAG, header])
    cell_text = ' '.join([ROW_TAG + ' '.format(i) + row for i, row in enumerate(cells)])
    text = ' '.join([text, cell_text])
    
    return text, lower_heads, lower_cells

def preprocess():
    """
    Main preprocessing function to handle data cleaning and transformation.
    This function reads input data, processes it in parallel, and saves the results.
    """
    # Define input and output directories
    input_dir = osp.join('./data/pretrain/', 'chunks')
    output_dir = osp.join('./data/pretrain/', 'arrow')

    # Collect all input files
    files = []
    for dirpath, _, filenames in os.walk(input_dir):
        for f in filenames:
            files.append(osp.abspath(osp.join(dirpath, f)))

    # Set up multiprocessing
    mg = mp.Manager()
    all_texts = mg.list()
    all_lower_heads, all_lower_cells = mg.list(), mg.list()
    
    # Process files in parallel
    pool = Pool(processes=len(files))
    for name in files:
        pool.apply_async(read_json, args=(name, all_texts, all_lower_heads, all_lower_cells))
    pool.close()
    pool.join()

    # Serialize processed text data
    serialize_text_to_arrow(all_texts, folder=output_dir)

    print('Counting for ELECTRA pretraining....')
    # Create counters for headers and cells
    heads_counter = Counter(all_lower_heads)
    cells_counter = Counter(all_lower_cells)
    
    print('Storing...')
    # Save counters to files
    with open(osp.join(output_dir, 'heads_counter.pkl'), 'wb') as f:
        pickle.dump(heads_counter, f)

    with open(osp.join(output_dir, 'cells_counter.pkl'), 'wb') as f:
        pickle.dump(cells_counter, f)

def serialize_text_to_arrow(all_text: List[str], folder: str, split: Optional[str] = None):
    """
    Serialize processed text data to Arrow format.

    Args:
        all_text (List[str]): List of processed text data.
        folder (str): Output folder path.
        split (Optional[str]): Data split identifier (e.g., 'train', 'valid'). Defaults to None.
    """
    print("Total lines: ", len(all_text))
    print("Starting serializing data")

    # Define Arrow schema
    schema = pa.schema({'text': pa.large_string()})
    arr = pa.array(all_text, type=pa.large_string())
    pa_batches = pa.RecordBatch.from_arrays([arr], schema=schema)
    pa_table = pa.Table.from_batches([pa_batches], schema=schema)

    # Create output directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Define output filename
    if split is not None:
        output_name = folder + "/dataset" + "_" + str(split) + ".arrow"
    else:
        output_name = folder + "/dataset.arrow"
    
    # Write data to Arrow file
    feather.write_feather(pa_table, output_name)

if __name__ == "__main__":
    preprocess()