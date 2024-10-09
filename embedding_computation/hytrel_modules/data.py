import re
import pickle
import random
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import pyarrow as pa
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass

# Constants
CAP_TAG = "<caption>"
HEADER_TAG = "<header>"
ROW_TAG = "<row>"

MISSING_CAP_TAG = '[TAB]'
MISSING_CELL_TAG = "[CELL]"
MISSING_HEADER_TAG = "[HEAD]"

@dataclass
class NumpyDataset(Dataset):
    """A dataset wrapper for numpy arrays."""
    array: np.ndarray

    def __getitem__(self, index: int) -> Any:
        return self.array[index]

    def __len__(self) -> int:
        return len(self.array)

@dataclass
class ArrowDataset(Dataset):
    """A dataset wrapper for Arrow ChunkedArrays."""
    array: pa.ChunkedArray

    def __getitem__(self, index: int) -> Any:
        return self.array[index].as_py()

    def __len__(self) -> int:
        return len(self.array)

class BipartiteData(Data):
    """
    A class representing bipartite graph data.

    Attributes:
        edge_index (torch.Tensor): Edge indices.
        x_s (torch.Tensor): Source node features.
        x_t (torch.Tensor): Target node features.
    """
    def __init__(self, edge_index: Optional[torch.Tensor] = None, 
                 x_s: Optional[torch.Tensor] = None, 
                 x_t: Optional[torch.Tensor] = None, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Union[int, torch.Tensor]: 
        if key in ['edge_index', 'corr_edge_index', 'edge_index_corr1', 'edge_index_corr2']:
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

class TableDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for processing tabular data.

    This class handles data loading, preprocessing, and batching for the HyTrel model.
    """
    def __init__(
            self,
            tokenizer: Any,
            data_args: Any,
            seed: int,
            batch_size: int,
            py_logger: Any,
            objective: str):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.seed = seed
        self.batch_size = batch_size
        self.objective = objective
        self.py_logger = py_logger
        self.py_logger.info("Using tabular data collator")
        self.samples = []

    def setup(self, stage: Optional[str] = None):
        """
        Prepare data for training, validation, and testing.

        This method loads the dataset, splits it into train and validation sets,
        and prepares indices for shuffling during training.
        """
        self.py_logger.info(f"Preparing data... \n")
        self.py_logger.info("Create memory map\n")

        mmap = pa.memory_map(self.data_args.data_path + '/arrow/dataset.arrow')
        self.py_logger.info("MMAP Read ALL")
        self.dataset = (pa.ipc.open_file(mmap).read_all()).combine_chunks()
        
        self.py_logger.info(f"Loaded dataset... \n")
        self.data_lengths = len(self.dataset)  
        self.py_logger.info("Dataset size {}".format(self.data_lengths))
        assert len(self.dataset["text"].chunks) == 1

        # Split dataset into train and validation
        rng = np.random.default_rng(seed=42)
        dataset_indices = np.arange(0, self.data_lengths)
        self.py_logger.info(f"shuffle indices and split data... ")
        rng.shuffle(dataset_indices)
        valid_size = int(self.data_lengths * self.data_args.valid_ratio)
        valid_dataset_indices = dataset_indices[:valid_size]
        train_dataset_indices = dataset_indices[valid_size:]

        self.py_logger.info("First 100 train data ids: {}".format(' '.join(map(str, train_dataset_indices[:100]))))
        self.py_logger.info("First 100 valid data ids: {}".format(' '.join(map(str, valid_dataset_indices[:100]))))
        
        self.valid_dataset = self.dataset.take(valid_dataset_indices)
        self.train_dataset = self.dataset.take(train_dataset_indices)

        self.py_logger.info(
            "Finished taking indices: train size {}, valid size {}".format(len(self.train_dataset),len(self.valid_dataset))
        )

        # Prepare shuffled indices for each epoch
        self.py_logger.info("Shuffling training epoch dataset indices.... ")
        self.shuffled_train_indices_by_epochs = []
        initial_arr = list(range(len(self.train_dataset)))
        for idx in range(self.data_args.max_epoch):
            self.py_logger.info(f"Create train dataset indices slot {idx}")
            np.random.shuffle(initial_arr)
            self.shuffled_train_indices_by_epochs.append(initial_arr)

        if self.objective == 'electra':
            self._load_electra_data()

    def _load_electra_data(self):
        """Load and prepare data for ELECTRA training objective."""
        self.py_logger.info("Loading lexicon counter.... ")
        with open(self.data_args.data_path + '/arrow/heads_counter.pkl', 'rb') as f:
            self.heads_counter = pickle.load(f)
        with open(self.data_args.data_path + '/arrow/cells_counter.pkl', 'rb') as f:
            self.cells_counter = pickle.load(f)

        # Remove missing values from counters
        self.heads_counter[MISSING_HEADER_TAG] = 0
        self.cells_counter[MISSING_CELL_TAG] = 0
        
        # Keep only the top 5% of tokens
        self.heads_counter = dict(self.heads_counter.most_common(int(0.05*len(self.heads_counter))))
        self.cells_counter = dict(self.cells_counter.most_common(int(0.05*len(self.cells_counter))))
        
        # Prepare arrays for efficient sampling
        self.heads_counter_keys = np.asarray(list(self.heads_counter.keys()))
        self.heads_counter_values = np.asarray(list(self.heads_counter.values()))
        self.heads_counter_values = self.heads_counter_values/np.sum(self.heads_counter_values)
        
        self.cells_counter_keys = np.asarray(list(self.cells_counter.keys()))
        self.cells_counter_values = np.asarray(list(self.cells_counter.values()))
        self.cells_counter_values = self.cells_counter_values/np.sum(self.cells_counter_values)
        
        self.py_logger.info(
            "Head counter size {}, cell counter size {}".format(len(self.heads_counter),len(self.cells_counter))
        )

    def _tokenize_word(self, word: str) -> Tuple[List[str], List[int]]:
        """
        Tokenize a word and apply scientific notation to numbers.

        Args:
            word (str): The input word to tokenize.

        Returns:
            Tuple[List[str], List[int]]: Tokenized word and its mask.
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
            return (pre + post).rstrip("0") + " scinotexp " + str(exponent)
        
        def apply_scientific_notation(line):
            return re.sub(number_pattern, number_repl, line)
        
        word = apply_scientific_notation(word)        
        wordpieces = self.tokenizer.tokenize(word)[:self.data_args.max_token_length]

        mask = [1 for _ in range(len(wordpieces))]
        while len(wordpieces) < self.data_args.max_token_length:
            wordpieces.append('[PAD]')
            mask.append(0)
        return wordpieces, mask

    def _text2table(self, sample: str) -> Tuple[str, List[str], List[List[str]]]:
        """
        Convert a text sample to table format.

        Args:
            sample (str): The input text sample.

        Returns:
            Tuple[str, List[str], List[List[str]]]: Caption, headers, and cells of the table.
        """
        smpl = sample.split(HEADER_TAG)
        cap = smpl[0].replace(CAP_TAG, '').strip()
        smpl = smpl[1].split(ROW_TAG)
        headers = [h.strip() for h in smpl[0].strip().split(' | ')]
        cells = [list(map(lambda x: x.strip(), row.strip().split(' | '))) for row in smpl[1:]]
        for row in cells:
            assert len(row) == len(headers)

        return cap, headers, cells

    def _electra_corrupt(self, header: List[str], data: List[List[str]]) -> Tuple[List[str], List[List[str]], List[int], List[Tuple[int, int]]]:
        """
        Corrupt headers and cells for ELECTRA training.

        Args:
            header (List[str]): List of column headers.
            data (List[List[str]]): Table data.

        Returns:
            Tuple[List[str], List[List[str]], List[int], List[Tuple[int, int]]]: 
            Corrupted headers, corrupted data, corrupted header indices, corrupted cell indices.
        """
        h_length, c_length = len(header), len(data)*len(data[0])
        h_count = round(h_length*self.data_args.mask_ratio)
        c_count = round(c_length*self.data_args.mask_ratio)

        h_samples = [' '.join(h.split()[:self.data_args.max_token_length])  for h in np.random.choice(self.heads_counter_keys, p=self.heads_counter_values, size=h_count)]
        c_samples = [' '.join(c.split()[:self.data_args.max_token_length]) for c in np.random.choice(self.cells_counter_keys, p=self.cells_counter_values, size=c_count)]

        h_idx = random.sample(range(h_length), h_count)
        c_idx = random.sample(range(c_length), c_count)
        c_idx_grid = [(i//h_length, i%h_length) for i in c_idx]

        h_idx = [i for i in h_idx if header[i]!=MISSING_HEADER_TAG]
        h_corrupt_dict = dict(zip(h_idx, h_samples[:len(h_idx)]))
        header = [h_corrupt_dict[i] if i in h_corrupt_dict else h for i,h in enumerate(header)]

        c_corrupt_dict = dict(zip(c_idx_grid, c_samples))
        for row_i in range(len(data)):
            for col_j in range(len(data[0])):
                if (row_i, col_j) in c_corrupt_dict:
                    if data[row_i][col_j] == MISSING_CELL_TAG:
                        c_idx_grid.remove((row_i, col_j))
                        c_idx.remove(row_i*h_length+col_j)
                    else:
                        data[row_i][col_j] = c_corrupt_dict[(row_i, col_j)]

        return header, data, h_idx, c_idx_grid
    
    def _contrast_corrupt(self, header: List[str], data: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        """
        Corrupt headers and cells for contrastive learning.

        Args:
            header (List[str]): List of column headers.
            data (List[List[str]]): Table data.

        Returns:
            Tuple[List[str], List[List[str]]]: Corrupted headers and data.
        """
        h_length, c_length = len(header), len(data)*len(data[0])
        h_count = round(h_length*self.data_args.node_corrupt_ratio)
        c_count = round(c_length*self.data_args.node_corrupt_ratio)

        h_idx = random.sample(range(h_length), h_count)
        c_idx = random.sample(range(c_length), c_count)
        c_idx_grid = [(i//h_length, i%h_length) for i in c_idx]

        h_idx = [i for i in h_idx if header[i]!=MISSING_HEADER_TAG]
        header = [MISSING_HEADER_TAG if i in h_idx else h for i,h in enumerate(header)]

        for row_i in range(len(data)):
            for col_j in range(len(data[0])):
                if (row_i, col_j) in c_idx_grid:
                    if data[row_i][col_j] == MISSING_CELL_TAG:
                        c_idx_grid.remove((row_i, col_j))
                        c_idx.remove(row_i*h_length+col_j)
                    else:
                        data[row_i][col_j] = MISSING_CELL_TAG

        return header, data
    
    def _construct_graph(self, cap: str, header: List[str], data: List[List[str]], 
                         h_corr_idx: Optional[List[int]] = None, 
                         c_corr_idx_grid: Optional[List[Tuple[int, int]]] = None) -> Union[
                             Tuple[List[List[str]], List[List[str]], List[List[int]]],
                             Tuple[List[List[str]], List[List[str]], List[List[int]], torch.Tensor, torch.Tensor]
                         ]:
        """
        Construct a graph representation of the table.

        Args:
            cap (str): Table caption.
            header (List[str]): List of column headers.
            data (List[List[str]]): Table data.
            h_corr_idx (Optional[List[int]]): Indices of corrupted headers for ELECTRA.
            c_corr_idx_grid (Optional[List[Tuple[int, int]]]): Indices of corrupted cells for ELECTRA.

        Returns:
            Union[Tuple[List[List[str]], List[List[str]], List[List[int]]], 
                  Tuple[List[List[str]], List[List[str]], List[List[int]], torch.Tensor, torch.Tensor]]:
            Tokenized source nodes, target nodes, edge indices, and optionally ELECTRA labels.
        """
        wordpieces_xs_all, mask_xs_all = [], []
        wordpieces_xt_all, mask_xt_all = [], []
        nodes, edge_index = [], []
        
        if self.data_args.electra and h_corr_idx is not None:                
            h_length, c_length = len(header), len(data) * len(data[0])
            xs_lbl = torch.zeros(c_length, dtype=torch.float32)
            xt_lbl = torch.zeros(h_length, dtype=torch.float32)

        # Process caption
        if cap == MISSING_CAP_TAG:
            wordpieces = [cap] + ['[PAD]' for _ in range(self.data_args.max_token_length - 1)]
            mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
            wordpieces_xt_all.append(wordpieces)
            mask_xt_all.append(mask)
        else:
            wordpieces, mask = self._tokenize_word(cap)
            wordpieces_xt_all.append(wordpieces)
            mask_xt_all.append(mask)

        # Process headers
        for i, head in enumerate(header):
            if head == MISSING_HEADER_TAG:
                wordpieces = [head] + ['[PAD]' for _ in range(self.data_args.max_token_length - 1)]
                mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                wordpieces_xt_all.append(wordpieces)
                mask_xt_all.append(mask)
            else:
                wordpieces, mask = self._tokenize_word(head)
                wordpieces_xt_all.append(wordpieces)
                mask_xt_all.append(mask)

            if self.data_args.electra and h_corr_idx is not None:
                if i in h_corr_idx:
                    xt_lbl[i] = 1.

        # Process rows
        for i in range(len(data)):
            wordpieces = ['[ROW]'] + ['[PAD]' for _ in range(self.data_args.max_token_length - 1)]
            mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
            wordpieces_xt_all.append(wordpieces)
            mask_xt_all.append(mask)

        # Process cells
        for row_i, row in enumerate(data):
            for col_i, word in enumerate(row):
                if word == MISSING_CELL_TAG:
                    wordpieces = [word] + ['[PAD]' for _ in range(self.data_args.max_token_length - 1)]
                    mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                else:
                    word = ' '.join(word.split()[:self.data_args.max_token_length])
                    wordpieces, mask = self._tokenize_word(word)

                wordpieces_xs_all.append(wordpieces)
                mask_xs_all.append(mask)
                node_id = len(nodes)
                nodes.append(node_id)
                edge_index.append([node_id, 0])  # connect to table-level hyper-edge
                edge_index.append([node_id, col_i+1])  # connect to col-level hyper-edge
                edge_index.append([node_id, row_i + 1 + len(header)])  # connect to row-level hyper-edge

                if self.data_args.electra and h_corr_idx is not None:
                    if (row_i, col_i) in c_corr_idx_grid:
                        xs_lbl[row_i*h_length+col_i] = 1.

        assert len(nodes) == len(wordpieces_xs_all)
        assert (len(header)+len(data)+1) == len(wordpieces_xt_all)
        
        if self.data_args.electra and h_corr_idx is not None:
            return wordpieces_xs_all, wordpieces_xt_all, edge_index, xs_lbl, xt_lbl 
        else: 
            return wordpieces_xs_all, wordpieces_xt_all, edge_index

    def _text2graph(self, samples: List[str]) -> List[BipartiteData]:
        """
        Convert text samples to graph data.

        Args:
            samples (List[str]): List of text samples representing tables.

        Returns:
            List[BipartiteData]: List of graph data representations of the tables.
        """
        data_list = []
        for smpl in samples:  
            try:
                cap, headers, data = self._text2table(smpl)
            except:
                print('Fail to parse the table...')
                continue

            cap = ' '.join(cap.split()[:self.data_args.max_token_length])
            header = [' '.join(h.split()[:self.data_args.max_token_length]) for h in headers][:self.data_args.max_column_length]
            data = [row[:self.data_args.max_column_length] for row in data[:self.data_args.max_row_length]]
            
            assert len(header) <= self.data_args.max_column_length
            assert len(data[0]) == len(header)
            assert len(data) <= self.data_args.max_row_length
            
            h_length, c_length = len(header), len(data) * len(data[0])

            if self.data_args.electra:
                header, data, h_corr_idx, c_corr_idx_grid = self._electra_corrupt(header, data)
                wordpieces_xs_all, wordpieces_xt_all, edge_index, xs_lbl, xt_lbl = self._construct_graph(cap, header, data, h_corr_idx, c_corr_idx_grid)
            else:
                wordpieces_xs_all, wordpieces_xt_all, edge_index = self._construct_graph(cap, header, data)
                    
            xs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xs_all], dtype=torch.long)
            xt_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xt_all], dtype=torch.long)

            # Check for all-zero input exceptions
            try:
                xs_tem = torch.count_nonzero(xs_ids, dim=1)
                xt_tem = torch.count_nonzero(xt_ids, dim=1)
                assert torch.count_nonzero(xs_tem) == len(xs_tem)
                assert torch.count_nonzero(xt_tem) == len(xt_tem)
            except:
                print('All 0 input exists!')
                continue

            edge_index = torch.tensor(edge_index, dtype=torch.long).T 

            if self.data_args.contrast_bipartite_edge:
                corr_idx1 = torch.randperm(edge_index.size(1))[:int(edge_index.size(1)*(1-self.data_args.bipartite_edge_corrupt_ratio))]
                corr_idx2 = torch.randperm(edge_index.size(1))[:int(edge_index.size(1)*(1-self.data_args.bipartite_edge_corrupt_ratio))]
                edge_index_corr1 = torch.index_select(edge_index, 1, corr_idx1)
                edge_index_corr2 = torch.index_select(edge_index, 1, corr_idx2)
                hyper_mask = torch.zeros(len(wordpieces_xt_all), dtype=torch.long)
                hyper_mask[:1 + h_length] = torch.ones(h_length+1, dtype=torch.long)

            if self.data_args.electra:
                col_mask = torch.zeros(len(wordpieces_xt_all), dtype=torch.long)
                col_mask[1:1 + h_length] = torch.ones(h_length, dtype=torch.long)
                assert len(xs_lbl) == len(wordpieces_xs_all)
                assert len(xt_lbl) == len(wordpieces_xt_all) - 1 - len(data)

            if self.data_args.electra:
                bigraph = BipartiteData(edge_index=edge_index, x_s=xs_ids, x_t=xt_ids, electra_c=xs_lbl, electra_h=xt_lbl, col_mask=col_mask, num_nodes=len(xs_ids))    
            elif self.data_args.contrast_bipartite_edge:
                bigraph = BipartiteData(edge_index=edge_index, edge_index_corr1=edge_index_corr1, edge_index_corr2=edge_index_corr2, x_s=xs_ids, x_t=xt_ids, hyper_mask=hyper_mask, num_nodes=len(xs_ids))
            else:
                bigraph = BipartiteData(edge_index=edge_index, x_s=xs_ids, x_t=xt_ids, num_nodes=len(xs_ids))

            data_list.append(bigraph)

        return data_list

    def train_collate_fn_(self, indices: List[int]) -> Batch:
        """
        Collate function for training data.

        Args:
            indices (List[int]): List of indices to sample from the training dataset.

        Returns:
            Batch: A batch of graph data.
        """
        samples = self.train_dataset.take(indices)["text"].to_pylist()
        graphs = self._text2graph(samples)
        batch = Batch.from_data_list(graphs)
        return batch

    def eval_collate_fn_(self, samples: List[str]) -> Batch:
        """
        Collate function for evaluation data.

        Args:
            samples (List[str]): List of text samples to process.

        Returns:
            Batch: A batch of graph data.
        """
        graphs = self._text2graph(samples)
        batch = Batch.from_data_list(graphs)
        return batch

    def train_dataloader(self) -> DataLoader:
        """
        Create the training data loader.

        Returns:
            DataLoader: DataLoader for training data.
        """
        process_global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        total_rank = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.py_logger.info(f"Starting mmap {process_global_rank}/{total_rank}")
        self.py_logger.info(f"Current epoch and Max epochs {self.trainer.current_epoch, self.data_args.max_epoch}")

        # Load the data indices for this epoch for each device
        slot = self.trainer.current_epoch
        dataset_indices = self.shuffled_train_indices_by_epochs[slot]
        num_sample_per_rank = int(len(dataset_indices) / total_rank)
        start_ind_rank = num_sample_per_rank * process_global_rank
        end_ind_rank = num_sample_per_rank * (process_global_rank + 1)
        dataset_indices = dataset_indices[start_ind_rank:end_ind_rank]

        self.py_logger.info(
            f"Create train dataset indices rank {process_global_rank} out of {total_rank}, \
                                    Loaded samples {len(dataset_indices)}"
        )

        dl = DataLoader(
            NumpyDataset(dataset_indices),
            batch_size=self.batch_size,
            collate_fn=self.train_collate_fn_,
            num_workers=self.data_args.num_workers,
        )
        self.py_logger.info(f"Finished loading training data")
        return dl

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation data loader.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        self.py_logger.info(f"Load validation data")
        return DataLoader(
            ArrowDataset(self.valid_dataset["text"]),
            batch_size=self.batch_size,
            collate_fn=self.eval_collate_fn_,
            num_workers=self.data_args.num_workers
        )