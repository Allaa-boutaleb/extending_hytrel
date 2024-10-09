import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from hytrel_modules.layers import AllSetTrans

class Embedding(nn.Module):
    """
    Embedding layer for the HyTrel model.
    
    This layer applies token embedding, layer normalization, and dropout to the input tensors.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Embedding layer.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters.
        """
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(config['vocab_size'], config['hidden_size'], padding_idx=config['pad_token_id'])
        self.norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Embedding layer.

        Args:
            x_s (torch.Tensor): Source node input tensor.
            x_t (torch.Tensor): Target node input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Embedded and normalized source and target tensors.
        """
        embedding_s, embedding_t = self.tok_embed(x_s), self.tok_embed(x_t)
        embedding_s = torch.div(torch.sum(embedding_s, dim=1), torch.count_nonzero(x_s, dim=1).unsqueeze(-1))
        embedding_t = torch.div(torch.sum(embedding_t, dim=1), torch.count_nonzero(x_t, dim=1).unsqueeze(-1))
        return self.dropout(self.norm(embedding_s)), self.dropout(self.norm(embedding_t))

class EncoderLayer(nn.Module):
    """
    EncoderLayer for the HyTrel model.
    
    This layer applies the AllSetTrans transformation to the input embeddings.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the EncoderLayer.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters.
        """
        super().__init__()
        self.dropout = config['hidden_dropout_prob']
        self.V2E = AllSetTrans(config=config)
        self.fuse = nn.Linear(config['hidden_size']*2, config['hidden_size'])
        self.E2V = AllSetTrans(config=config)

    def forward(self, embedding_s: torch.Tensor, embedding_t: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the EncoderLayer.

        Args:
            embedding_s (torch.Tensor): Source node embeddings.
            embedding_t (torch.Tensor): Target node embeddings.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated source and target embeddings.
        """
        # Reverse the index
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        
        # From nodes to hyper-edges
        embedding_t_tem = F.relu(self.V2E(embedding_s, edge_index))

        # From hyper-edges to nodes
        embedding_t = torch.cat([embedding_t, embedding_t_tem], dim=-1)
        # Fuse the output t_embeds with original t_embeds
        embedding_t = F.dropout(self.fuse(embedding_t), p=self.dropout, training=self.training)
        embedding_s = F.relu(self.E2V(embedding_t, reversed_edge_index))
        embedding_s = F.dropout(embedding_s, p=self.dropout, training=self.training)

        return embedding_s, embedding_t

class Encoder(nn.Module):
    """
    Encoder for the HyTrel model.
    
    This module applies multiple EncoderLayers to process the input data.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Encoder.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters.
        """
        super(Encoder, self).__init__()
        self.config = config
        self.embed_layer = Embedding(config)
        self.layer = nn.ModuleList([EncoderLayer(config) for _ in range(config['num_hidden_layers'])])

    def forward(self, data: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Encoder.

        Args:
            data (Any): Input data containing x_s, x_t, and edge information.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Final source and target embeddings.
        """
        embedding_s, embedding_t = self.embed_layer(data.x_s, data.x_t)
        embedding_t = torch.cat([embedding_t, embedding_s], dim=0)

        # Add self-loop
        num_nodes, num_hyper_edges = data.x_s.size(0), data.x_t.size(0)
        self_edge_index = torch.tensor([[i, num_hyper_edges+i] for i in range(num_nodes)]).T
        if ('edge_neg_view' in self.config and self.config['edge_neg_view'] == 1):
            edge_index = torch.cat([data.edge_index_corr1, self_edge_index.to(data.edge_index_corr1.device)], dim=-1)
        elif ('edge_neg_view' in self.config and self.config['edge_neg_view'] == 2):
            edge_index = torch.cat([data.edge_index_corr2, self_edge_index.to(data.edge_index_corr2.device)], dim=-1)
        else:
            edge_index = torch.cat([data.edge_index, self_edge_index.to(data.edge_index.device)], dim=-1)

        for layer_module in self.layer:
            embedding_s, embedding_t = layer_module(embedding_s, embedding_t, edge_index)
        
        outputs = (embedding_s, embedding_t[:num_hyper_edges])

        return outputs

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss module for the HyTrel model.
    
    This module implements the InfoNCE loss as described in the SimCLR paper.
    """
    def __init__(self, temperature: float = 0.5):
        """
        Initialize the ContrastiveLoss module.

        Args:
            temperature (float): Temperature parameter for the loss calculation. Defaults to 0.5.
        """
        super().__init__()
        self.temperature = temperature
        self.loss_fct = nn.CrossEntropyLoss()

    def calc_similarity_batch(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Calculate the similarity between two sets of embeddings.

        Args:
            a (torch.Tensor): First set of embeddings.
            b (torch.Tensor): Second set of embeddings.

        Returns:
            torch.Tensor: Similarity matrix.
        """
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1: torch.Tensor, proj_2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ContrastiveLoss module.

        Args:
            proj_1 (torch.Tensor): First set of projections.
            proj_2 (torch.Tensor): Second set of projections.

        Returns:
            torch.Tensor: Calculated contrastive loss.
        """
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)
        
        cos_sim = torch.einsum('id,jd->ij', z_i, z_j) / self.temperature
        labels = torch.arange(cos_sim.size(0)).long().to(proj_1.device)       
        loss = self.loss_fct(cos_sim, labels)

        return loss