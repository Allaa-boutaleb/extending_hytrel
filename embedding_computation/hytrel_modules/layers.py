import math
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.typing import Adj, OptTensor, SparseTensor

def get_activation(act: Optional[Union[str, Callable]], inplace: bool = False) -> Callable:
    """
    Get the activation function based on the provided name or callable.

    Args:
        act (Optional[Union[str, Callable]]): Name of the activation function or a callable.
        inplace (bool): Whether to perform inplace activation. Defaults to False.

    Returns:
        Callable: The activation function.

    Raises:
        NotImplementedError: If the requested activation function is not supported.
    """
    if act is None:
        return lambda x: x

    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1, inplace=inplace)
        if act == 'identity':
            return nn.Identity()
        if act == 'elu':
            return nn.ELU(inplace=inplace)
        if act == 'gelu':
            return nn.GELU()
        if act == 'relu':
            return nn.ReLU()
        if act == 'sigmoid':
            return nn.Sigmoid()
        if act == 'tanh':
            return nn.Tanh()
        if act in {'softrelu', 'softplus'}:
            return nn.Softplus()
        if act == 'softsign':
            return nn.Softsign()
        raise NotImplementedError(f'act="{act}" is not supported. '
                                  'Try to include it if you can find that in '
                                  'https://pytorch.org/docs/stable/nn.html')

    return act

class PositionwiseFFN(nn.Module):
    """
    The Position-wise FFN layer used in Transformer-like architectures.

    If pre_norm is True:
        norm(data) -> fc1 -> act -> act_dropout -> fc2 -> dropout -> res(+data)
    Else:
        data -> fc1 -> act -> act_dropout -> fc2 -> dropout -> norm(res(+data))

    If gated projection is used:
        fc1_1 * act(fc1_2(data)) is used to map the data.
    """
    def __init__(self, config):
        """
        Initialize the PositionwiseFFN layer.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        self.config = config
        self.dropout_layer = nn.Dropout(self.config.hidden_dropout_prob)
        self.activation_dropout_layer = nn.Dropout(self.config.activation_dropout)
        self.ffn_1 = nn.Linear(in_features=self.config.hidden_size, out_features=self.config.intermediate_size,
                               bias=True)
        if self.config.gated_proj:
            self.ffn_1_gate = nn.Linear(in_features=self.config.hidden_size,
                                        out_features=self.config.hidden_size,
                                        bias=True)
        self.activation = get_activation(self.config.hidden_act)
        self.ffn_2 = nn.Linear(in_features=self.config.intermediate_size, out_features=self.config.hidden_size,
                               bias=True)
        self.layer_norm = nn.LayerNorm(eps=self.config.layer_norm_eps,
                                       normalized_shape=self.config.hidden_size)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights of the linear layers."""
        for module in self.children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data: Tensor) -> Tensor:
        """
        Forward pass of the PositionwiseFFN layer.

        Args:
            data (Tensor): Input tensor of shape (B, seq_length, C_in).

        Returns:
            Tensor: Output tensor of shape (B, seq_length, C_out).
        """
        residual = data
        if self.config.pre_norm:
            data = self.layer_norm(data)
        if self.config.gated_proj:
            out = self.activation(self.ffn_1_gate(data)) * self.ffn_1(data)
        else:
            out = self.activation(self.ffn_1(data))
        out = self.activation_dropout_layer(out)
        out = self.ffn_2(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self.config.pre_norm:
            out = self.layer_norm(out)
        return out

def glorot(tensor: Optional[Tensor]):
    """
    Initialize the given tensor using Glorot initialization.

    Args:
        tensor (Optional[Tensor]): The tensor to be initialized.
    """
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor: Optional[Tensor]):
    """
    Initialize the given tensor with zeros.

    Args:
        tensor (Optional[Tensor]): The tensor to be initialized.
    """
    if tensor is not None:
        tensor.data.fill_(0)

class AllSetTrans(MessagePassing):
    """
    AllSetTrans layer for graph attention.

    This layer implements a variation of the graph attention mechanism
    similar to the one used in the original PMA (Pooling by Multihead Attention).
    """
    def __init__(self, config, negative_slope: float = 0.2, **kwargs):
        """
        Initialize the AllSetTrans layer.

        Args:
            config: Configuration object containing model parameters.
            negative_slope (float): LeakyReLU angle of the negative slope. Defaults to 0.2.
            **kwargs: Additional keyword arguments for the MessagePassing base class.
        """
        super(AllSetTrans, self).__init__(node_dim=0, **kwargs)

        self.in_channels = config.hidden_size
        self.heads = config.num_attention_heads
        self.hidden = config.hidden_size // self.heads
        self.out_channels = config.hidden_size

        self.negative_slope = negative_slope
        self.dropout = config.attention_probs_dropout_prob
        self.aggr = 'add'

        self.lin_K = Linear(self.in_channels, self.heads * self.hidden)
        self.lin_V = Linear(self.in_channels, self.heads * self.hidden)
        self.att_r = Parameter(torch.Tensor(1, self.heads, self.hidden))  # Seed vector
        self.rFF = PositionwiseFFN(config)

        self.ln0 = nn.LayerNorm(self.heads * self.hidden)
        self.ln1 = nn.LayerNorm(self.heads * self.hidden)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the layer."""
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x: Tensor, edge_index: Adj, 
                return_attention_weights: Optional[bool] = None) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        """
        Forward pass of the AllSetTrans layer.

        Args:
            x (Tensor): Input node features.
            edge_index (Adj): Graph connectivity in COO format.
            return_attention_weights (Optional[bool]): If set to True, returns the attention weights
                                                       for each edge. Defaults to None.

        Returns:
            Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]: 
                - If return_attention_weights is False, returns the output tensor.
                - If return_attention_weights is True, returns a tuple containing the output tensor
                  and a tuple of (edge_index, attention_weights).
        """
        H, C = self.heads, self.hidden
        alpha_r: OptTensor = None
        
        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
        x_K = self.lin_K(x).view(-1, H, C)
        x_V = self.lin_V(x).view(-1, H, C)
        alpha_r = (x_K * self.att_r).sum(dim=-1)

        out = self.propagate(edge_index, x=x_V,
                             alpha=alpha_r, aggr=self.aggr)

        alpha = self._alpha
        self._alpha = None
        out += self.att_r  # Seed + Multihead
        # concat heads then LayerNorm
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        # rFF and skip connection
        out = self.ln1(out + F.relu(self.rFF(out)))

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        """
        Compute the message passed from nodes to edges.

        Args:
            x_j (Tensor): Source node features.
            alpha_j (Tensor): Source node attention coefficients.
            index (Tensor): Target node indices.
            ptr (OptTensor): If given, the ptr vector
            size_i (Optional[int]): Size of target nodes.

        Returns:
            Tensor: The message passed from nodes to edges.
        """
        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        """
        Aggregates messages from neighbors.

        Args:
            inputs (Tensor): Transformed messages.
            index (Tensor): Indices of target nodes.
            ptr (Optional[Tensor]): If given, the ptr vector. Defaults to None.
            dim_size (Optional[int]): Size of the output tensor. Defaults to None.

        Returns:
            Tensor: The aggregated messages.
        """
        if ptr is not None:
            ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(s