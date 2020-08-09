"""
Self attention modules per Vaswani et al. 2017 (arXiv 1706.03762)
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
from transformers.modeling_utils import prune_linear_layer

from ane_research.models.modules.attention import Attention
from ane_research.models.modules.attention.activations import AttentionActivationFunction

@Attention.register('multihead_self')
class MultiHeadSelfAttention(Attention):
    """Multi-head attention allows the model to jointly attend to information from different representation
       subspaces at different positions (Vaswani et al. 2017)."""
    def __init__(self, n_heads: int, dim: int, activation_function: AttentionActivationFunction, dropout: float):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim

        self.activation = activation_function

        self.dropout = nn.Dropout(p=dropout)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=dim, out_features=dim)
        self.k_lin = nn.Linear(in_features=dim, out_features=dim)
        self.v_lin = nn.Linear(in_features=dim, out_features=dim)
        self.out_lin = nn.Linear(in_features=dim, out_features=dim)

        self.pruned_heads = set()

    def prune_heads(self, heads: List[int]):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor,
                head_mask: Optional[torch.Tensor] = None, output_attentions: Optional[bool] = False):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)
        head_mask [Optional]: torch.tensor(num_hidden_layers, bs, num_heads, seq_length, seq_length)
        output_attentions [Optional]: bool 
            Include attention weights in outputs

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)

        weights = self.activation(scores, mask)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)
