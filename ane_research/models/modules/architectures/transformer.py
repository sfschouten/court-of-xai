"""
The building blocks for a basic Transformer model described in 'Attention is All You Need' (Vaswani et al. 2017)
Code taken from the HuggingFace Transformer v2.11.0 library with minor modifications
"""

import copy
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from allennlp.common import FromParams
from transformers.activations import gelu
from transformers.modeling_utils import prune_linear_layer

from ane_research.models.modules.attention.activations import attention_bind_map

class FFN(nn.Module):
    """Fully connected feed-forward network"""
    def __init__(self, dim: int, hidden_dim: int, activation: str, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        assert activation in ["relu", "gelu"], "activation ({}) must be in ['relu', 'gelu']".format(
            activation
        )
        self.activation = gelu if activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    """Create fixed sinusoidal positional encodings"""
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class Embeddings(nn.Module, FromParams):
    """Combined word embeddings and positional encoding"""
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        pad_token_id: int,
        max_position_embeddings: int,
        sinusoidal_pos_embds: bool,
        dropout: float
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, dim)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings, dim=dim, out=self.position_embeddings.weight
            )

        self.LayerNorm = nn.LayerNorm(dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.
        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """Multi-head attention allows the model to jointly attend to information from different representation
       subspaces at different positions (Vaswani et al. 2017)."""
    def __init__(self, n_heads: int, dim: int, activation: str, attention_dropout: float):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim

        assert activation in attention_bind_map.keys(), f"activation ({activation}) must be in {list(attention_bind_map.keys)}"
        self.activation = attention_bind_map[activation]

        self.dropout = nn.Dropout(p=attention_dropout)

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

        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)
        weights = self.activation(scores)  # (bs, n_heads, q_length, k_length)
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


class TransformerBlock(nn.Module):
    def __init__(self, n_heads: int, dim: int, hidden_dim: int, ffn_activation: str, attention_activation: str, attention_dropout: float):
        super().__init__()

        assert dim % n_heads == 0

        self.attention = MultiHeadSelfAttention(n_heads, dim, attention_activation, attention_dropout)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

        self.ffn = FFN(dim, hidden_dim, ffn_activation, attention_dropout)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = False):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)
        head_mask [Optional]: torch.tensor(num_hidden_layers, bs, num_heads, seq_length, seq_length)
        output_attentions [Optional]: bool 
            Include attention weights in outputs

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x, key=x, value=x, mask=attn_mask, head_mask=head_mask, output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):
    def __init__(self,
        n_layers: int,
        n_heads: int,
        dim: int,
        hidden_dim: int,
        ffn_activation: str,
        attention_activation: str,
        attention_dropout: float
    ):
        super().__init__()
        self.n_layers = n_layers
        layer = TransformerBlock(
            n_heads=n_heads,
            dim=dim,
            hidden_dim=hidden_dim,
            ffn_activation=ffn_activation, 
            attention_activation=attention_activation,
            attention_dropout=attention_dropout
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = False, output_hidden_states: Optional[bool] = False):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)
        head_mask [Optional]: torch.tensor(num_hidden_layers, bs, num_heads, seq_length, seq_length)
        output_attentions [Optional]: bool 
            Include attention weights in outputs
        output_hidden_states [Optional]: bool
            Include all hidden states in outputs

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = ()
        all_attentions = ()

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(
                x=hidden_state,
                attn_mask=attn_mask,
                head_mask=head_mask[i],
                output_attentions=output_attentions
            )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        outputs = (hidden_state,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
