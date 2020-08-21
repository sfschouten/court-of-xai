"""Define base class for attention modules"""
from enum import Enum
from typing import List

from allennlp.common import Registrable
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionAnalysisMethods(Enum):
    """Defines the possible methods for analyzing attention."""

    # Raw attention weights
    weight_based = 'attn_weights'
    # Weighted vector norms as described by Kobayashi et al. 2020 (arXiv 2004.10102)
    norm_based   = 'attn_norm'

class AttentionAggregator():

    def id(self):
        raise NotImplementedError("Implement the id method")

    def aggregate(attention_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
          attention: attention from the various attention mechanisms in the model.
        Returns:
          the attention, but aggregated to be of shape (bs, seq_len)
        """
        raise NotImplementedError("Implement the aggregation method.")

class AttentionAverager(AttentionAggregator):

    def id(self):
        return "avg"
    
    def aggregate(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Averages attention accross layers, heads, and max pools across the last
        dimension of attention matrix.

        Parameters
        ----------
        attention_matrix: torch.tensor(bs, n_layers, n_heads, seq_length, seq_length)
            Matrix of attention weights
        """
        attention = attention.mean(dim=1)
        attention = attention.mean(dim=1)
        attention, _ = attention.max(dim=2)
        return attention

class AttentionRollout(AttentionAggregator):

    def id(self):
        return "roll"

    def aggregate(self, attention_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute input Attention Rollout as described by Abnar and Zuidema 2020 (arXiv 2005.00928):

        From the paper: "Attention Rollout assumes that the identities of input tokens are linearly combined
        through the layers based on the attention weights. To adjust attention weights, it rolls out the weights
        to capture the propagation of information from input tokens to intermediate hidden embeddings."

        Calculation
        -----------
        Attention Rollout: A~
        Attention Weights: A

        A~(l_i) = A(l_i)A~(l_i−1) if i > j
        A~(l_i) = A(l_i) if i = j

        For input attention, i = n_layers, j = 0

        Parameters
        ----------
        attention_matrix: torch.tensor(bs, n_layers, n_heads, seq_length, seq_length)
            Full matrix of attention weights

        Outputs
        -------
        attention_rollout: Tuple[torch.tensor(bs, seq_length, seq_length)]
            Attention rollout weights at each layer
        """
        B, L, H, S, _ = attention_matrix.shape

        # Use a single attention graph by averaging all heads
        attn_avg_heads = attention_matrix.mean(dim=2) # (bs, n_layers, seq_length, seq_length)

        # To account for residual connections, we add an identity matrix to the attention matrix
        # and re-normalize the weights.
        residual_atttention_matrix = torch.eye(attn_avg_heads.shape[2]) # (seq_length, seq_length)
        augmented_attention_matrix = attn_avg_heads + residual_atttention_matrix # (bs, n_layers, seq_length, seq_length)
        augmented_attention_matrix = F.normalize(augmented_attention_matrix, dim=-1, p=1)

        rollout_attention = torch.zeros(augmented_attention_matrix.shape) # (bs, n_layers, seq_length, seq_length)

        # Roll out the weights
        rollout_attention[:,0] = augmented_attention_matrix[:,0]
        for i in range(1, L):
            rollout_attention[:,i] = torch.matmul(
                augmented_attention_matrix[:,i],
                rollout_attention[:,i-1]
            )

        # Tuple of L tensors of shape (bs, seq_length, seq_length)i
        #results = torch.unbind(rollout_attention, dim=1)
        
        return rollout_attention[:,-1,0]


class Attention(nn.Module, Registrable):
    """Attention module base class"""

    def forward(self, **kwargs) -> torch.Tensor:
        """Perform module forward pass"""
        raise NotImplementedError("Implement forward Model")
