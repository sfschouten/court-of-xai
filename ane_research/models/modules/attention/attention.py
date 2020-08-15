"""Define base class for attention modules"""
from enum import Enum
from typing import List

from allennlp.common import Registrable
import torch
import torch.nn as nn


class AttentionAnalysisMethods(Enum):
    """Defines the possible methods for analyzing the approximate attention to input tokens"""
    # Raw attention weights
    weight_based = 'attn_weights'
    # Weighted vector norms as described by Kobayashi et al. 2020 (arXiv 2004.10102)
    norm_based   = 'attn_norm'
    # Attention rollout as described by Abnar and Zuidema 2020 (arXiv 2005.00928)
    rollout      = 'attn_rollout'
    # Attention flow as described by Abnar and Zuidema 2020 (arXiv 2005.00928)
    flow         = 'attn_flow'


class Attention(nn.Module, Registrable):
    """Attention module base class"""

    def forward(self, **kwargs) -> torch.Tensor:
        """Perform module forward pass"""
        raise NotImplementedError("Implement forward Model")
