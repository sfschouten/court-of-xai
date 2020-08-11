"""Define base class for attention modules"""
from allennlp.common import Registrable
import torch
import torch.nn as nn


class Attention(nn.Module, Registrable):
    """Attention module base class"""

    def forward(self, **kwargs) -> torch.Tensor:
      """Perform module forward pass"""
      raise NotImplementedError("Implement forward Model")
