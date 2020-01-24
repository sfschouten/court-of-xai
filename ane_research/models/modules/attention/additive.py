'''
Bahdanau et al. 2015 (arXiv 1409.0473) - Additive Explanation as implemented in PyTorch
by  Jain and Wallace 2019 (arXiv 1902.10186)
'''

from __future__ import annotations

import torch
import torch.nn as nn

from ane_research.models.modules.attention.activations import masked_sparsemax, masked_softmax
from ane_research.models.modules.attention.base import Attention

@Attention.register('additive')
class AdditiveAttention(Attention):
  '''Bahdanau additive attention with support for different activation functions'''

  activation_function_map = {
    'masked_sparsemax': masked_sparsemax,
    'masked_softmax': masked_softmax
  }

  def __init__(self, hidden_size: int, activation_function: str = 'masked_sparsemax'):
    super().__init__()
    self.activation_function = self.activation_function_map[activation_function]
    self.num_intermediate_features = hidden_size // 2
    self.layer1 = nn.Linear(hidden_size, self.num_intermediate_features)
    self.layer2 = nn.Linear(self.num_intermediate_features, 1, bias=False)

  def forward(self, hidden_state: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    '''Perform forward pass

    Args:
      hidden_state (torch.Tensor): (Batch x Sequence Length) Decoder hidden state
      masks (torch.Tensor): (Batch x Sequence Length) masks to apply to padded elements in hidden state

    Returns:
        torch.Tensor: Distribution (dense/sparse depending on activation function)
    '''
    layer1 = nn.Tanh()(self.layer1(hidden_state))
    layer2 = self.layer2(layer1).squeeze(-1)
    attn = self.activation_function(layer2, masks)

    return attn
