'''
Bahdanau et al. 2015 (arXiv 1409.0473) - Additive Explanation Modules as implemented in PyTorch
by  Jain and Wallace 2019 (arXiv 1902.10186)
'''

# pylint: disable=E1101
# pylint incorrectly identifies some types as tuples

from __future__ import annotations

import torch
import torch.nn as nn

from ane_research.models.modules.attention.attention import Attention
from ane_research.models.modules.attention.activations import activation_function_map


@Attention.register('additive_tanh')
class TanhAdditiveAttention(Attention):
  '''Bahdanau tanh additive attention with support for different activation functions'''
  def __init__(self, hidden_size: int, activation_function: str):
    super().__init__()
    self.activation_function = activation_function_map[activation_function]
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

@Attention.register('additive_sdp')
class SDPAdditiveAttention(Attention):
  '''Bahdanau scaled dot product additive attention with support for different activation functions'''
  def __init__(self, hidden_size: int, activation_function: str):
    super().__init__()
    self.activation_function = activation_function_map[activation_function]
    self.attention = nn.Linear(hidden_size, 1, bias=False)
    self.hidden_size = hidden_size

  def forward(self, hidden_state: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    '''Perform forward pass

    Args:
      hidden_state (torch.Tensor): (Batch x Sequence Length) Decoder hidden state
      masks (torch.Tensor): (Batch x Sequence Length) masks to apply to padded elements in hidden state

    Returns:
        torch.Tensor: Distribution (dense/sparse depending on activation function)
    '''
    attention = self.attention(hidden_state) / (self.hidden_size)**0.5
    attention = attention.squeeze(-1)
    attn = self.activation_function(attention, masks)

    return attn
