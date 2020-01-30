'''Define base class for attention modules'''

import torch
import torch.nn as nn

from allennlp.common import Registrable


class Attention(nn.Module, Registrable):
  '''Attention module base class'''

  def forward(self, **kwargs) -> torch.Tensor:
    '''Perform module forward pass'''
    raise NotImplementedError('Implement forward Model')
