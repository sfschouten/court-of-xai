'''Suite of activation functions to convert attention weights to probability distributions'''

import torch

from allennlp.nn.util import replace_masked_values
from entmax import sparsemax


def masked_sparsemax(weight_vector: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
  '''Masked Sparsemax Activation. See Martins & Astudillo 2016 (arXiv 1602.02068)

  Args:
    - weight_vector (torch.Tensor): Output weights of final hidden layer
    - masks (torch.Tensor): Boolean tensor specify mask indices

  Returns:
    torch.Tensor: Sparse distribution
  '''
  a = replace_masked_values(weight_vector, masks, -float('inf'))
  attention = sparsemax(a, dim=-1)
  return attention

def masked_softmax(weight_vector: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
  '''Masked Softmax Activation

  Args:
    weight_vector (torch.Tensor): Output weights of final hidden layer
    masks (torch.Tensor): Boolean tensor specify mask indices

  Returns:
    torch.Tensor: Dense distribution
  '''
  a = replace_masked_values(weight_vector, masks, -float('inf'))
  attention = torch.nn.Softmax(dim=-1)(weight_vector)
  return attention

activation_function_map = {
  'softmax': masked_softmax,
  'sparsemax': masked_sparsemax
}
