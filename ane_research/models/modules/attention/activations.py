"""Suite of activation functions to convert attention weights to probability distributions"""
from overrides import overrides

from allennlp.common import Registrable
from allennlp.nn.util import replace_masked_values
from allennlp.nn import util

from entmax import entmax15, sparsemax, entmax_bisect

import torch
import torch.nn as nn


class AttentionActivationFunction(nn.Module, Registrable):
    """Attention activation function base class"""

    def forward(self, **kwargs) -> torch.Tensor:
        """Map a score vector to a probability distribution"""
        raise NotImplementedError('Implement forward Model')


@AttentionActivationFunction.register("softmax")
class SoftmaxActivation(AttentionActivationFunction):
    @overrides
    def forward(self, scores: torch.Tensor, mask: torch.Tensor, invert_mask: bool = True) -> torch.Tensor:
        """Map a score vector to a dense probability distribution

        Args:
            scores (torch.Tensor): (Batch x Sequence Length)
                Attention scores (also referred to as weights)
            mask (torch.Tensor): (Batch x Sequence Length)
                Specifies which indices are just padding
            invert_mask (bool)
                PyTorch fills in things with a mask value of 1, AllenNLP modules expect the opposite

        Returns:
            torch.Tensor: Dense distribution
        """
        if invert_mask:
            masked_scores = replace_masked_values(scores, mask, -float("inf"))
        else:
            masked_scores = scores.masked_fill(mask, -float("inf"))
        return torch.nn.Softmax(dim=-1)(scores)


@AttentionActivationFunction.register("sparsemax")
class SparsemaxActivation(AttentionActivationFunction):
    @overrides
    def forward(self, scores: torch.Tensor, mask: torch.Tensor, invert_mask: bool = True) -> torch.Tensor:
        """Map a score vector to a sparse probability distribution

        Args:
            scores (torch.Tensor): (Batch x Sequence Length)
                Attention scores (also referred to as weights)
            mask (torch.Tensor): (Batch x Sequence Length)
                Specifies which indices are just padding
            invert_mask (bool)
                PyTorch fills in things with a mask value of 1, AllenNLP modules expect the opposite

        Returns:
            torch.Tensor: Sparse distribution
        """
        if invert_mask:
            masked_scores = replace_masked_values(scores, mask, -float("inf"))
        else:
            masked_scores = scores.masked_fill(mask, -float("inf"))
        return sparsemax(masked_scores, dim=-1)


@AttentionActivationFunction.register("entmax15")
class Entmax15Activation(AttentionActivationFunction):
    @overrides
    def forward(self, scores: torch.Tensor, mask: torch.Tensor, invert_mask: bool = True) -> torch.Tensor:
        """Map a score vector to a probability distribution halfway between softmax and sparsemax

        Args:
            scores (torch.Tensor): (Batch x Sequence Length)
                Attention scores (also referred to as weights)
            mask (torch.Tensor): (Batch x Sequence Length)
                Specifies which indices are just padding
            invert_mask (bool)
                PyTorch fills in things with a mask value of 1, AllenNLP modules expect the opposite

        Returns:
            torch.Tensor: Distribution halfway between softmax and sparsemax
        """
        if invert_mask:
            masked_scores = replace_masked_values(scores, mask, -float("inf"))
        else:
            masked_scores = scores.masked_fill(mask, -float("inf"))
        return entmax15(masked_scores, dim=-1)


@AttentionActivationFunction.register("entmax-alpha")
class EntmaxAlphaActivation(AttentionActivationFunction):

    def __init__(self, alpha: float):
        super().__init__()

        self.alpha = torch.nn.Parameter(torch.tensor(alpha, requires_grad=True)) 

    @overrides
    def forward(self, scores: torch.Tensor, mask: torch.Tensor, invert_mask: bool = True) -> torch.Tensor:
        """Map a score vector to a probability distribution akin to softmax (alpha=1) and sparsemax (alpha=2)

        Args:
            scores (torch.Tensor): (Batch x Sequence Length)
                Attention scores (also referred to as weights)
            mask (torch.Tensor): (Batch x Sequence Length)
                Specifies which indices are just padding
            invert_mask (bool)
                PyTorch fills in things with a mask value of 1, AllenNLP modules expect the opposite

        Returns:
            torch.Tensor: Distribution resulting from entmax with specified alpha
        """
        if invert_mask:
            masked_scores = replace_masked_values(scores, mask, -float("inf"))
        else:
            masked_scores = scores.masked_fill(mask, -float("inf"))

        return entmax_bisect(masked_scores, self.alpha, dim=-1)
