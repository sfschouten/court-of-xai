"""Suite of activation functions to convert attention weights to probability distributions"""
from overrides import overrides

from allennlp.common import Registrable
from allennlp.nn.util import replace_masked_values
from entmax import entmax15, sparsemax
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
    def forward(self, scores: torch.Tensor, mask: torch.Tensor):
        """Map a score vector to a dense probability distribution

        Args:
            scores (torch.Tensor): (Batch x Sequence Length)
                Attention scores (also referred to as weights)
            mask (torch.Tensor): (Batch x Sequence Length)
                Specifies which indices are just padding

        Returns:
            torch.Tensor: Dense distribution
        """
        masked_scores = replace_masked_values(scores, mask, -float("inf"))
        return torch.nn.Softmax(dim=-1)(masked_scores)


@AttentionActivationFunction.register("sparsemax")
class SparsemaxActivation(AttentionActivationFunction):
    @overrides
    def forward(self, scores: torch.Tensor, mask: torch.Tensor):
        """Map a score vector to a sparse probability distribution

        Args:
            scores (torch.Tensor): (Batch x Sequence Length)
                Attention scores (also referred to as weights)
            mask (torch.Tensor): (Batch x Sequence Length)
                Specifies which indices are just padding

        Returns:
            torch.Tensor: Sparse distribution
        """
        masked_scores = replace_masked_values(scores, mask, -float("inf"))
        return sparsemax(masked_scores, dim=-1)


@AttentionActivationFunction.register("entmax15")
class Entmax15Activation(AttentionActivationFunction):
    @overrides
    def forward(self, scores: torch.Tensor, mask: torch.Tensor):
        """Map a score vector to a probability distribution halfway between softmax and sparsemax

        Args:
            scores (torch.Tensor): (Batch x Sequence Length)
                Attention scores (also referred to as weights)
            mask (torch.Tensor): (Batch x Sequence Length)
                Specifies which indices are just padding

        Returns:
            torch.Tensor: Distribution halfway between softmax and sparsemax
        """
        masked_scores = replace_masked_values(scores, mask, -float("inf"))
        return entmax15(masked_scores, dim=-1)
