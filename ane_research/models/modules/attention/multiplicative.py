"""
Scaled, multiplicative attention module per Luong et al. 2015 (arXiv 1508.04025) and Vaswani et al. 2017 (arXiv 1706.03762)
"""
import torch
import torch.nn as nn

from ane_research.models.modules.attention import Attention
from ane_research.models.modules.attention.activations import AttentionActivationFunction


@Attention.register('scaled_multiplicative')
class ScaledMultiplicativeAttention(Attention):
    """
    Query-less scaled, multiplicative attention module variant as described by Luong et al. 2015 (arXiv 1508.04025)
    and Vaswani et al. 2017 (arXiv 1706.03762).
    Calculates a weight distribution with a feedforward alignment model operating exclusively on a key vector

    Parameters:
        hidden_size (int):
            Input dimensionality of the alignment model
        activation_function (AttentionActivationFunction):
            Attention activation function module
    """
    def __init__(self, hidden_size: int, activation_function: AttentionActivationFunction):
        super().__init__()
        self.activation = activation_function
        self.alignment_model = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size = hidden_size

    def forward(self, key: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute a weight distribution on the input sequence, (hopefully) assigning higher values to more relevant elements.

        Args:
            key (torch.Tensor): (Batch x Sequence Length x Hidden Dimension)
                The encoded data features. In a BiLSTM, this corresponds to the encoder annotation, h
            mask (torch.Tensor): (Batch x Sequence Length)
                Mask to apply to padded key elements

        Returns:
            torch.Tensor: (Batch x Sequence Length)
                Attention scores
        """
        transformed = self.alignment_model(key) / (self.hidden_size)**0.5
        transformed = transformed.squeeze(-1)
        scores = self.activation(transformed, mask)

        return scores


@Attention.register('scaled_multiplicative_query')
class ScaledMultiplicativeAttentionQuery(Attention):
    """
    Full scaled, multiplicative attention module variant as described by Luong et al. 2015 (arXiv 1508.04025)
    and Vaswani et al. 2017 (arXiv 1706.03762).
    Calculates a weight distribution with a feedforward alignment model operating on key and query vectors

    Parameters:
        hidden_size (int):
            Input dimensionality of the alignment model
        activation_function (AttentionActivationFunction):
            Attention activation function module
    """
    def __init__(self, hidden_size: int, activation_function: AttentionActivationFunction):
        super().__init__()
        self.activation = activation_function
        self.alignment_model = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size = hidden_size

    def forward(self, key: torch.Tensor, query: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute a weight distribution on the input sequence, (hopefully) assigning higher values to more relevant elements.

        Args:
            key (torch.Tensor): (Batch x Sequence Length x Hidden Dimension)
                The encoded data features. In a BiLSTM, this corresponds to the encoder annotation, h
            query (torch.Tensor): (Batch x Sequence Length)
                The reference when computing the attention distribution. In a BiLSTM, this corresponds to the decoder hidden state, s
            mask (torch.Tensor): (Batch x Sequence Length)
                Mask to apply to padded key elements

        Returns:
            torch.Tensor: (Batch x Sequence Length)
                Attention scores
        """


        transformed = torch.bmm(key, query.unsqueeze(-1)) / self.hidden_size**0.5
        transformed = transformed.squeeze(-1)
        scores = self.activation(transformed, mask)

        return scores
