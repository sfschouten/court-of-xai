"""
Scaled, multiplicative attention module per Luong et al. 2015 (arXiv 1508.04025) and Vaswani et al. 2017 (arXiv 1706.03762)
"""
from typing import List, Optional, Tuple

from allennlp.common import JsonDict
import torch
import torch.nn as nn

from xai_court.models.modules.attention import Attention, AttentionAnalysisMethods
from xai_court.models.modules.attention.activations import AttentionActivationFunction


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

    def forward(
        self,
        key: torch.Tensor,
        mask: torch.Tensor,
        output_attentions: Optional[List[AttentionAnalysisMethods]] = None
    ) -> Tuple[torch.Tensor, JsonDict]:
        """Compute a weight distribution on the input sequence, (hopefully) assigning higher values to more relevant elements.

        Args:
            key (torch.Tensor): (Batch x Sequence Length x Hidden Dimension)
                The encoded data features. In a BiLSTM, this corresponds to the encoder annotation, h
            mask (torch.Tensor): (Batch x Sequence Length)
                Mask to apply to padded key elements
            output_attentions [Optional]: List[AttentionAnalysisMethods]
                Attention analysis methods to returns. At the attention module level, supports
                AttentionAnalysisMethods.weight_based and AttentionAnalysisMethods.norm_based

        Returns:
            context (torch.Tensor): (Batch x Sequence Length)
                Attention weights
            attention_dict: JsonDict
                Optional if `output_attentions` is not empty. Dict of requested AttentionAnalysisMethods
                and their corresponding tensors
        """
        transformed = self.alignment_model(key) / (self.hidden_size)**0.5 # (bs, seq_len, 1)
        transformed = transformed.squeeze(-1) # (bs, seq_len)
        weights = self.activation(transformed, mask) # (bs, seq_len)

        alpha_fx = weights.unsqueeze(-1) * key # (bs, seq_len, hidden_dim)
        context = alpha_fx.sum(dim=1) # (bs, hidden_dim)

        if output_attentions:
            attention_dict = {}
            if AttentionAnalysisMethods.weight_based in output_attentions:
                attention_dict[AttentionAnalysisMethods.weight_based] = weights
            if AttentionAnalysisMethods.norm_based in output_attentions:
                norm = torch.norm(alpha_fx, dim=-1) # (bs, seq_len)
                attention_dict[AttentionAnalysisMethods.norm_based] = norm
            return (context, attention_dict,)
        else:
            return (context,)


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

    def forward(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor,
        output_attentions: Optional[List[AttentionAnalysisMethods]] = None
    ) -> Tuple[torch.Tensor, JsonDict]:
        """Compute a weight distribution on the input sequence, (hopefully) assigning higher values to more relevant elements.

        Args:
            key (torch.Tensor): (Batch x Sequence Length x Hidden Dimension)
                The encoded data features. In a BiLSTM, this corresponds to the encoder annotation, h
            query (torch.Tensor): (Batch x Sequence Length)
                The reference when computing the attention distribution. In a BiLSTM, this corresponds to the decoder hidden state, s
            mask (torch.Tensor): (Batch x Sequence Length)
                Mask to apply to padded key elements
            output_attentions [Optional]: List[AttentionAnalysisMethods]
                Attention analysis methods to returns. At the attention module level, supports
                AttentionAnalysisMethods.weight_based and AttentionAnalysisMethods.norm_based

        Returns:
            context (torch.Tensor): (Batch x Sequence Length)
                Attention weights
            attention_dict: JsonDict
                Optional if `output_attentions` is not empty. Dict of requested AttentionAnalysisMethods
                and their corresponding tensors
        """
        transformed = torch.bmm(key, query.unsqueeze(-1)) / self.hidden_size**0.5
        transformed = transformed.squeeze(-1)
        weights = self.activation(transformed, mask) # (bs, seq_len)

        alpha_fx = weights.unsqueeze(-1) * key # (bs, seq_len, hidden_dim)
        context = alpha_fx.sum(dim=1) # (bs, hidden_dim)

        if output_attentions:
            attention_dict = {}
            if AttentionAnalysisMethods.weight_based in output_attentions:
                attention_dict[AttentionAnalysisMethods.weight_based] = weights
            if AttentionAnalysisMethods.norm_based in output_attentions:
                norm = torch.norm(alpha_fx, dim=-1) # (bs, seq_len)
                attention_dict[AttentionAnalysisMethods.norm_based] = norm
            return (context, attention_dict,)
        else:
            return (context,)
