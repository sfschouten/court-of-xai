"""
Bahdanau et al. 2015 (arXiv 1409.0473) additive attention modules
"""

# pylint: disable=E1101
# pylint incorrectly identifies some types as tuples
from typing import List, Optional, Tuple

from allennlp.common import JsonDict
import torch
import torch.nn as nn

from xai_court.models.modules.attention.attention import Attention, AttentionAnalysisMethods
from xai_court.models.modules.attention.activations import AttentionActivationFunction


@Attention.register('additive_basic')
class AdditiveAttentionBasic(Attention):
    """
    Query-less additive attention module variant as described by Bahdanau et al. 2015 (arXiv 1409.0473)
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
        self.num_intermediate_features = hidden_size // 2
        self.alignment_layer1 = nn.Linear(hidden_size, self.num_intermediate_features)
        self.alignment_layer2 = nn.Linear(self.num_intermediate_features, 1, bias=False)

    def forward(
        self,
        key: torch.Tensor,
        mask: torch.Tensor,
        output_attentions: Optional[List[AttentionAnalysisMethods]] = None
    ) -> Tuple[torch.Tensor, JsonDict]:
        """
        Compute a weight distribution on the input sequence, (hopefully) assigning higher values to more relevant elements.
        We use the alternative formulation of Kobayashi et al. 2020 (arXiv 2004.10102) to facilitate the optional calculation
        of the weighted vector norm ||alpha f(x)||.

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
        #  (bs, n_heads, q_length, k_length) * (bs, n_heads, q_length, dim_per_head)
        layer1 = nn.Tanh()(self.alignment_layer1(key))
        layer2 = self.alignment_layer2(layer1).squeeze(-1) # (bs, seq_len)

        weights = self.activation(layer2, mask) # (bs, seq_len)

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


@Attention.register('additive_query')
class AdditiveAttentionQuery(Attention):
    """
    Full additive attention module variant as described by Bahdanau et al. 2015 (arXiv 1409.0473)
    Calculates a weight distribution with a feedforward alignment model operating on key and query vectors.
    We use the alternative formulation of Kobayashi et al. 2020 (arXiv 2004.10102) to facilitate the optional calculation
    of the weighted vector norm ||alpha f(x)||.

    Parameters:
        hidden_size (int):
            Input dimensionality of the alignment model
        activation_function (AttentionActivationFunction):
            Attention activation function module
    """
    def __init__(self, hidden_size: int, activation_function: AttentionActivationFunction):
        super().__init__()
        self.activation = activation_function
        self.num_intermediate_features = hidden_size // 2
        self.alignment_layer1_k = nn.Linear(hidden_size, self.num_intermediate_features)
        self.alignment_layer1_q = nn.Linear(hidden_size, self.num_intermediate_features)
        self.alignment_layer2 = nn.Linear(self.num_intermediate_features, 1, bias=False)

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
        layer1 = nn.Tanh()(self.alignment_layer1_k(key) + self.alignment_layer1_q(query).unsqueeze(1))
        layer2 = self.alignment_layer2(layer1).squeeze(-1)
        weights = self.activation(layer2, mask) # (bs, seq_len)

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
