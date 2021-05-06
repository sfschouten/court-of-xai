"""Define base class for attention modules"""
from enum import Enum
from typing import Dict, List, Tuple

from allennlp.common import Registrable
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionAnalysisMethods(Enum):
    """Defines the possible methods for analyzing attention scores."""

    # Raw attention weights
    weight_based = 'weights'
    # Weighted vector norms as described by Kobayashi et al. 2020 (arXiv 2004.10102)
    norm_based   = 'norm'

class AttentionAggregator(Registrable):

    def __init__(self, identifier: str):
        self._id = identifier

    @property
    def id(self):
        return self._id

    def aggregate(self, attention_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
          attention: attention from the various attention mechanisms in the model.
        Returns:
          the attention, but aggregated to be of shape (bs, seq_len)
        """
        raise NotImplementedError("Implement the aggregation method.")


@AttentionAggregator.register("attention-averager")
class AttentionAverager(AttentionAggregator):

    def __init__(self):
        super().__init__(identifier="avg")

    def aggregate(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Averages attention accross layers, heads, and max pools across the last
        dimension of attention matrix.

        Parameters
        ----------
        attention_matrix: torch.tensor(bs, n_layers, n_heads, seq_length, seq_length)
            Matrix of attention weights
        """
        attention = attention.mean(dim=1)
        attention = attention.mean(dim=1)
        attention, _ = attention.max(dim=2)
        return attention


@AttentionAggregator.register("attention-rollout")
class AttentionRollout(AttentionAggregator):

    def __init__(self):
        super().__init__(identifier="roll")

    def aggregate(self, attention_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute input Attention Rollout as described by Abnar and Zuidema 2020 (arXiv 2005.00928):

        From the paper: "Attention Rollout assumes that the identities of input tokens are linearly combined
        through the layers based on the attention weights. To adjust attention weights, it rolls out the weights
        to capture the propagation of information from input tokens to intermediate hidden embeddings."

        Calculation
        -----------
        Attention Rollout: A~
        Attention Weights: A

        A~(l_i) = A(l_i)A~(l_i−1) if i > j
        A~(l_i) = A(l_i) if i = j

        For input attention, i = n_layers, j = 0

        Parameters
        ----------
        attention_matrix: torch.tensor(bs, n_layers, n_heads, seq_length, seq_length)
            Full matrix of attention weights

        Outputs
        -------
        attention_rollout: Tuple[torch.tensor(bs, seq_length, seq_length)]
            Attention rollout weights at each layer
        """
        B, L, H, S, _ = attention_matrix.shape

        # Use a single attention graph by averaging all heads
        attn_avg_heads = attention_matrix.mean(dim=2) # (bs, n_layers, seq_length, seq_length)

        # To account for residual connections, we add an identity matrix to the attention matrix
        # and re-normalize the weights.
        residual_atttention_matrix = torch.eye(attn_avg_heads.shape[2]) # (seq_length, seq_length)
        augmented_attention_matrix = attn_avg_heads + residual_atttention_matrix # (bs, n_layers, seq_length, seq_length)
        augmented_attention_matrix = F.normalize(augmented_attention_matrix, dim=-1, p=1)

        rollout_attention = torch.zeros(augmented_attention_matrix.shape) # (bs, n_layers, seq_length, seq_length)

        # Roll out the weights
        rollout_attention[:,0] = augmented_attention_matrix[:,0]
        for i in range(1, L):
            rollout_attention[:,i] = torch.matmul(
                augmented_attention_matrix[:,i],
                rollout_attention[:,i-1]
            )


        # Return amount of attention the last layer's CLS token pays to the input tokens.
        return rollout_attention[:,-1,0]


@AttentionAggregator.register("attention-flow")
class AttentionFlow(AttentionAggregator):

    def __init__(self):
        super().__init__(identifier="flow")

    def _build_adjacency_matrix(self, avg_heads_attn_matrix: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Builds an adjacency matrix representing a Directed Acyclic Graph (DAG), in which the nodes are
        input tokens and hidden embeddings

        Parameters
        ----------
        avg_heads_attn_matrix: torch.tensor(n_layers, seq_length, seq_length)
            Attention matrix, averaged over heads if applicable

        Outputs
        -------
        adjacency_matrix: torch.tensor((n_layers + 1) * seq_length, (n_layers + 1) * seq_length)
            Adjacency matrix representing the DAG
        labels_to_index: Dict[str, int]
            Maps node labels to their numerical index from 0 to ((n_layers + 1) * seq_length)
            Nodes are labeled as `L{layer_index_in_sequence}_{token_index_in_sequence}`
            L0 corresponds to the input nodes
        """
        n_layers, seq_length, _ = avg_heads_attn_matrix.shape

        number_of_nodes = (n_layers + 1) * seq_length
        adj_mat = torch.zeros(number_of_nodes, number_of_nodes)
        labels_to_index = {}

        # The first nodes represent the input sequence tokens
        for token_index in range(seq_length):
            labels_to_index[f"L0_{token_index}"] = token_index

        # The remaining nodes represent the hidden embeddings
        for layer_index in range(1, n_layers + 1):
            for k_from in range(seq_length):
                index_from = (layer_index * seq_length) + k_from
                labels_to_index[f"L{layer_index}_{k_from}"] = index_from
                for k_to in range(seq_length):
                    index_to = ((layer_index - 1) * seq_length) + k_to
                    adj_mat[index_from][index_to] = avg_heads_attn_matrix[layer_index - 1][k_from][k_to]

        return adj_mat, labels_to_index

    def _compute_node_flow(
        self,
        G: nx.DiGraph,
        labels_to_index: Dict[str, int],
        input_node_labels: List[str],
        output_node_labels: List[str],
        seq_length: int
    ) -> torch.Tensor:
        """Compute the maximum flow through a DAG from the input node(s) to the output node(s)

        Parameters
        ----------
        G: nx.DiGraph
            Directed Acyclic Graph (DAG), in which the nodes are input tokens and hidden embeddings
        labels_to_index: Dict[str, int]
            Map of node labels to their numerical index
        input_node_labels: List[int]
            List of node labels which define the group of input nodes
        output_node_labels: List[int]
            List of node labels which define the group of output nodes
        seq_length: int
            Length of the input sequence over which attention was calculated

        Outputs
        -------
        flow_values: torch.tensor(num_nodes, num_nodes)
            Flow value matrix
        """
        number_of_nodes = len(labels_to_index)
        flow_values = torch.zeros(number_of_nodes, number_of_nodes)

        for key in output_node_labels:
            if key not in input_node_labels:
                current_layer = labels_to_index[key] // seq_length
                pre_layer = current_layer - 1
                u = labels_to_index[key]
                for inp_node_key in input_node_labels:
                    v = labels_to_index[inp_node_key]
                    flow_value = nx.maximum_flow_value(G, u, v)
                    flow_values[u][(pre_layer * seq_length) + v] = flow_value
                flow_values[u] /= flow_values[u].sum()

        return flow_values

    def aggregate(self, attention_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute input Attention Flow as described by Abnar and Zuidema 2020 (arXiv 2005.00928):

        From the paper: "Attention Flow considers the attention graph as a flow network. Using a maximum
        flow algorithm, it computes maximum flow values, from hidden embeddings (sources) to input tokens
        (sinks)."

        Parameters
        ----------
        attention_matrix: torch.tensor(bs, n_layers, n_heads, seq_length, seq_length)
            Full matrix of attention weights

        Outputs
        -------
        attention_flow: torch.tensor(bs, seq_length)
            Attention flow weights at the last layer's CLS token
        """
        # Use a single attention graph by averaging all heads
        attn_avg_heads = attention_matrix.mean(dim=2) # (bs, n_layers, seq_length, seq_length)
        bs, n_layers,  _, seq_length = attn_avg_heads.shape

        # To account for residual connections, we add an identity matrix to the attention matrix
        # and re-normalize the weights.
        residual_atttention_matrix = torch.eye(attn_avg_heads.shape[2]) # (seq_length, seq_length)
        augmented_attention_matrix = attn_avg_heads + residual_atttention_matrix # (bs, n_layers, seq_length, seq_length)
        augmented_attention_matrix = torch.nn.functional.normalize(augmented_attention_matrix, dim=-1, p=1)

        attention_flow = torch.zeros(bs, seq_length)

        # TODO: can this be batched?
        instance_matrices = torch.unbind(augmented_attention_matrix, dim=0)
        for idx, instance_matrix in enumerate(instance_matrices):

            # Build DiGraph from attention matrix
            A, labels_to_index = self._build_adjacency_matrix(instance_matrix)
            A = A.detach().cpu().numpy()
            res_G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())

            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    nx.set_edge_attributes(res_G, {(i,j): A[i,j]}, 'capacity')

            # Compute flow from input tokens to the [CLS] token in the last layer
            input_nodes = [f"L0_{k}" for k in range(seq_length)]
            output_nodes = [f'L{n_layers}_0']

            flow_values = self._compute_node_flow(res_G, labels_to_index, input_nodes, output_nodes, seq_length)

            # Extract flow values from the matrix
            cls_last_layer_index = labels_to_index[f'L{n_layers}_0']
            from_index = (n_layers - 1) * seq_length
            to_index = n_layers * seq_length

            final_layer_cls_attn = flow_values[cls_last_layer_index][from_index: to_index]

            attention_flow[idx] = final_layer_cls_attn

        return attention_flow


class Attention(nn.Module, Registrable):
    """Attention module base class"""

    def forward(self, **kwargs) -> torch.Tensor:
        """Perform module forward pass"""
        raise NotImplementedError("Implement forward Model")
