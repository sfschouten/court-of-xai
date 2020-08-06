# '''
# DistilBERT code taken from the HuggingFace Transformer 2.11.0 library with minor modifications
# Allen NLP compatibility code taken from https://github.com/allenai/allennlp/pull/4495/files
# '''
from copy import deepcopy
import math
from typing import Dict, List, Optional

from allennlp.data.batch import Batch
from allennlp.common import FromParams, JsonDict
from allennlp.data import TextFieldTensors, Vocabulary, Instance
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
import numpy as np
from overrides import overrides
import torch
from torch import Tensor
import torch.nn as nn
from transformers.modeling_auto import AutoModel
from transformers.modeling_utils import PreTrainedModel

from ane_research.interpret.saliency_interpreters.captum_interpreter import CaptumCompatible
from ane_research.models.modules.architectures.transformer import Transformer, Embeddings


class DistilBertEncoder(torch.nn.Module, FromParams):
    def __init__(
        self,
        n_layers: int = 6,
        n_heads: int = 12,
        dim: int = 768,
        hidden_dim: int = 4*768,
        ffn_activation: str = "gelu",
        attention_activation: str = "softmax",
        attention_dropout: float = 0.2
    ):
        super().__init__()
        self.n_layers=n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.ffn_activation = ffn_activation
        self.attention_activation = attention_activation
        self.attention_dropout =  attention_dropout

        self.transformer = Transformer(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dim=self.dim,
            hidden_dim=self.hidden_dim,
            ffn_activation=self.ffn_activation,
            attention_activation=self.attention_activation,
            attention_dropout=self.attention_dropout)

    @classmethod
    def from_huggingface_model(cls,
        model: PreTrainedModel,
        ffn_activation: str,
        attention_activation: str,
        attention_dropout: float
    ):
        config = model.config
        encoder = cls(
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            ffn_activation=ffn_activation,
            attention_activation=attention_activation,
            attention_dropout=attention_dropout
        )
        # After creating the encoder, we copy weights over from the transformer.  This currently
        # requires that the internal structure of the text side of this encoder *exactly matches*
        # the internal structure of whatever transformer you're using. 
        encoder_parameters = dict(encoder.named_parameters())
        for name, parameter in model.named_parameters():
            if name.startswith("encoder."):
                name = name[8:]
                name = name.replace("LayerNorm", "layer_norm")
                if name not in encoder_parameters:
                    raise ValueError(
                        f"Couldn't find a matching parameter for {name}. Is this transformer "
                        "compatible with the joint encoder you're using?"
                    )
                encoder_parameters[name].data.copy_(parameter.data)

        return encoder

    @overrides
    def forward(
        self,
        attention_mask: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None
    ):
        return self.transformer(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )


@Model.register("distilbert_sequence_classification")
@Model.register("distilbert_sequence_classification_from_huggingface", constructor="from_huggingface_model_name")
class DistilBertForSequenceClassification(Model, CaptumCompatible):
    def __init__(
        self,
        vocab: Vocabulary,
        embeddings: Embeddings,
        encoder: DistilBertEncoder,
        num_labels: int,
        seq_classif_dropout: float
    ):
        super().__init__(vocab)

        self.embeddings = embeddings
        self.encoder = encoder
        self.num_labels = num_labels
        self.seq_classif_dropout = seq_classif_dropout

        self.pre_classifier = nn.Linear(self.encoder.dim, self.encoder.dim)
        self.classifier = nn.Linear(self.encoder.dim, self.num_labels)
        self.dropout = nn.Dropout(self.seq_classif_dropout)

        self.metrics = {
            'accuracy': CategoricalAccuracy()
        }
        self.loss = torch.nn.CrossEntropyLoss()

    @classmethod
    def from_huggingface_model_name(
        cls,
        vocab: Vocabulary,
        model_name: str,
        ffn_activation: str,
        attention_activation: str,
        attention_dropout: float,
        num_labels: int,
        seq_classif_dropout: float
    ):
        transformer = AutoModel.from_pretrained(model_name)
        embeddings = deepcopy(transformer.embeddings)
        encoder = DistilBertEncoder.from_huggingface_model(
            model=transformer,
            ffn_activation=ffn_activation,
            attention_dropout=attention_dropout,
            attention_activation=attention_activation
        )
        return cls(
            vocab=vocab,
            embeddings=embeddings,
            encoder=encoder,
            num_labels=num_labels,
            seq_classif_dropout=seq_classif_dropout
        )

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self.metrics['accuracy'].get_metric(reset=reset)
        }

    def forward_inner(self,
        embedded_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool,
        output_dict: JsonDict
    ):
        # (bs, seq_len) -> (num_hidden_layers, batch, num_heads, seq_length, seq_length)
        head_mask = attention_mask.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
        head_mask = head_mask.expand(self.encoder.n_layers, -1, self.encoder.n_heads, -1, attention_mask.shape[1])
        encoder_output = self.encoder(
            inputs_embeds=embedded_tokens,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions
        )

        hidden_state = encoder_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        output_dict["logits"] = logits

        if output_attentions:
            output_dict["attention"] = encoder_output[1][0]

        class_probabilities = torch.nn.Softmax(dim=-1)(logits)
        return class_probabilities

    @overrides
    def forward(
        self,
        tokens: TextFieldTensors,
        label: torch.LongTensor = None,
        output_attentions: bool = False
    ):
        output_dict = {}

        input_ids = tokens["tokens"]["token_ids"] # (bs, seq_len)
        attention_mask = tokens["tokens"]["mask"] # (bs, seq_len)

        embedding_output = self.embeddings(input_ids) # (bs, seq_len, dim)
        class_probabilities = self.forward_inner(
            embedded_tokens=embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_dict=output_dict
        )
        output_dict["class_probabilities"] = class_probabilities

        if label is not None:
            loss = self.loss(output_dict["logits"].view(-1, self.num_labels), label.view(-1))
            output_dict['loss'] = loss
            self.metrics['accuracy'](class_probabilities, label)

        return output_dict

    @overrides
    def forward_on_instances(self, instances: List[Instance]) -> List[Dict[str, np.ndarray]]:
        # An exact copy of the original method, but includes the flag to output attention scores
        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self.make_output_human_readable(self(**model_input, output_attentions=True))
            instance_separated_output: List[Dict[str, np.ndarray]] = [
                {} for _ in dataset.instances
            ]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        '''
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``'label'`` key to the dictionary with the result.
        '''
        output_dict['label'] = torch.argmax(output_dict['class_probabilities'], dim=1)
        return output_dict

    @overrides
    def captum_sub_model(self):
        return _CaptumSubModel(self)

    @overrides
    def instances_to_captum_inputs(self, labeled_instances):
        batch_size = len(labeled_instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            batch = Batch(labeled_instances)
            batch.index_instances(self.vocab)
            model_input = util.move_to_device(batch.as_tensor_dict(), cuda_device)
            input_ids = model_input["tokens"]["tokens"]["token_ids"]
            label = model_input["label"]
            attention_mask = model_input["tokens"]["tokens"]["mask"]
            embedded_tokens = self.embeddings(input_ids)

            output_dict = {}
            output_dict["embedding"] = embedded_tokens
            return (embedded_tokens,), label, (attention_mask, output_dict)

class _CaptumSubModel(torch.nn.Module):

    def __init__(self, model: DistilBertForSequenceClassification):
        super().__init__()
        self.model = model

    @overrides
    def forward(self,
        embedded_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        output_dict: JsonDict
    ):
        return self.model.forward_inner(
            embedded_tokens=embedded_tokens,
            attention_mask=attention_mask,
            output_attentions=True,
            output_dict=output_dict
        )
