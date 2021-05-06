"""
Implement Binary Classifier model described in Attention is not Explanation (Jain and Wallace) 2019 (arXiv 1902.10186)
"""
from typing import Dict, Iterable, List, Optional
from overrides import overrides

import numpy as np
import torch

from allennlp.common import JsonDict
from allennlp.data import Batch, Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util
from allennlp.training.metrics import Auc, CategoricalAccuracy, F1Measure

from xai_court.models.modules.attention import Attention, AttentionAnalysisMethods
from xai_court.interpret.saliency_interpreters.captum_interpreter import CaptumCompatible


@Model.register("jain_wallace_attention_binary_classifier")
class JWAED(Model, CaptumCompatible):
    """
    AllenNLP implementation of the Encoder/decoder with attention model for binary classification as described
    in 'Attention is Not Explanation' (Jain and Wallace 2019 - https://arxiv.org/pdf/1902.10186.pdf)
    """
    def __init__(self, vocab: Vocabulary, word_embeddings: TextFieldEmbedder, encoder: Seq2SeqEncoder,
                attention: Attention, decoder: FeedForward):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1_measure": F1Measure(positive_label=1),
            "auc": Auc(positive_label=1)
        }
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.supported_attention_analysis_methods = [
            AttentionAnalysisMethods.weight_based,
            AttentionAnalysisMethods.norm_based
        ]

    def forward_inner(
        self,
        embedded_tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        output_attentions: List[AttentionAnalysisMethods],
        output_dict: JsonDict
    ) -> torch.Tensor:
        encoded_tokens = self.encoder(embedded_tokens, tokens_mask)

        # Compute attention
        attention_output = self.attention(encoded_tokens, tokens_mask, output_attentions)
        context = attention_output[0] # (bs, hidden_dim)

        # Decode
        logits = self.decoder(context) # (bs, 1)
        output_dict["logits"] = logits
        prediction = torch.sigmoid(logits)

        if output_attentions:
            attention_dict = attention_output[1]
            weight_label = AttentionAnalysisMethods.weight_based
            norm_label = AttentionAnalysisMethods.norm_based

            if weight_label in output_attentions:
                output_dict[weight_label] = attention_dict[weight_label]
            if norm_label in output_attentions:
                output_dict[norm_label] = attention_dict[norm_label]

        return prediction

    @overrides
    def forward(
        self,
        tokens: Dict[str, torch.LongTensor],
        label: torch.LongTensor = None,
        output_attentions: Optional[List[AttentionAnalysisMethods]] = None
    ) -> JsonDict:

        # https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments
        if output_attentions is None:
            output_attentions = []

        # AllenNLP models return an output dictionary with information needed for future calculations
        output_dict = {}

        # Encode
        tokens_mask = util.get_text_field_mask(tokens)
        embedded_tokens = self.word_embeddings(tokens)
        output_dict["embedding"] = embedded_tokens

        prediction = self.forward_inner(embedded_tokens, tokens_mask, output_attentions, output_dict) # (bs, 1)
        output_dict["prediction"] = prediction

        class_probabilities = torch.cat((1 - prediction, prediction), dim=1)
        output_dict["class_probabilities"] = class_probabilities

        # A label is not necessary to perform a forward pass. There are situations where you don"t have a label,
        # such as in a demo or when this model is a component in a larger model
        if label is not None:
            output_dict["actual"] = label
            loss = self.loss(output_dict["logits"], label.unsqueeze(-1).float())
            output_dict["loss"] = loss
            self.metrics["accuracy"](class_probabilities, label)
            self.metrics["f1_measure"](class_probabilities, label)
            self.metrics["auc"](prediction.squeeze(-1), label)

        return output_dict

    @overrides
    def captum_sub_model(self):
        return _CaptumSubModel(self)

    @overrides
    def get_field_names(self) -> Iterable[str]:
        # We already assume a field called tokens in forward, so we do so here too.
        return ("tokens",)

    @overrides
    def instances_to_captum_inputs(self, labeled_instances):
        batch_size = len(labeled_instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            batch = Batch(labeled_instances)
            batch.index_instances(self.vocab)
            model_input = util.move_to_device(batch.as_tensor_dict(), cuda_device)

            tokens = model_input["tokens"]
            label = model_input["label"]

            tokens_mask = util.get_text_field_mask(tokens)
            embedded_tokens = self.word_embeddings(tokens)

            output_dict = {}
            output_dict["embedding"] = embedded_tokens
            # target = None because output is a single scalar
            return (embedded_tokens,), None, (tokens_mask, output_dict)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # f1 get_metric returns (precision, recall, f1)
        f1_measure = self.metrics["f1_measure"].get_metric(reset=reset)
        return {
            "precision": f1_measure["precision"],
            "recall": f1_measure["recall"],
            "f1_measure": f1_measure["f1"],
            "accuracy": self.metrics["accuracy"].get_metric(reset=reset),
            "auc": self.metrics["auc"].get_metric(reset=reset)
        }

    @overrides
    def forward_on_instances(self, instances: List[Instance], **kwargs) -> List[Dict[str, np.ndarray]]:
        # An exact copy of the original method, but supports kwargs
        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self.make_output_human_readable(self(**model_input, **kwargs))
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
    def forward_on_instance(self, instance: Instance, **kwargs) -> Dict[str, np.ndarray]:
        return self.forward_on_instances([instance], **kwargs)[0]

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a `label` key to the dictionary with the result.
        """
        class_probabilities = output_dict["class_probabilities"]
        predictions = class_probabilities.cpu().data.numpy()
        output_dict["label"] = np.argmax(predictions, axis=1)
        return output_dict


class _CaptumSubModel(torch.nn.Module):

    def __init__(self, model: JWAED):
        super().__init__()
        self.model = model

    @overrides
    def forward(self, word_embeddings, tokens_mask, output_dict):
        return self.model.forward_inner(
            embedded_tokens=word_embeddings,
            tokens_mask=tokens_mask,
            output_attentions=None,
            output_dict=output_dict
        )
