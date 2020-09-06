from typing import Dict, Tuple, Iterable, List, Optional, Union
from overrides import overrides

import numpy as np
import torch

import logging

from allennlp.common import JsonDict
from allennlp.data import Batch, Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util
from allennlp.training.metrics import Auc, CategoricalAccuracy, F1Measure

from ane_research.models.modules.attention.attention import Attention, AttentionAnalysisMethods
from ane_research.interpret.saliency_interpreters.captum_interpreter import CaptumCompatible


@Model.register("pair_sequence_classifier")
class PairSequenceClassifier(Model, CaptumCompatible):
    """
    """
    def __init__(
        self,
        vocab: Vocabulary,
        word_embeddings: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        attention: Attention,
        decoder: FeedForward,
        field_names: Tuple[str, str]
    ):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        self.loss = torch.nn.CrossEntropyLoss()

        #TODO check for appropriate dimensions.

        self.supported_attention_analysis_methods = [
            AttentionAnalysisMethods.weight_based,
            AttentionAnalysisMethods.norm_based
        ]
        self.field_names = field_names

    def forward_inner(
        self,
        embedded1: torch.Tensor,
        embedded2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        label: Union[torch.Tensor, None],
        output_attentions: List[AttentionAnalysisMethods],
        output_dict: JsonDict
    ) -> Union[None, torch.Tensor]:

        encoded1 = self.encoder(embedded1, mask1)
        encoded2 = self.encoder(embedded2, mask2)

        attention1_output = self.attention(encoded1, mask1, output_attentions)
        attention2_output = self.attention(encoded2, mask2, output_attentions)

        context1 = attention1_output[0]
        context2 = attention2_output[0]

        if output_attentions:
            attention1_dict = attention1_output[1]
            attention2_dict = attention2_output[1]
            weight_label = AttentionAnalysisMethods.weight_based
            norm_label = AttentionAnalysisMethods.norm_based

            if weight_label in output_attentions:
                output_dict[f"{weight_label.value}_1"] = attention1_dict[weight_label]
                output_dict[f"{weight_label.value}_2"] = attention2_dict[weight_label]
            if norm_label in output_attentions:
                output_dict[f"{norm_label.value}_1"] = attention1_dict[norm_label]
                output_dict[f"{norm_label.value}_2"] = attention2_dict[norm_label]

        diff = (context1 - context2).abs()
        prod = context1 * context2

        combined = torch.cat((context1, context2, diff, prod), dim=-1)

        # Decode
        logits = self.decoder(combined)
        output_dict["logits"] = logits

        if label is not None:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            nr_classes = len(self.vocab.get_index_to_token_vocabulary("labels").values())

            B, = label.shape
            label2 = label.unsqueeze(-1).expand(B, nr_classes)

            mask = torch.arange(nr_classes, device=logits.device).unsqueeze(0).expand(*probs.shape) == label2
            preds = probs[mask].unsqueeze(-1) # (bs, 1)
            return preds


    @overrides
    def forward(
        self,
        metadata,
        label=None,
        output_attentions: Optional[List[AttentionAnalysisMethods]] = None,
        **inputs
    ) -> JsonDict:

        # https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments
        if output_attentions is None:
            output_attentions = []

        output_dict = {}

        key1, key2 = self.field_names
        sentence1 = inputs[key1]
        sentence2 = inputs[key2]

        def embed(key, tokens):
            tokens_mask = util.get_text_field_mask(tokens)
            embedded_tokens = self.word_embeddings(tokens)
            output_dict[f"{key}_embedding"] = embedded_tokens
            return embedded_tokens, tokens_mask

        embedded1, mask1 = embed(key1, sentence1)
        embedded2, mask2 = embed(key2, sentence2)

        prediction = self.forward_inner(embedded1, embedded2, mask1, mask2, label, output_attentions, output_dict)
        if prediction is not None:
            output_dict["prediction"] = prediction

        logits = output_dict["logits"]
        class_probabilities = torch.nn.functional.softmax(logits, dim=-1)
        output_dict["class_probabilities"] = class_probabilities

        # A label is not necessary to perform a forward pass. There are situations where you don't have a label,
        # such as in a demo or when this model is a component in a larger model
        if label is not None:
            output_dict["actual"] = label
            loss = self.loss(output_dict["logits"], label)
            output_dict["loss"] = loss
            self.metrics["accuracy"](class_probabilities, label)

        return output_dict

    @overrides
    def get_field_names(self) -> Iterable[str]:
        return self.field_names

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

            key1, key2 = self.field_names
            tokens1 = model_input[key1]
            tokens2 = model_input[key2]
            label = model_input["label"]

            tokens_mask1 = util.get_text_field_mask(tokens1)
            tokens_mask2 = util.get_text_field_mask(tokens2)
            embedded_tokens1 = self.word_embeddings(tokens1)
            embedded_tokens2 = self.word_embeddings(tokens2)

            output_dict = {}
            output_dict[f"{key1}_embedding"] = embedded_tokens1
            output_dict[f"{key2}_embedding"] = embedded_tokens2

            return (embedded_tokens1, embedded_tokens2), None, (tokens_mask1, tokens_mask2, label, output_dict)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "accuracy": self.metrics["accuracy"].get_metric(reset=reset),
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
        class_probabilities = output_dict['class_probabilities']
        predictions = class_probabilities.cpu().data.numpy()
        output_dict['label'] = np.argmax(predictions, axis=1)
        return output_dict


class _CaptumSubModel(torch.nn.Module):

    def __init__(self, model: PairSequenceClassifier):
        super().__init__()
        self.model = model

    @overrides
    def forward(self, *inputs):
        # (embedded1, embedded2, mask1, mask2, label, output_dict)
        # add output_attentions=None
        inputs_no_attention = inputs[:5]+(None,)+inputs[5:]
        return self.model.forward_inner(*inputs_no_attention)
