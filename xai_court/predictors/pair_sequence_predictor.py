from overrides import overrides
import numpy as np
import torch
from typing import List, Dict

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.batch import Batch
from allennlp.data.fields import LabelField
from allennlp.nn import util
from allennlp.predictors.predictor import Predictor

from xai_court.interpret.saliency_interpreters.attention import AttentionModelPredictor
from xai_court.models.modules.attention.attention import AttentionAnalysisMethods


@Predictor.register("pair_sequence_classifier")
class PairSequencePredictor(Predictor, AttentionModelPredictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence1 = json_dict["sentence1"]
        sentence2 = json_dict["sentence2"]

        instance = self._dataset_reader.text_to_instance(sentence1, sentence2)
        return instance

    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        instance = self._json_to_instance(json_dict)
        label_dict = self._model.vocab.get_index_to_token_vocabulary("labels")
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return {
            "instance": instance,
            "prediction": self.predict_instance(instance),
            "label": all_labels
        }

    @overrides
    def predict_instance(self, instance: Instance, **kwargs) -> JsonDict:
        outputs = self._model.forward_on_instance(instance, **kwargs)
        return sanitize(outputs)

    @overrides
    def predictions_to_labeled_instances(self, instance: Instance, outputs: Dict[str, np.ndarray]) -> Instance:
        new_instance = instance.duplicate()
        label = np.argmax(outputs["class_probabilities"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]

    @overrides
    def get_suitable_aggregators(self):
        return [type(None)]

    @overrides
    def get_attention_based_salience_for_instance(
            self, 
            labeled_instance: Instance, 
            analysis_method: AttentionAnalysisMethods,
            aggregate_method: None 
        ) -> JsonDict:
        output = self.predict_instance(labeled_instance, output_attentions=[analysis_method])

        attention1 = output[f"{analysis_method.value}_1"]
        attention2 = output[f"{analysis_method.value}_2"]

        fields = self._model.field_names
        return { fields[0] : attention1, fields[1] : attention2 }
