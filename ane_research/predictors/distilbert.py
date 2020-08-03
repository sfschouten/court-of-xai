'''AllenNLP Predictor class for a DistilBERT Sequence Classification model'''

from copy import deepcopy
from typing import Dict, List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.batch import Batch
from allennlp.data.fields import LabelField
from allennlp.nn import util
from allennlp.predictors.predictor import Predictor
import numpy as np
from overrides import overrides
import torch


@Predictor.register("distilbert_sequence_classification")
@Predictor.register("distilbert_sequence_classification_from_huggingface")
class DistilBertForSequenceClassificationPredictor(Predictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = json_dict['tokens']
        sentiment = json_dict.get('sentiment')
        instance = self._dataset_reader.text_to_instance(tokens=tokens, sentiment=sentiment)
        return instance

    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        instance = self._json_to_instance(json_dict)
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        output = self.predict_instance(instance)
        return {
          'tokens': json_dict['tokens'],
          'class_probabilities': output['class_probabilities'],
          'predicted_sentiment': label_dict[int(np.argmax(output['class_probabilities']))],
          'actual_sentiment': json_dict.get('sentiment')
        }

    @overrides
    def predictions_to_labeled_instances(self, instance: Instance, outputs: Dict[str, np.ndarray]) -> List[Instance]:
        new_instance = deepcopy(instance)
        label = np.argmax(outputs['class_probabilities'])
        new_instance.add_field("label", LabelField(int(label)))
        return [new_instance]

    def get_attention_based_salience_for_instance(self, labeled_instance: Instance):
        output = self.predict_instance(labeled_instance)
        return output['attention']
