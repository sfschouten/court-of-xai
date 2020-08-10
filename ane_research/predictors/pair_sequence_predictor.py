from copy import deepcopy
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

from ..interpret.saliency_interpreters.attention_interpreter import AttentionModelPredictor

@Predictor.register('pair_sequence_classifier')
class PairSequencePredictor(Predictor, AttentionModelPredictor):

  @overrides
  def _json_to_instance(self, json_dict: JsonDict) -> Instance:
    sentence1 = json_dict['sentence1']
    sentence2 = json_dict['sentence2']

    # Assuming it's already tokenized.
    tokens1 = sentence1.split()
    tokens2 = sentence2.split()

    instance = self._dataset_reader.text_to_instance(tokens1, tokens2)
    return instance

  @overrides
  def predict_json(self, json_dict: JsonDict) -> JsonDict:
    instance = self._json_to_instance(json_dict)
    label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
    all_labels = [label_dict[i] for i in range(len(label_dict))]

    return {
      'instance': instance,
      'prediction': self.predict_instance(instance),
      'label': all_labels
    }

  @overrides
  def predictions_to_labeled_instances(self, instance: Instance, outputs: Dict[str, np.ndarray]) -> Instance:
    new_instance = deepcopy(instance)
    label = np.argmax(outputs['class_probabilities'])
    new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
    return [new_instance]

  def get_attention_based_salience_for_instance(self, labeled_instance: Instance):
    output = self.predict_instance(labeled_instance)

    attention1 = output['attention1']
    attention2 = output['attention2']
   
    fields = self._model.field_names
    return { fields[0] : attention1, fields[1] : attention2 }


