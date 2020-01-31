'''AllenNLP Predictor class for the Jain Wallace Attention Binary Classification model'''

from copy import deepcopy
from overrides import overrides
import numpy as np
import torch
from typing import List, Dict

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import LabelField
from allennlp.nn import util
from allennlp.predictors.predictor import Predictor


@Predictor.register('jain_wallace_attention_binary_classification_predictor')
class JWAEDPredictor(Predictor):
  '''Predictor wrapper for the Jain Wallace Attention Binary Classification model'''

  @overrides
  def predict_json(self, json_dict: JsonDict) -> JsonDict:
    tokens = json_dict['tokens']
    instance = self._dataset_reader.text_to_instance(tokens=tokens)
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

  def predict_batch_instances_with_feature_erasure(self, instances: List[Instance]) -> Dict[str, np.ndarray]:
    dataset = Batch(instances)
    dataset.index_instances(self._model.vocab)
    model_input = util.move_to_device(dataset.as_tensor_dict(), self._model._get_prediction_device())
    outputs = self._model.forward_with_feature_erasure(**model_input)
    # predictions tensor(batch size x maximum sequence length x 1)
    # sanitize
    for key, value in outputs.items():
      outputs[key] = value.detach().cpu().numpy()
    return outputs

  def batch_instances_normalized_gradient_based_feature_importance(self, instances: List[Instance]) -> List[np.array]:
    '''Get normalized gradient feature importance as per Jain and Wallace algorithm 1 (gt)'''
    batch_gradients, batch_outputs = self.get_gradients(instances)
    # See Attention is Not Explanation p4. - Algorithm 1 Feature Importance Computations
    batch_embedding = batch_outputs['embedding'].detach().cpu().numpy()
    batch_embedding_gradient = batch_gradients['grad_input_1'] # (batch x max sequence length x embedding dim)
    batch_gradient_feature_importance = (batch_embedding * batch_embedding_gradient).sum(-1)
    # normalize
    for i in range(len(instances)):
      batch_gradient_magnitudes = np.abs(batch_gradient_feature_importance[i,:])
      batch_gradient_feature_importance[i,:] = batch_gradient_magnitudes / batch_gradient_magnitudes.sum()
    return batch_gradient_feature_importance
