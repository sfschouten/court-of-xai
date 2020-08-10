from typing import Dict, Tuple, Iterable
from overrides import overrides

import numpy as np
import torch

import logging

from allennlp.data import Batch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util
from allennlp.training.metrics import Auc, CategoricalAccuracy, F1Measure

from ane_research.models.modules import Attention
from ane_research.interpret.saliency_interpreters.captum_interpreter import CaptumCompatible


@Model.register('pair_sequence_classifier')
class PairSequenceClassifier(Model, CaptumCompatible):
  '''
  '''
  def __init__(self, vocab: Vocabulary, word_embeddings: TextFieldEmbedder, encoder: Seq2SeqEncoder,
          attention: Attention, decoder: FeedForward, field_names: Tuple[str, str]):
    super().__init__(vocab)
    self.word_embeddings = word_embeddings
    self.encoder = encoder
    self.attention = attention
    self.decoder = decoder
    self.metrics = {
      'accuracy': CategoricalAccuracy(),
    }
    self.loss = torch.nn.CrossEntropyLoss()

    #TODO check for appropriate dimensions.

    self.field_names = field_names 

  def forward_inner(self, embedded1, embedded2, mask1, mask2, label, output_dict):

    encoded1 = self.encoder(embedded1, mask1)
    encoded2 = self.encoder(embedded2, mask2)

    def compute_attention(encoded_tokens, tokens_mask):
        attention = self.attention(encoded_tokens, tokens_mask)
        context = (attention.unsqueeze(-1) * encoded_tokens).sum(1)
        return context, attention

    context1, attn1 = compute_attention(encoded1, mask1)
    context2, attn2 = compute_attention(encoded2, mask2)

    output_dict['attention1'] = attn1.tolist()
    output_dict['attention2'] = attn2.tolist()
    
    diff = (context1 - context2).abs()
    prod = context1 * context2

    combined = torch.cat((context1, context2, diff, prod), dim=-1)

    # Decode
    logits = self.decoder(combined)
    output_dict['logits'] = logits

    probs = torch.nn.functional.softmax(logits, dim=-1)
    nr_classes = len(self.vocab.get_index_to_token_vocabulary('labels').values())
    
    B, = label.shape
    label2 = label.unsqueeze(-1).expand(B, nr_classes)
    
    mask = torch.arange(nr_classes).unsqueeze(0).expand(*probs.shape) == label2
    preds = probs[mask]
    return preds

  
  @overrides
  def forward(self, label, metadata, **inputs):
    output_dict = {}

    key1, key2 = self.field_names
    sentence1 = inputs[key1]
    sentence2 = inputs[key2]

    def embed(key, tokens):
      tokens_mask = util.get_text_field_mask(tokens)
      embedded_tokens = self.word_embeddings(tokens)
      output_dict[f'{key}_embedding'] = embedded_tokens
      return embedded_tokens, tokens_mask

    embedded1, mask1 = embed(key1, sentence1)
    embedded2, mask2 = embed(key2, sentence2)

    prediction = self.forward_inner(embedded1, embedded2, mask1, mask2, label, output_dict)
    output_dict['prediction'] = prediction
    
    logits = output_dict['logits']
    class_probabilities = torch.nn.functional.softmax(logits, dim=-1)
    output_dict['class_probabilities'] = class_probabilities

    # A label is not necessary to perform a forward pass. There are situations where you don't have a label,
    # such as in a demo or when this model is a component in a larger model
    if label is not None:
      loss = self.loss(output_dict['logits'], label)
      output_dict['loss'] = loss
      self.metrics['accuracy'](class_probabilities, label)

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
      label = model_input['label']

      tokens_mask1 = util.get_text_field_mask(tokens1)
      tokens_mask2 = util.get_text_field_mask(tokens2)
      embedded_tokens1 = self.word_embeddings(tokens1)
      embedded_tokens2 = self.word_embeddings(tokens2)

      output_dict = {}
      output_dict[f'{key1}_embedding'] = embedded_tokens1
      output_dict[f'{key2}_embedding'] = embedded_tokens2

      return (embedded_tokens1, embedded_tokens2), None, (tokens_mask1, tokens_mask2, label, output_dict)

  @overrides
  def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    return {
      'accuracy': self.metrics['accuracy'].get_metric(reset=reset),
    }

  @overrides
  def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    ''' 
    Does a simple argmax over the class probabilities, converts indices to string labels, and
    adds a ``'label'`` key to the dictionary with the result.
    '''
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
    return self.model.forward_inner(*inputs)

