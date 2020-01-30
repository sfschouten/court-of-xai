'''Implement Binary Classifier model described in Attention is not Explanation (Jain and Wallace) 2019
   https://arxiv.org/pdf/1902.10186.pdf
'''

from typing import Dict
from overrides import overrides

import numpy as np
import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util
from allennlp.training.metrics import Auc, CategoricalAccuracy, F1Measure

from ane_research.models.modules import Attention


@Model.register('jain_wallace_attention_binary_classifier')
class JWAED(Model):
  '''
    AllenNLP implementation of the Encoder/decoder with attention model for binary classification as described
    in 'Attention is Not Explanation' (Jain and Wallace 2019 - https://arxiv.org/pdf/1902.10186.pdf)
  '''
  def __init__(self, vocab: Vocabulary, word_embeddings: TextFieldEmbedder, encoder: Seq2SeqEncoder,
               attention: Attention, decoder: FeedForward):
    super().__init__(vocab)
    self.word_embeddings = word_embeddings
    self.encoder = encoder
    self.attention = attention
    self.decoder = decoder
    self.metrics = {
      'accuracy': CategoricalAccuracy(),
      'f1_measure': F1Measure(positive_label=1),
      'auc': Auc(positive_label=1)
    }
    self.loss = torch.nn.BCEWithLogitsLoss()

  @overrides
  def forward(self, tokens: Dict[str, torch.LongTensor], label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
    # AllenNLP models return an output dictionary with information needed for future calculations
    output_dict = {}

    # Encode
    tokens_mask = util.get_text_field_mask(tokens)
    embedded_tokens = self.word_embeddings(tokens)
    output_dict['embedding'] = embedded_tokens
    encoded_tokens = self.encoder(embedded_tokens, tokens_mask)

    # Compute attention
    attention = self.attention(encoded_tokens, tokens_mask)
    output_dict['attention'] = attention
    context = (attention.unsqueeze(-1) * encoded_tokens).sum(1)

    # Decode
    logits = self.decoder(context)
    prediction = torch.sigmoid(logits)
    output_dict['prediction'] = prediction
    class_probabilities = torch.cat((1 - prediction, prediction), dim=1)
    output_dict['class_probabilities'] = class_probabilities

    # A label is not necessary to perform a forward pass. There are situations where you don't have a label,
    # such as in a demo or when this model is a component in a larger model
    if label is not None:
      loss = self.loss(logits, label.unsqueeze(-1).float())
      output_dict['loss'] = loss
      self.metrics['accuracy'](class_probabilities, label)
      self.metrics['f1_measure'](class_probabilities, label)
      self.metrics['auc'](prediction.squeeze(-1), label)

    return output_dict

  def forward_with_feature_erasure(self, tokens, label) -> Dict[str, torch.Tensor]:
    tokens_mask = util.get_text_field_mask(tokens)
    batch_size, max_sequence_length = tokens_mask.shape
    predictions = torch.zeros((batch_size, max_sequence_length, 1))
    class_probabilities = torch.zeros((batch_size, max_sequence_length, 2))
    sequences = tokens['tokens']
    for i in range(1, max_sequence_length - 1):
      batch_tokens = {'tokens': torch.cat([sequences[:, :i], sequences[:, i+1:]], dim=-1)}
      batch_masks = torch.cat([tokens_mask[:, :i], tokens_mask[:, i+1:]], dim=-1)
      embedded_tokens = self.word_embeddings(batch_tokens)
      encoded_tokens = self.encoder(embedded_tokens, batch_masks)
      attention = self.attention(encoded_tokens, batch_masks)
      context = (attention.unsqueeze(-1) * encoded_tokens).sum(1)
      logits = self.decoder(context)
      prediction = torch.sigmoid(logits)
      predictions[:, i] = prediction
      class_probabilities[:, i] = torch.cat((1 - prediction, prediction), dim=1)
    return {
      'prediction': predictions,
      'class_probabilities': class_probabilities
    }

  @overrides
  def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    # f1 get_metric returns (precision, recall, f1)
    precision, recall, f1_measure = self.metrics['f1_measure'].get_metric(reset=reset)
    return {
      'precision': precision,
      'recall': recall,
      'f1_measure': f1_measure,
      'accuracy': self.metrics['accuracy'].get_metric(reset=reset),
      'auc': self.metrics['auc'].get_metric(reset=reset)
    }

  @overrides
  def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    '''
    Does a simple argmax over the class probabilities, converts indices to string labels, and
    adds a ``'label'`` key to the dictionary with the result.
    '''
    # print(output_dict)
    class_probabilities = output_dict['class_probabilities']
    # print(f'Class probabilities shape: {class_probabilities.shape}')
    predictions = class_probabilities.cpu().data.numpy()
    # print(predictions.shape)
    output_dict['label'] = np.argmax(predictions, axis=1)
    # print(f'The new label shape is: { output_dict["label"][0].shape}')
    return output_dict
