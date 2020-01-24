'''Implement Binary Classifier model described in Attention is not Explanation (Jain and Wallace) 2019
   https://arxiv.org/pdf/1902.10186.pdf
'''

from typing import Dict
from overrides import overrides

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util
from allennlp.training.metrics import Auc, CategoricalAccuracy, F1Measure

from ane_research.models.modules.attention import AdditiveAttention


@Model.register('jain_wallace_attention_binary_classifier')
class JWAED(Model):
  '''
    Encoder/decoder with attention model for binary classification as described in 'Attention is Not
    Explanation' (Jain and Wallace 2019) - Jain and Wallace) 2019 (https://arxiv.org/pdf/1902.10186.pdf)
  '''
  def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder, encoder: Seq2SeqEncoder, 
               decoder: FeedForward):
    super().__init__(vocab)
    self.text_field_embedder = text_field_embedder
    self.num_classes = self.vocab.get_vocab_size('labels')
    self.encoder = encoder
    self.attention = AdditiveAttention(encoder.get_output_dim())
    self.decoder = decoder
    self.metrics = {
      'accuracy': CategoricalAccuracy(),
      'f1_measure': F1Measure(positive_label=1),
      'auc': Auc(positive_label=1)
    }
    self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')

  @overrides
  def forward(self, tokens: Dict[str, torch.LongTensor], label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
    # encode
    tokens_mask = util.get_text_field_mask(tokens)
    embedded_tokens = self.text_field_embedder(tokens)
    tokens_mask = util.get_text_field_mask(tokens)
    encoded_tokens = self.encoder(embedded_tokens, tokens_mask)

    # compute attention
    attention = self.attention(encoded_tokens, tokens_mask)
    output_dict = {'attention': attention}
    context = (attention.unsqueeze(-1) * encoded_tokens).sum(1)

    # decode
    logits = self.decoder(context)
    probabilities = torch.sigmoid(logits)
    predictions = torch.cat((1 - probabilities, probabilities), dim=1)

    if label is not None:
      loss = self.loss(logits, label.unsqueeze(-1).float())
      output_dict['loss'] = loss.mean(1).sum()

      target = label.squeeze(-1)
      self.metrics['accuracy'](predictions, target)
      self.metrics['f1_measure'](predictions, target)
      self.metrics['auc'](probabilities.squeeze(-1), target)

    return output_dict

  @overrides
  def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    precision, recall, f1_measure = self.metrics['f1_measure'].get_metric(reset=reset)
    return {
      # f1 get_metric returns (precision, recall, f1)
      'positive/precision': precision,
      'positive/recall': recall,
      'f1_measure': f1_measure,
      'accuracy': self.metrics['accuracy'].get_metric(reset=reset),
      'auc': self.metrics['auc'].get_metric(reset=reset)
    }
