from typing import Dict, List
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from transformers import DistilBertTokenizer


@DatasetReader.register('imdb_csv')
class ImdbCsvDatasetReader(DatasetReader):
  '''Reads IMDB reviews and labels from a CSV file'''
  def __init__(self,
    pretrained_tokenizer: str = None,
    token_indexers: Dict[str, TokenIndexer] = None,
    max_review_length: int = None,
    lazy: bool = False
  ):
    super().__init__(lazy=lazy)
    self.max_review_length = max_review_length
    self.pretrained_tokenizer = pretrained_tokenizer

    if self.pretrained_tokenizer:
      self._tokenizer = PretrainedTransformerTokenizer(self.pretrained_tokenizer)
      self._token_indexers: Dict[str, TokenIndexer] = {
        "tokens": PretrainedTransformerIndexer(self.pretrained_tokenizer)
      }
    else:
      self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

  @overrides
  def _read(self, file_path):
    with open(cached_path(file_path), "r") as data_file:
      for line in data_file.readlines():
        line = line.strip("\n")
        if not line:
          continue
        review, sentiment = line.split(',')
        review_tokens = review.split(' ')
        if self.max_review_length and len(review_tokens) > self.max_review_length:
          continue
        if self.pretrained_tokenizer:
          instance = self.text_to_instance(review, sentiment)
        else:
          instance = self.text_to_instance(review_tokens, sentiment)
        if instance is not None:
          yield instance

  @overrides
  def text_to_instance(self, tokens: List[str], sentiment: str) -> Instance:
    if self.pretrained_tokenizer:
      tokenized_sentence = self._tokenizer.tokenize(tokens)
      text = TextField(tokenized_sentence, self._token_indexers)
    else:
      text = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)

    label_field = LabelField(int(sentiment), skip_indexing=True)
    fields = {
      'tokens': text,
      'label': label_field
    }
    return Instance(fields)
