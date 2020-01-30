from typing import Dict, List
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token


@DatasetReader.register('imdb_csv')
class ImdbCsvDatasetReader(DatasetReader):
  '''Reads IMDB reviews and labels from a CSV file'''
  def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, lazy: bool = False) -> None:
    super().__init__(lazy=lazy)
    self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

  @overrides
  def _read(self, file_path):
    with open(cached_path(file_path), "r") as data_file:
      for line in data_file.readlines():
        line = line.strip("\n")
        if not line:
          continue
        review, sentiment = line.split(',')
        review = review.split(' ')
        instance = self.text_to_instance(review, sentiment)
        if instance is not None:
          yield instance

  @overrides
  def text_to_instance(self, tokens: List[str], sentiment: str) -> Instance:
    text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)
    fields: Dict[str, Field] = {"tokens": text_field}
    fields['label'] = LabelField(sentiment)
    return Instance(fields)
