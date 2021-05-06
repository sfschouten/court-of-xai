'''AllenNLP Predictor class for a DistilBERT Sequence Classification model'''

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

from xai_court.models.modules.attention.attention import AttentionAnalysisMethods, AttentionAggregator, AttentionAverager, AttentionRollout, AttentionFlow
from xai_court.interpret.saliency_interpreters.attention import AttentionModelPredictor


@Predictor.register("distilbert_sequence_classification")
@Predictor.register("distilbert_sequence_classification_from_huggingface")
class DistilBertForSequenceClassificationPredictor(Predictor, AttentionModelPredictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = json_dict['sentence']
        sentiment = json_dict.get('sentiment')
        instance = self._dataset_reader.text_to_instance(tokens=tokens, sentiment=sentiment)
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
    def predictions_to_labeled_instances(self, instance: Instance, outputs: Dict[str, np.ndarray]) -> List[Instance]:
        new_instance = instance.duplicate()
        label = np.argmax(outputs['class_probabilities'])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]
    
    @overrides
    def get_suitable_aggregators(self):
        return [AttentionAverager, AttentionRollout, AttentionFlow]

    @overrides
    def get_attention_based_salience_for_instance(
            self, 
            labeled_instance: Instance, 
            analysis_method: AttentionAnalysisMethods,
            aggregate_method: AttentionAggregator
        ) -> JsonDict:

        output = self._model.forward_on_instance(labeled_instance, output_attentions=[analysis_method])
        attention = torch.from_numpy(output[analysis_method]).unsqueeze(0)
        aggr_attn = aggregate_method.aggregate(attention)

        aggr_attn = aggr_attn.squeeze().numpy()
        return { 'tokens' : aggr_attn }
