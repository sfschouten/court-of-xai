from typing import List, Dict, Iterable, Optional, Type, Union

import logging

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter

from ane_research.models.modules.attention.attention import AttentionAnalysisMethods, AttentionAggregator


class AttentionModelPredictor():
    """
    Interface for predictors with models that are to be interpreted through their attention mechanism.
    """

    def get_attention_based_salience_for_instance(
        self,
        labeled_instance: Instance, 
        analysis_method: AttentionAnalysisMethods,
        aggregate_method: AttentionAggregator
    ) -> Dict[str, Iterable[float]]:
        """
        Returns a dictionary with for each TextField in the instance, an iterable with the attention paid
        to the tokens in that field.
        """
        raise NotImplementedError()

    def get_suitable_aggregators(self) -> Iterable[Type[Union[None, AttentionAggregator]]]:
        """
        Returns one or more suitable aggregator types, if no aggregation is necessary the iterable 
        should include NoneType.
        """
        raise NotImplementedError()

@SaliencyInterpreter.register("attention-interpreter")
class AttentionInterpreter(SaliencyInterpreter):

    def __init__(
            self, 
            predictor: AttentionModelPredictor, 
            analysis_method: AttentionAnalysisMethods,
            aggregate_method: Optional[AttentionAggregator] = None
    ):
        """
        """
        if not isinstance(predictor, AttentionModelPredictor):
            raise TypeError("predictor must be of :class:`~.interpret.saliency_interpreters.AttentionModelPredictor`")

        super().__init__(predictor)
        self.analysis_method = analysis_method

        # Make sure aggregate_method is suitable for predictor
        if not any(isinstance(aggregate_method, suitable) for suitable in  predictor.get_suitable_aggregators()):
            logger = logging.getLogger(__name__)
            logger.warning("The supplied aggregator is not suitable for this predictor!")
        
        self.aggregate_method = aggregate_method

        agg_method = f"{self.aggregate_method.id}_" if self.aggregate_method else ""
        self._id = f"attn_{agg_method}{self.analysis_method.value}"

    @property
    def id(self):
       return self._id

    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict:
        instances_with_attn = dict()

        for i_idx, instance in enumerate(labeled_instances):
            attn_scores = self.predictor.get_attention_based_salience_for_instance(
                    instance, 
                    analysis_method=self.analysis_method,
                    aggregate_method=self.aggregate_method
            )
            
            instances_with_attn[f'instance_{i_idx+1}'] = {}
            # AllenNLP SaliencyInterpreters index the input sequences in reverse order.
            for f_idx, field in enumerate(reversed(list(attn_scores.keys()))):
                instances_with_attn[f'instance_{i_idx+1}'][f'{self.analysis_method}_{f_idx}'] = list(attn_scores[field])

        return sanitize(instances_with_attn)



