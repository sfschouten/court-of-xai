from typing import List, Dict, Iterable 

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter

from ane_research.models.modules.attention.attention import AttentionAnalysisMethods, AttentionAggregator


class AttentionModelPredictor():
    """
    Interface for predictors with models that are to be interpreted through their attention mechanism.
    """

    def get_attention_based_salience_for_instance(
            labeled_instance: Instance, 
            analysis_method: AttentionAnalysisMethods,
            aggregate_method: AttentionAnalysisMethods
        ) -> Dict[str, Iterable[float]]:
        """
        Returns a dictionary with for each TextField in the instance, an iterable with the attention paid
        to the tokens in that field.
        """
        raise NotImplementedError()


class AttentionInterpreter(SaliencyInterpreter):

    def __init__(
            self, 
            predictor: AttentionModelPredictor, 
            analysis_method: AttentionAnalysisMethods,
            aggregate_method: AttentionAggregator
    ):
        if not isinstance(predictor, AttentionModelPredictor):
            raise TypeError("predictor must be of :class:`~.interpret.saliency_interpreters.AttentionModelPredictor`")

        super().__init__(predictor)
        self.analysis_method = analysis_method
        self.aggregate_method = aggregate_method

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


@SaliencyInterpreter.register("attention-avg-weights")
class AttentionWeightInterpreter(AttentionInterpreter):
    def __init__(self, predictor: AttentionModelPredictor):
        super().__init__(
            predictor, 
            AttentionAnalysisMethods.weight_based,
            AttentionAverager()
        )

@SaliencyInterpreter.register("attention-avg-weighted-vector-norm")
class AttentionWeightedVectorNormInterpreter(AttentionInterpreter):
    def __init__(self, predictor: AttentionModelPredictor):
        super().__init__(
            predictor, 
            AttentionAnalysisMethods.norm_based,
            AttentionAverager()
        )

@SaliencyInterpreter.register("attention-rollout-weights")
class AttentionRolloutWeightInterpreter(AttentionInterpreter):
    def __init__(self, predictor: AttentionModelPredictor):
        super().__init__(
            predictor, 
            AttentionAnalysisMethods.weight_based,
            AttentionRollout()
        )

@SaliencyInterpreter.register("attention-rollout-weighted-vector-norm")
class AttentionRolloutWVNInterpreter(AttentionInterpreter):
    def __init__(self, predictor: AttentionModelPredictor):
        super().__init__(
            predictor, 
            AttentionAnalysisMethods.norm_based,
            AttentionRollout()
            )

#@SaliencyInterpreter.register("attention-flow-weights")
#class AttentionFlowWeightInterpreter(AttentionInterpreter):
#    def __init__(self, predictor: AttentionModelPredictor):
#        super().__init__(predictor, AttentionAnalysisMethods.flow)

#@SaliencyInterpreter.register("attention-flow-weighted-vector-norm")
#class AttentionFlowWVNInterpreter(AttentionInterpreter):
#    def __init__(self, predictor: AttentionModelPredictor):
#        super().__init__(predictor, AttentionAnalysisMethods.flow)

#TODO fix this
AnalysisMethodToInterpreter = {
    AttentionAnalysisMethods.weight_based: AttentionWeightInterpreter,
    AttentionAnalysisMethods.norm_based: AttentionWeightedVectorNormInterpreter,
#    AttentionAnalysisMethods.rollout: AttentionRolloutInterpreter,
#    AttentionAnalysisMethods.flow: AttentionFlowInterpreter
}
