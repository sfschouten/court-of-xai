from typing import List, Dict, Iterable 

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter

from ane_research.models.modules.attention.attention import AttentionAnalysisMethods


class AttentionModelPredictor():
    """
    Interface for predictors with models that are to be interpreted through their attention mechanism.
    """

    def get_attention_based_salience_for_instance(self, input_: Instance) -> Dict[str, Iterable[float]]:
        """
        Returns a dictionary with for each TextField in the instance, an iterable with the attention paid
        to the tokens in that field.
        """
        raise NotImplementedError()


class AttentionInterpreter(SaliencyInterpreter):

    def __init__(self, predictor: AttentionModelPredictor, analysis_method: AttentionAnalysisMethods):

        if not isinstance(predictor, AttentionModelPredictor):
            raise TypeError("predictor must be of :class:`~.interpret.saliency_interpreters.AttentionModelPredictor`")

        super().__init__(predictor)

        self.analysis_method = analysis_method


    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict:
    
        instances_with_attn = dict()

        for i_idx, instance in enumerate(labeled_instances):
            attn_scores = self.predictor.get_attention_based_salience_for_instance(instance, analysis_method=self.analysis_method)
            
            instances_with_attn[f'instance_{i_idx+1}'] = {}
            # AllenNLP SaliencyInterpreters index the input sequences in reverse order.
            for f_idx, field in enumerate(reversed(list(attn_scores.keys()))):
                instances_with_attn[f'instance_{i_idx+1}'][f'{self.analysis_method}_{f_idx}'] = list(attn_scores[field])

        return sanitize(instances_with_attn)


@SaliencyInterpreter.register("attention-weights")
class AttentionWeightInterpreter(AttentionInterpreter):
    def __init__(self, predictor: AttentionModelPredictor):
        super().__init__(predictor, AttentionAnalysisMethods.weight_based)


@SaliencyInterpreter.register("attention-weighted-vector-norm")
class AttentionWeightedVectorNormInterpreter(AttentionInterpreter):
    def __init__(self, predictor: AttentionModelPredictor):
        super().__init__(predictor, AttentionAnalysisMethods.norm_based)


@SaliencyInterpreter.register("attention-rollout")
class AttentionRolloutInterpreter(AttentionInterpreter):
    def __init__(self, predictor: AttentionModelPredictor):
        super().__init__(predictor, AttentionAnalysisMethods.rollout)


@SaliencyInterpreter.register("attention-flow")
class AttentionFlowInterpreter(AttentionInterpreter):
    def __init__(self, predictor: AttentionModelPredictor):
        super().__init__(predictor, AttentionAnalysisMethods.flow)


AnalysisMethodToInterpreter = {
    AttentionAnalysisMethods.weight_based: AttentionWeightInterpreter,
    AttentionAnalysisMethods.norm_based: AttentionWeightedVectorNormInterpreter,
    AttentionAnalysisMethods.rollout: AttentionRolloutInterpreter,
    AttentionAnalysisMethods.flow: AttentionFlowInterpreter
}
