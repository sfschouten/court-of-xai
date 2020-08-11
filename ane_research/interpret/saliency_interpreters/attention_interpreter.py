from typing import List, Dict, Iterable 

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter


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


@SaliencyInterpreter.register('attention-interpretator')
class AttentionInterpreter(SaliencyInterpreter):
    """
    Provides a saliency interpretation based on the attention mechanism(s) in the model.

    DISCLAIMER:
    Under what conditions (if any) attention mechanisms can be used to provide 
    saliency (token importance) based interpretation is currently a matter of debate
    in the scientific community.

    """

    def __init__(self, predictor: AttentionModelPredictor):

        if not isinstance(predictor, AttentionModelPredictor):
            raise TypeError("predictor must be of :class:`~.interpret.saliency_interpreters.AttentionModelPredictor`")
        
        super().__init__(predictor)


    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict:
    
        instances_with_attn = dict()

        for i_idx, instance in enumerate(labeled_instances):
            attn_scores = self.predictor.get_attention_based_salience_for_instance(instance)
            
            instances_with_attn[f'instance_{i_idx+1}'] = {}
            # AllenNLP SaliencyInterpreters index the input sequences in reverse order.
            for f_idx, field in enumerate(reversed(list(attn_scores.keys()))):
                instances_with_attn[f'instance_{i_idx+1}'][f'attn_scores_{f_idx}'] = list(attn_scores[field])

        return sanitize(instances_with_attn)
 
