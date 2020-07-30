from typing import List, Dict, Iterable 

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter


class AttentionModelPredictor():
    """
    Interface for predictors with models that are to be interpreted through their attention mechanism.
    """

    def get_attention_based_salience_for_instance(self, input_: Instance):
        pass 


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
            original_pred = self.predictor.predict_instance(instance)['prediction']
           
            attn_scores = self.predictor.get_attention_based_salience_for_instance(instance)
            
            instances_with_attn[f'instance_{i_idx+1}'] = {'attn_scores' : []}
            for s_idx, score in enumerate(attn_scores):
                instances_with_attn[f'instance_{i_idx+1}']['attn_scores'].append(score)

        return sanitize(instances_with_attn)
 
