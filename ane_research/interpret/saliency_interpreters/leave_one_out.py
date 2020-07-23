
from typing import List, Dict

import numpy

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allenlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter


@SaliencyInterpreter.register('leave-one-out')
class LeaveOneOut(SaliencyInterpreter):
    """
    """

    def saliency_interpret(self, inputs: ???) -> JsonDict:


    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        # Convert inputs to labeled instances
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)
    
        instances_with_loo = dict()
        for idx, instance in enumerate(labeled_instances):
            original_pred = self.predictor.predict_instance(instance)['prediction']
            loo_preds = self._leave_one_out(instance)
            
            loo_scores = { key : math.abs(original_pred - loo_pred) for key, loo_pred in loo_preds } 
            instances_with_loo[f'instance_{idx+1}'] = loo_scores

        return sanitize(instances_with_loo)


    def _register_pre_forward_hook(self, i: int):
        
        def pre_forward_hook(module, inputs):
            # remove the i-th input token

            # TODO figure out what inputs looks like here 

        # Register the hook
        handle = self.predictor._model.register_forward_pre_hook(pre_forward_hook)
        return handle


    def _leave_one_out(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        """
        Returns the leave-one-out based saliences for the given :class:`~allennlp.data.instance.Instance`.
        """

        tokens_mask = util.get_text_field_mask(instance['tokens']) 
        _, max_sequence_length = tokens_mask.shape

        predictions: Dict[str, Any] = {}
        for idx in range(max_sequence_length):
            handle = self._register_pre_forward_hook(idx)

            outputs = self.predictor.predict_instance(instance)
            predictions[f'loo_pred_{idx+1}'] = outputs['prediction']

            handle.remove()

        return predictions 


