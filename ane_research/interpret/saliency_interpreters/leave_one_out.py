from typing import List, Dict, Union, Iterator

import numpy

from allennlp.nn import util
from allennlp.common.util import JsonDict, sanitize
from allennlp.common import Tqdm
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter

@SaliencyInterpreter.register('leave-one-out')
class LeaveOneOut(SaliencyInterpreter):
    """
    """

    def saliency_interpret_instances(self, labeled_instances: Iterator[Instance]) -> JsonDict:
        instances_with_loo = dict()
        for idx, instance in Tqdm.tqdm(enumerate(labeled_instances), desc="interpreting instances"):
            original_pred = self.predictor.predict_instance(instance)['prediction']
            loo_preds = self._leave_one_out(instance)
            loo_scores = { key : [ abs(original_pred[i] - loo_pred[i]) for i in range(len(loo_pred)) ] for key, loo_pred in loo_preds.items() } 
            instances_with_loo[f'instance_{idx+1}'] = loo_scores
        return sanitize(instances_with_loo)


    def _register_pre_forward_hook(self, i: int):
        
        def pre_forward_hook(module, inputs):
            # remove the i-th input token

            # TODO figure out what inputs looks like here 
            pass
        # Register the hook
        handle = self.predictor._model.register_forward_pre_hook(pre_forward_hook)
        return handle


    def _leave_one_out(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        """
        Returns the leave-one-out based saliences for the given :class:`~allennlp.data.instance.Instance`.
        """

        sequence_length = instance['tokens'].sequence_length() 

        predictions: Dict[str, Any] = {}
        for idx in range(sequence_length):
            handle = self._register_pre_forward_hook(idx)

            outputs = self.predictor.predict_instance(instance)
            predictions[f'loo_pred_{idx+1}'] = outputs['prediction']

            handle.remove()

        return predictions 


