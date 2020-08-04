from typing import List, Dict, Union, Iterable 

import numpy
import itertools

from allennlp.nn import util
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, Batch
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter

@SaliencyInterpreter.register('leave-one-out')
class LeaveOneOut(SaliencyInterpreter):
    """
    """

    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict:
        batch = list(labeled_instances)
        instances_with_loo = dict()
        
        batch_outputs = self.predictor.predict_batch_instance(batch)
        original_preds = [instance['prediction'] for instance in batch_outputs]
        nr_preds_per_instance = len(original_preds[0])

        loo_preds = self._leave_one_out(batch)
        for i_idx, (_, instance_preds) in enumerate(loo_preds.items()):
            instance_preds = instance_preds['loo_scores']
            instance_originals = original_preds[i_idx]

            for i in range(len(instance_preds)):
                pairs = zip(instance_originals, instance_preds[i])
                instance_preds[i] = sum( abs(b - a) for a,b in pairs ) / len(instance_preds[i])

        return sanitize(loo_preds)


    def _register_pre_forward_hook(self, i: int):
        vocab = self.predictor._model.vocab
        oov_token_index = vocab.get_token_index(vocab._oov_token)

        def pre_forward_hook(module, inputs):
            # remove the i-th input token
            indices, = inputs
            batch_size, padding_length = indices.shape
            indices[:, i] = oov_token_index 
        
        # Register the hook
        embed_layer = util.find_embedding_layer(self.predictor._model)
        handle = embed_layer.register_forward_pre_hook(pre_forward_hook)
        return handle


    def _leave_one_out(self, instances: List[Instance]) -> Dict[str, numpy.ndarray]:
        """
        Returns the leave-one-out based saliences for the given Instances.
        """
        batch = Batch(instances)

        instance_lengths = [ next(iter(instance.get_padding_lengths()['tokens'].values())) for instance in batch.instances ]
        
        padding_lengths = batch.get_padding_lengths()
        max_sequence_length = max(len_ for _, dict_ in padding_lengths.items() for _, len_ in dict_.items())

        predictions: Dict[str, Dict[str, Any]] = {}
        for idx in range(len(batch.instances)):
            predictions[f'instance_{idx+1}'] = {'loo_scores' : []}

        for idx in range(max_sequence_length):
            handle = self._register_pre_forward_hook(idx)

            outputs = self.predictor.predict_batch_instance(batch.instances)
            for idx2, output in enumerate(outputs):
                length = instance_lengths[idx2]
                if idx >= length:
                    continue 

                predictions[f'instance_{idx2+1}']['loo_scores'].append(output['prediction'])

            handle.remove()

        return predictions 


