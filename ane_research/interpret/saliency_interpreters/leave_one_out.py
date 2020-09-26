from typing import Any, List, Dict, Union, Iterable
import itertools
import logging
import torch
from collections import defaultdict
from allennlp.nn import util
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, Batch
from allennlp.predictors import Predictor
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter

from ane_research.config import Config
from ane_research.models.distilbert import DistilBertForSequenceClassification


@SaliencyInterpreter.register('leave-one-out')
class LeaveOneOut(SaliencyInterpreter):

    def __init__(self, predictor: Predictor):
        super().__init__(predictor)
        self._id = 'leave-one-out'
        self.logger = logging.getLogger(Config.logger_name)

    @property
    def id(self):
        return self._id

    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict:

        batch = list(labeled_instances)
        instances_with_loo = dict()
        batch_outputs = self.predictor.predict_batch_instance(batch)
        original_preds = [instance['prediction'] for instance in batch_outputs]
        nr_preds_per_instance = len(original_preds[0])
        loo_preds = self._leave_one_out(batch)
        fields = self.predictor._model.get_field_names()

        for instance_originals, predictions_per_field in zip(original_preds, loo_preds.values()):
            for f_idx in range(len(fields)):
                prediction_for_field = predictions_per_field[f'loo_scores_{f_idx}']
                for i in range(len(prediction_for_field)):
                    pairs = zip(instance_originals, prediction_for_field[i])
                    prediction_for_field[i] = sum( abs(b - a) for a,b in pairs ) / len(prediction_for_field[i])
        return sanitize(loo_preds)


    def _register_pre_forward_hook(self, token_idx: int, sequence_idx: int) -> torch.utils.hooks.RemovableHandle:
        """Replace the ith token in the jth input sequence to a model's embeddings with an OOV token

        Args:
            token_idx (int):
                The index of the token to be replaced
            sequence_idx (int):
                Some models process more than one sequence per instance, e.g. in pair classification, so you need
                to specify the index of the sequence in which the token will be replaced

        Returns:
            torch.utils.hooks.RemovableHandle: Handle to remove the hook
        """
        vocab = self.predictor._model.vocab
        oov_token_index = vocab.get_token_index(vocab._oov_token)

        global current_sequence_idx
        current_sequence_idx = 0

        def pre_forward_hook(module, inputs):
            global current_sequence_idx
            if current_sequence_idx == sequence_idx:
                # remove the i-th input token
                indices, = inputs
                batch_size, padding_length = indices.shape
                indices[:, token_idx] = oov_token_index
            current_sequence_idx += 1

        # Register the hook
        # TODO: fix this hack
        if isinstance(self.predictor._model, DistilBertForSequenceClassification):
            embed_layer = self.predictor._model.embeddings
        else:
            embed_layer = util.find_embedding_layer(self.predictor._model)

        handle = embed_layer.register_forward_pre_hook(pre_forward_hook)
        return handle


    def _leave_one_out(self, instances: List[Instance]) -> Dict[str, Dict[str, List[List[Any]]]]:
        """Returns the leave-one-out based saliences for the given Instances

        Args:
            instances (List[Instance]): A collection of instances that will be collected in a Batch

        Returns:
            Dict[str, Dict[str, List[List[Any]]]]:
                A mapping of the instance index in the batch to a list of prediction scores, of the signature
                `{'instance_index': {'loo_scores': [...}` We leave exactly one token out at a time, regardless
                of how many sequences a model processes per instance. Thus, the length of the output list per
                instance will equal the total length of all of the sequences it contains.
        """
        batch = Batch(instances)
        fields = self.predictor._model.get_field_names()
        predictions = {f'instance_{i}': {f'loo_scores_{f}': [] for f in range(len(fields))} for i in range(1, len(instances) + 1)}

        for f_idx, field in enumerate(fields):
            instance_lengths = [next(iter(instance.get_padding_lengths()[field].values()))for instance in batch.instances]
            max_sequence_length = max(instance_lengths)

            for idx in range(max_sequence_length):
                handle = self._register_pre_forward_hook(idx, f_idx)

                outputs = self.predictor.predict_batch_instance(batch.instances)
                for idx2, output in enumerate(outputs):
                    length = instance_lengths[idx2]
                    if idx >= length:
                        continue

                    predictions[f'instance_{idx2+1}'][f'loo_scores_{f_idx}'].append(output['prediction'])

                handle.remove()

        return predictions
