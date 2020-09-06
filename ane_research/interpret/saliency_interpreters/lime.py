from typing import List, Dict, Union, Iterable

from lime import lime_text
from lime.lime_text import LimeTextExplainer

import numpy
import itertools
import logging
import warnings

from ane_research.config import Config

from allennlp.predictors import Predictor
from allennlp.nn import util
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, Batch
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter

@SaliencyInterpreter.register('lime')
class LimeInterpreter(SaliencyInterpreter):
    """
    """

    def __init__(self, predictor: Predictor, num_samples: int = 250) -> None:
        super().__init__(predictor)

        self._id = 'lime'
        self.explainer = LimeTextExplainer(bow=False, split_expression=r'\s+')
        self.num_samples = num_samples

    @property
    def id(self):
        return self._id

    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict:
        instances_with_lime = dict()

        for idx, instance in enumerate(labeled_instances):
            explanation = self._lime(instance)
            instances_with_lime[f'instance_{idx+1}'] = { "lime_scores" : explanation }

        return sanitize(instances_with_lime)


    def _lime(self, instance: Instance) -> List[float]:
        """
        Calculates LIME attribution for the given Instance.

        LIME does not natively support an input consisting of multiple sequences.
        So if instance has multiple sequences we concatentate them, using a separator token
        to allow splitting the concatentation back into its constituent parts.
        """
        SEPARATOR = "[LIME_SEP]"

        fields = self.predictor._model.get_field_names()
        nr_tokens = sum( len(instance[field]) for field in fields )
        label = int(instance['label'].label)
        seq_ends: List[int] = None

        def split_by_fields(tokens):
            """ splits list of tokens including (masked) seperator using seq_ends """
            # indices of first token of each sequence
            l = [0] + [int(l) for l in seq_ends]

            return [tokens[l[i]:l[i+1]-1] for i in range(len(seq_ends))]

        def wrap_fn(input_strings):
            nonlocal seq_ends

            # the first string in the input_string is the original input (none of the tokens replaced with UNK)
            # we store the lengths of the constituent sequences (including SEPERATOR)
            seq_lengths = [len(string.strip().split()) + 1 for string in input_strings[0].split(SEPARATOR)]

            # after splitting by the separator we expect one string per field.
            assert len(seq_lengths) == len(fields)

            # accumulate lengths to get the index of the token after each sequence (including SEPERATOR)
            seq_ends = list(itertools.accumulate(seq_lengths))  # NON-LOCAL

            if len(fields) > 1:
                # one json field per sequence
                json = [ { f"sentence{i+1}" : " ".join(part) for i,part in enumerate( split_by_fields(string.split()) ) } for string in input_strings ]
            else:
                json = [ { f"sentence" : string } for string in input_strings ]

            # batched prediction
            BATCH_SIZE = 256
            batches = (itertools.islice(json, x, x+BATCH_SIZE) for x in range(0, len(json), BATCH_SIZE))

            predictions = []
            for idx, batch in enumerate(batches):
                batch = list(batch)
                predictions.extend(self.predictor.predict_batch_json(batch))
            cls_probs = [results['class_probabilities'] for results in predictions]

            return numpy.array(cls_probs)

        joined_sequences = [ " ".join(token.text for token in instance[field]) for field in fields]
        instance_text = f' {SEPARATOR} '.join(joined_sequences)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            explanation = self.explainer.explain_instance(
                    instance_text,
                    wrap_fn,
                    labels=(label,),
                    num_features=nr_tokens + len(fields) - 1, # account for the separator token
                    num_samples=self.num_samples)

        exp_list = explanation.local_exp[label]
        exp_list.sort(key=lambda x: x[0]) # sort by index

        # Remove attribution to seperator, and concatenate constituent sequences in reverse order
        # for compatibility with AllenNLP interpreters.
        exp_list = list(itertools.chain.from_iterable(reversed(split_by_fields(exp_list))))
        exp_list = [abs(x[1]) for x in exp_list]
        return exp_list
