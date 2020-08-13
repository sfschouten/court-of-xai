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

        self.explainer = LimeTextExplainer(bow=False, split_expression=r'\s+')
        self.num_samples = num_samples

    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict:
        instances_with_lime = dict()

        for idx, instance in enumerate(labeled_instances):
            explanation = self._lime(instance)
            instances_with_lime[f'instance_{idx+1}'] = { "lime_scores" : explanation }

        return sanitize(instances_with_lime)


    def _lime(self, instance: Instance) -> Dict[str, List[float]]:
        SEPARATOR = "[LIME_SEP]"

        fields = self.predictor._model.get_field_names()
        nr_tokens = sum( len(instance[field]) for field in fields )
        label = int(instance['label'].label)
        acc = -1

        def split_by_fields(tokens):
            l = [-1] + [int(l) for l in acc]
            l[-1] += 1
            return [tokens[l[i]+1:l[i+1]] for i in range(len(acc))]

        def wrap_fn(input_strings):
            nonlocal acc

            split_original = [len(string.strip().split()) for string in input_strings[0].split(SEPARATOR)]
            assert len(split_original) == len(fields)

            acc = list(itertools.accumulate(split_original))

            if len(fields) > 1:
                json = [ { f"sentence{i+1}" : " ".join(part) for i,part in enumerate( split_by_fields(string.split()) ) } for string in input_strings ]
            else:
                json = [ { f"sentence" : string } for string in input_strings ]
            
            predictions = self.predictor.predict_batch_json(json)
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
                    num_features=nr_tokens+1, 
                    num_samples=self.num_samples)

        exp_list = explanation.local_exp[label]
        exp_list.sort(key=lambda x: x[0])
        #print(exp_list)
        #print(len(exp_list))
        exp_list = list(itertools.chain.from_iterable(reversed(split_by_fields(exp_list))))
        #print(exp_list)
        #print(len(exp_list))
        exp_list = [abs(x[1]) for x in exp_list]
        #print(len(exp_list), nr_tokens)
        assert len(exp_list) == nr_tokens
        return exp_list
