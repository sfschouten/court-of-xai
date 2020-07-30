from typing import List, Dict, Union, Iterator

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

    def saliency_interpret_instances(self, labeled_instances: Iterator[Instance]) -> JsonDict:
        instances_with_lime = dict()
        
        for idx, instance in enumerate(labeled_instances):
            explanation = self._lime(instance)
            instances_with_lime[f'instance_{idx+1}'] = { "lime_scores" : explanation }

        return sanitize(instances_with_lime)


    def _lime(self, instance: Instance) -> None:

        nr_tokens = len(instance['tokens'])
        label = int(instance['label'].label)

        def wrap_fn(input_strings):
            json = [ { "sentence" : input_string } for input_string in input_strings ]
            predictions = self.predictor.predict_batch_json(json)
            cls_probs = [results['class_probabilities'] for results in predictions]
            return numpy.array(cls_probs)
        
        instance_text = " ".join(token.text for token in instance['tokens'])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            explanation = self.explainer.explain_instance(
                    instance_text, 
                    wrap_fn, 
                    labels=(label,), 
                    num_features=nr_tokens, 
                    num_samples=self.num_samples)

        exp_list = explanation.local_exp[label]
        exp_list.sort(key=lambda x: x[0])
        return [abs(x[1]) for x in exp_list]
