from typing import List, Dict, Union, Iterable

from ane_research.interpret.saliency_interpreters.modules.lime_allennlp_instance import LimeAllenNLPInstanceExplainer

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

    def __init__(self, predictor: Predictor, num_samples: int = 250, batch_size: int = 16) -> None:
        super().__init__(predictor)

        self._id = 'lime'
        self.explainer = LimeAllenNLPInstanceExplainer(bow=False, batch_size=batch_size)
        self.num_samples = num_samples
        self.logger = logging.getLogger(Config.logger_name)

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
        fields = self.predictor._model.get_field_names()
        nr_tokens = sum( len(instance[field]) for field in fields )
        label = int(instance['label'].label)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            explanation = self.explainer.explain_instance(
                    instance,
                    self.predictor._model,
                    labels=(label,),
                    num_features=nr_tokens + len(fields) - 1, # account for the separator token
                    num_samples=self.num_samples)

        exp_list = explanation.local_exp[label]
        exp_list.sort(key=lambda x: x[0]) # sort by index

        exp_list = [abs(x[1]) for x in exp_list]
        return exp_list


