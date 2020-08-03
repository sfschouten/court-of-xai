from typing import List, Dict, Union, Iterable, cast 

from lime import lime_text
from lime.lime_text import LimeTextExplainer

import torch
import numpy
import itertools
import logging
import warnings

from ane_research.config import Config

from allennlp.predictors import Predictor
from allennlp.nn import util
from allennlp.common import Registrable
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, Batch
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter

# Registrable for captum registrations.
class CaptumAttribution(Registrable):

    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict:
        raise NotImplementedError()


# Captum expects the input to the model you are interpreting to be one or more
# Tensor objects, but AllenNLP Model classes often take Dict[...] objects as
# their input. To fix this we require the Models that are to be used with Captum
# to implement a set of methods that make it possible to use Captum. 
class CaptumCompatible():
    
    def captum_sub_model(self):
        """
        Returns a PyTorch nn.Module instance with a forward that performs
        the same steps as the Model would normally, but starting from word embeddings.
        As such it accepts FloatTensors as input, which is required for Captum.
        """
        raise NotImplementedError()

    def instances_to_captum_inputs(self, *inputs):
        """
        Converts a set of Instances to a Tensor suitable to pass to the submodule
        obtained through captum_sub_model.

        Returns
          Tuple with (inputs, additional_forward_args)

        """
        raise NotImplementedError()


@SaliencyInterpreter.register('captum')
class CaptumInterpreter(SaliencyInterpreter):

    def __init__(self, predictor: Predictor, captum: CaptumAttribution) -> None:
        super().__init__(predictor)

        self.captum = captum

    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict:
        return self.captum.saliency_interpret_instances(labeled_instances)        


# Below are wrapper classes for various Captum Attribution sub-classes. 

# ---DeepLiftShap---
from captum.attr import DeepLiftShap
@CaptumAttribution.register('captum-deepliftshap')
class CaptumDeepLiftShap(CaptumAttribution, DeepLiftShap):

    def __init__(self, predictor: Predictor):

        self.predictor = predictor

        if not isinstance(self.predictor._model, CaptumCompatible):
            raise TypeError("Predictor._model must be CaptumCompatible.")

        self.submodel = self.predictor._model.captum_sub_model()
        super().__init__(self.submodel)


    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict: 
        instances_with_captum_attr = dict()
        
        model = self.predictor._model
        inputs, additional = model.instances_to_captum_inputs(labeled_instances)

        baselines = tuple(torch.zeros_like(tensor) for tensor in inputs)
       
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            attributions, = self.attribute(
                    inputs=inputs, 
                    baselines=baselines, 
                    additional_forward_args=additional)

        batch_size, _, _ = attributions.shape

        # sum out the embedding dimensions to get token importance
        token_attr = attributions.sum(dim=-1).abs() 

        for idx, instance in zip(range(batch_size), labeled_instances):
            sequence_length = len(instance['tokens'])
            explanation = token_attr[idx].tolist()[:sequence_length]
            instances_with_captum_attr[f'instance_{idx+1}'] = { "captum_attr_scores" : explanation }

        return sanitize(instances_with_captum_attr)



