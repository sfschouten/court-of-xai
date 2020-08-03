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

    def attribute_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict:
        raise NotImplementedError()


# Captum expects the input to the model you are interpreting to be one or more
# Tensor objects, but AllenNLP Model classes often take Dict[...] objects as
# their input. To fix this we require Models that are to be used with Captum
# to implement a method that takes individual tensors and returns them packaged
# in whatever structures necessary for the Model.forward.
# If the Model does not implement this class, it will be used as if it takes
# one or more Tensors as input.
class CaptumCompatible(torch.nn.Module):
    
    @override
    def convert_inputs(*inputs):
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
        super().__init__(predictor._model)

        self.predictor = predictor

    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict: 
        instances_with_captum_attr = dict()
        
        model = self.predictor._model
        batch_size = len(labeled_instances)
        with torch.no_grad():
            cuda_device = model._get_prediction_device()
            batch = Batch(labeled_instances)
            batch.index_instances(model.vocab)
            model_input = util.move_to_device(batch.as_tensor_dict(), cuda_device)

            print(model_input)

            attributions = self.attribute(inputs=model_input)
            
        print(attributions.shape)
        token_attr = attributions.sum(dim=0)
        
        for i in range(batch_size):
            explanation = token_attr[i].to_list()
            instances_with_captum_attr[f'instance_{idx+1}'] = { "captum_attr_scores" : explanation }

        return sanitize(instances_with_captum_attr)


# ---DeepLiftShap---
from captum.attr import DeepLiftShap
@CaptumAttribution.register('captum-deepliftshap')
class CaptumDeepLiftShap(CaptumAttribution, DeepLiftShap):

    def __init__(self, predictor: Predictor):
        super().__init__(predictor._model)

        self.predictor = predictor

    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict: 
        instances_with_captum_attr = dict()
        
        model = self.predictor._model
        batch_size = len(labeled_instances)
        with torch.no_grad():
            cuda_device = model._get_prediction_device()
            batch = Batch(labeled_instances)
            batch.index_instances(model.vocab)
            model_input = util.move_to_device(batch.as_tensor_dict(), cuda_device)

            print(model_input)
            # Right now it looks like the DeepLiftShap will not work, since it 
            # expects the nn.Module's forward to take Tensors, but our Model takes
            # tokens: Dict[str, Tensor] ...

            attributions = self.attribute(inputs=model_input)
            
        print(attributions.shape)
        token_attr = attributions.sum(dim=0)
        
        for i in range(batch_size):
            explanation = token_attr[i].to_list()
            instances_with_captum_attr[f'instance_{idx+1}'] = { "captum_attr_scores" : explanation }

        return sanitize(instances_with_captum_attr)




