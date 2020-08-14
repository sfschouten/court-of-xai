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

    def __init__(self, predictor: Predictor):
        self.predictor = predictor

        if not isinstance(self.predictor._model, CaptumCompatible):
            raise TypeError("Predictor._model must be CaptumCompatible.")

    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance]) -> JsonDict:
        instances_with_captum_attr = dict()

        model = self.predictor._model

        captum_inputs = model.instances_to_captum_inputs(labeled_instances)
        kwargs = self.attribute_kwargs(captum_inputs)

        with warnings.catch_warnings(), torch.backends.cudnn.flags(enabled=False):
            warnings.simplefilter("ignore", category=UserWarning)
            tensors = self.attribute(**kwargs)

        field_names = model.get_field_names()

        # sum out the embedding dimensions to get token importance
        attributions = {}
        for field, tensor in zip(field_names, tensors):
            attributions[field] = tensor.sum(dim=-1).abs()
            batch_size, _, _ = tensor.shape


        for idx, instance in zip(range(batch_size), labeled_instances):
            instances_with_captum_attr[f'instance_{idx+1}'] = { }
            # AllenNLP SaliencyInterpreters index the input sequences in reverse order.
            for field_idx, (field, token_attr) in enumerate(reversed(list(attributions.items()))):
                sequence_length = len(instance[field])
                explanation = token_attr[idx].tolist()[:sequence_length]

                # this should be the same order as returned by get_field_names
                instances_with_captum_attr[f'instance_{idx+1}'][f"{self.id()}_scores_{field_idx}"] = explanation

        return sanitize(instances_with_captum_attr)

    def id(self):
        """
        Returns a short identifier for the type of attribution calculated in this class.
        """
        raise NotImplementedError()

    def attribute_kwargs(self, captum_inputs):
        """
        Args:
          captum_inputs: result of CaptumCompatible.instances_to_captum_inputs.
        Returns:
          key-word arguments to be given to the attribute method of the
          relevant Captum Attribution sub-class.
        """
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
          Tuple with (inputs, target, additional_forward_args)
          Both inputs and target tensors should have the Batch dimension first.
          The inputs Tensors should have the Embedding dimension last.
        """
        raise NotImplementedError()

    def get_field_names(self) -> Iterable[str]:
        """
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

        CaptumAttribution.__init__(self, predictor)

        self.submodel = self.predictor._model.captum_sub_model()
        DeepLiftShap.__init__(self, self.submodel)

    def id(self):
        return 'dlshap'

    def attribute_kwargs(self, captum_inputs):
        inputs, target, additional = captum_inputs
        baselines = tuple(torch.zeros_like(tensor) for tensor in inputs)
        return {'inputs' : inputs,
                'target': target,
                'baselines' : baselines,
                'additional_forward_args' : additional}


# ---GradientShap---
from captum.attr import GradientShap
@CaptumAttribution.register('captum-gradientshap')
class CaptumGradientShap(CaptumAttribution, GradientShap):

    def __init__(self, predictor: Predictor):

        CaptumAttribution.__init__(self, predictor)

        self.submodel = self.predictor._model.captum_sub_model()
        GradientShap.__init__(self, self.submodel)

    def id(self):
        return 'gradshap'

    def attribute_kwargs(self, captum_inputs):
        inputs, target, additional = captum_inputs
        baselines = tuple(torch.zeros_like(tensor) for tensor in inputs)
        return {'inputs' : inputs,
                'baselines' : baselines,
                'target': target,
                'additional_forward_args' : additional}


