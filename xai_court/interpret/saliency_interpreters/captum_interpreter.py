from typing import Any, List, Dict, Tuple, Union, Iterable, cast

import torch
import logging
import warnings

from xai_court.config import Config
from xai_court.interpret.saliency_interpreters.baseline_generators.baseline import Baseline, BaselineCompatibleInterpreter

from allennlp.predictors import Predictor
from allennlp.nn import util
from allennlp.common import Registrable
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, Batch
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter


# pylint: disable=E1101


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


from xai_court.models.distilbert import DistilBertForSequenceClassification

# Registrable for captum registrations.
class CaptumAttribution(Registrable):

    def __init__(
        self,
        identifier: str,
        predictor: Predictor,
        mask_features_by_token: bool = False,
        attribute_args: Dict[str, Any] = None,
    ):
        """
        Generic constructor for Captum Attribution Classes

        Args:
            identifier (str): unique string identifier for the attribution method (e.g., 'captum-lime')
            predictor (Predictor): Predictor for the Model to be interpreted
            mask_features_by_token (bool, optional): Used in Captum methods that require a feature mask.
                                                     If True, treat each token as one feature. If False,
                                                     treat each scalar in the embedding dimension as one
                                                     feature. Defaults to False.
            attribute_args (Dict[str, Any], optional): Any additional parameters passed to the `attribute`
                                                       method (e.g., `n_samples` in LIME). Defaults to None.
        """    
        self._id = identifier
        self.predictor = predictor
        self.mask_features_by_token = mask_features_by_token
        self.attribute_args = attribute_args
        if self.attribute_args is None:
            self.attribute_args = {}
        self.logger = logging.getLogger(Config.logger_name)

        if not isinstance(self.predictor._model, CaptumCompatible):
            raise TypeError("Predictor._model must be CaptumCompatible.")

    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance], **kwargs) -> JsonDict:

        instances_with_captum_attr = dict()

        model = self.predictor._model

        captum_inputs = model.instances_to_captum_inputs(labeled_instances)

        all_args = dict(
            **kwargs,
            **self.attribute_kwargs(captum_inputs, self.mask_features_by_token),
            **self.attribute_args
        )

        # TODO: remove disabling of CUDNN when https://github.com/pytorch/pytorch/issues/10006 is fixed
        with warnings.catch_warnings(), torch.backends.cudnn.flags(enabled=False):
            warnings.simplefilter("ignore", category=UserWarning)
            tensors = self.attribute(**all_args)
        field_names = model.get_field_names()

        attributions = {}
        for field, tensor in zip(field_names, tensors):
            # Attributions were calculated per token, so each scalar in the embedding dimension will be the same
            if self.mask_features_by_token:
                attributions[field] = tensor.mean(dim=-1).abs()
            # Otherwise, sum out the embedding dimensions to get token importance
            else:
                attributions[field] = tensor.sum(dim=-1).abs()

            batch_size, _, _ = tensor.shape


        for idx, instance in zip(range(batch_size), labeled_instances):
            instances_with_captum_attr[f'instance_{idx+1}'] = { }
            # AllenNLP SaliencyInterpreters index the input sequences in reverse order.
            for field_idx, (field, token_attr) in enumerate(reversed(list(attributions.items()))):
                sequence_length = len(instance[field])
                explanation = token_attr[idx].tolist()[:sequence_length]

                # this should be the same order as returned by get_field_names
                instances_with_captum_attr[f'instance_{idx+1}'][f"{self.id}_scores_{field_idx}"] = explanation

        return sanitize(instances_with_captum_attr)

    @property
    def id(self):
        return self._id

    def attribute_kwargs(self, captum_inputs: Tuple, mask_features_by_token: bool = False) -> Dict:
        """
        Args:
            captum_inputs (Tuple): result of CaptumCompatible.instances_to_captum_inputs.
            mask_features_by_token (bool, optional): For Captum methods that require a feature mask,
                                                     define each token as a feature if True. If False,
                                                     define each scalar in the embedding dimension as a
                                                     feature (e.g., default behavior in LIME).
                                                     Defaults to False.

        Returns:
            Dict: key-word arguments to be given to the attribute method of the
                  relevant Captum Attribution sub-class.
        """
        inputs, target, additional = captum_inputs
        vocab = self.predictor._model.vocab

        # Manually check for distilbert.
        if isinstance(self.predictor._model, DistilBertForSequenceClassification):
            embedding = self.predictor._model.embeddings 
        else:
            embedding = util.find_embedding_layer(self.predictor._model)
    

        baselines =

        attr_kwargs = {
            'inputs' : inputs,
            'target': target,
            'baselines' : baselines,
            'additional_forward_args' : additional
        }

        # For methods that require a feature mask, define each token as one feature
        if mask_features_by_token:
            # see: https://captum.ai/api/lime.html for the definition of a feature mask
            feature_mask_tuple = tuple()
            for i in range(len(inputs)):
                input_tensor = inputs[i]
                bs, seq_len, emb_dim = input_tensor.shape
                feature_mask = torch.tensor(list(range(bs * seq_len))).reshape([bs, seq_len, 1])
                feature_mask = feature_mask.to(inputs[0].device)
                feature_mask = feature_mask.expand(-1, -1, emb_dim)
                feature_mask_tuple += (feature_mask,) # (bs, seq_len, emb_dim)
            attr_kwargs['feature_mask'] = feature_mask_tuple

        return attr_kwargs


@SaliencyInterpreter.register('captum')
class CaptumInterpreter(SaliencyInterpreter, BaselineCompatibleInterpreter):

    def __init__(self, predictor: Predictor, captum: CaptumAttribution, baseline: Baseline) -> None:
        super().__init__(predictor)
        BaselineCompatibleInterpreter.__init__(self, baseline)

        self.captum = captum
        self._id = self.captum.id

    @property
    def id(self):
        return self._id

    def saliency_interpret_instances(self, labeled_instances: Iterable[Instance], **kwargs) -> JsonDict:
        return self.captum.saliency_interpret_instances(labeled_instances, **kwargs)


# Below are wrapper classes for various Captum Attribution sub-classes.

# ---DeepLiftShap---
from captum.attr import DeepLiftShap
@CaptumAttribution.register('captum-deepliftshap')
class CaptumDeepLiftShap(CaptumAttribution, DeepLiftShap):

    def __init__(self, predictor: Predictor):

        CaptumAttribution.__init__(self, 'dlshap', predictor)

        self.submodel = self.predictor._model.captum_sub_model()
        DeepLiftShap.__init__(self, self.submodel)


# ---GradientShap---
from captum.attr import GradientShap
@CaptumAttribution.register('captum-gradientshap')
class CaptumGradientShap(CaptumAttribution, GradientShap):

    def __init__(self, predictor: Predictor):

        CaptumAttribution.__init__(self, 'gradshap', predictor)

        self.submodel = self.predictor._model.captum_sub_model()
        GradientShap.__init__(self, self.submodel)


# ---Integrated Gradients---
from captum.attr import IntegratedGradients
@CaptumAttribution.register('captum-integrated-gradients')
class CaptumIntegratedGradients(CaptumAttribution, IntegratedGradients):

    def __init__(self, predictor: Predictor):

        CaptumAttribution.__init__(self, 'intgrad', predictor)

        self.submodel = self.predictor._model.captum_sub_model()
        IntegratedGradients.__init__(self, self.submodel)



# ---DeepLift---
from captum.attr import DeepLift 
@CaptumAttribution.register('captum-deeplift')
class CaptumDeepLift(CaptumAttribution, DeepLift):

    def __init__(self, predictor: Predictor):

        CaptumAttribution.__init__(self, 'deeplift', predictor)

        self.submodel = self.predictor._model.captum_sub_model()
        DeepLift.__init__(self, self.submodel)


# ---LIME---
from captum.attr import Lime
from captum._utils.models.linear_model import SkLearnRidge
@CaptumAttribution.register('captum-lime')
class CaptumLIME(CaptumAttribution, Lime):

    def __init__(
        self,
        predictor: Predictor,
        mask_features_by_token: bool = True,
        attribute_args: Dict[str, Any] = None
    ):
        CaptumAttribution.__init__(self, 'lime', predictor, mask_features_by_token, attribute_args)
        self.lin_model = SkLearnRidge()
        self.submodel = self.predictor._model.captum_sub_model()
        Lime.__init__(self, self.submodel, self.lin_model)


# ---Feature Ablation---
from captum.attr import FeatureAblation
@CaptumAttribution.register('captum-ablation')
class CaptumFeatureAblation(CaptumAttribution, FeatureAblation):

    def __init__(
        self,
        predictor: Predictor,
        mask_features_by_token: bool = True,
        attribute_args: Dict[str, Any] = None
    ):
        CaptumAttribution.__init__(self, 'feature_ablation', predictor, mask_features_by_token, attribute_args)
        self.submodel = self.predictor._model.captum_sub_model()
        FeatureAblation.__init__(self, self.submodel)
