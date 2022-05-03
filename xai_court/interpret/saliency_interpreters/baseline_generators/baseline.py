
from allennlp.common import Registrable
from allennlp.predictors import Predictor
from allennlp.data import DatasetReader

import torch

class Baseline(Registrable):

    default_implementation = "padding"

    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def baselines_like(self, input_tensor: torch.Tensor):
        """
        Args:
            input_tensor:

        Returns:
            A tensor of baselines based on the input_tensor.
        """
        raise NotImplementedError()



class BaselineCompatibleInterpreter():

    def __init__(self, baseline: Baseline):
        self.baseline = baseline