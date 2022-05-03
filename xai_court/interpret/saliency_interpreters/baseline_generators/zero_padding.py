
from xai_court.interpret.saliency_interpreters.baseline_generators.baseline import Baseline

from allennlp.predictors import Predictor

import torch

@Baseline.register('padding')
class PaddingBaseline(Baseline):

    def __init__(self, predictor: Predictor):
        super().__init__(predictor)

    def baselines_like(self, input_tensor):

        # create padding tensor
        pad_idxs = tuple(pad_idx.expand(tensor.size()[:2]) for tensor in input_tensor)
        pad_idxs
        # retrieve padding embeddings
        baselines = tuple(embedding(idx) for idx in pad_idxs)

        return baselines
