
from xai_court.interpret.saliency_interpreters.baseline_generators.baseline import Baseline

from allennlp.predictors import Predictor

import torch

@Baseline.register('padding')
class PaddingBaseline(Baseline):

    def __init__(self, predictor: Predictor):
        super().__init__(predictor)

        vocab = self.predictor._model.vocab
        self._pad_idx = vocab.get_token_index(vocab._padding_token)

    def baselines_like(self, input_tensor: torch.Tensor):

        # create padding tensor
        pad_idx = torch.LongTensor([[pad_idx]]).to(input_tensor[0].device)
        pad_idxs = tuple(pad_idx.expand(tensor.size()[:2]) for tensor in input_tensor)

        # retrieve padding embeddings
        baselines = tuple(embedding(idx) for idx in pad_idxs)

        return baselines
