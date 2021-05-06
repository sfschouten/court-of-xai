import logging
import re

from typing import Dict, Iterable, Any

from allennlp.training.trainer import EpochCallback

logger = logging.getLogger(__name__)

@EpochCallback.register('print-parameter')
class PrintParameter:

    def __init__(self, param_re: str):
        """
        Params:
            param_re: regular expression 
        """
        self.param_re = param_re

    def __call__(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ) -> None:
        for name, param in trainer.model.named_parameters():
            if re.match(self.param_re, name):
                logger.info(f'{name} = {param.data}')
