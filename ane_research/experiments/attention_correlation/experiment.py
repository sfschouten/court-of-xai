import logging
from os import PathLike
from typing import List

from allennlp.common import Registrable
import pandas as pd

from ane_research.common.correlation_measures import CorrelationMeasure
import ane_research.common.utils as utils
from ane_research.config import Config
from ane_research.experiments.attention_correlation.trial import AttentionCorrelationTrial


class AttentionCorrelationExperiment(Registrable):

    default_implementation = "default"

    def __init__(
        self,
        dataset: str,
        model: str,
        compatibility_function: str,
        activation_function: str,
        serialization_dir: PathLike,
        feature_importance_measures: List[str],
        attention_measures: List[str],
        correlation_measures: List[CorrelationMeasure],
        feature_importance_results: pd.DataFrame,
        correlation_results: pd.DataFrame
    ):
        self.dataset = dataset
        self.model = model
        self.compatibility_function = compatibility_function
        self.activation_function = activation_function
        self.serialization_dir = serialization_dir
        self.feature_importance_measures = feature_importance_measures
        self.attention_measures = attention_measures

        self.correlation_measures = correlation_measures
        self.feature_importance = feature_importance_results
        self.correlation = correlation_results

        self.logger =  logging.getLogger(Config.logger_name)


    def _summarize(self):
        summary = self.correlation.copy()
        summary = summary.drop(['instance_id', 'seed', 'actual', 'predicted', 'instance_fields', 'instance_text'], axis=1)
        summary = summary.groupby(['feature_importance_measure_1', 'feature_importance_measure_2', 'correlation_measure'], as_index = False).mean()
        self.summary = summary

    def generate_artifacts(self):
        self._summarize()
        # Generate Frames
        utils.write_frame(self.summary, self.serialization_dir, 'summary')
        utils.write_frame(self.correlation, self.serialization_dir, 'correlations_all')
        utils.write_frame(self.feature_importance, self.serialization_dir, 'feature_importance_all')

        # Save config
        config = {
            "dataset": self.dataset,
            "model": self.model,
            "compatibility_function": self.compatibility_function,
            "activation_function": self.activation_function,
            "feature_importance_measures": self.feature_importance_measures,
            "attention_measures": self.attention_measures
        }
        utils.write_json(config, self.serialization_dir, 'config')

    @classmethod
    def from_completed_trials(
        cls,
        dataset: str,
        model: str,
        compatibility_function: str,
        activation_function: str,
        serialization_dir: str,
        trials: List[AttentionCorrelationTrial]
    ):
        # TODO: handle case if different trials have different measures? Perhaps take intersection?
        feature_importance_measures = [fi.id for fi in trials[0].feature_importance_interpreters]
        attention_measures = [ai.id for ai in trials[0].attention_interpreters]
        correlation_measures = trials[0].correlation_measures

        feature_importance_results = pd.concat([trial.feature_importance_results for trial in trials])
        correlation_results = pd.concat([trial.correlation_results for trial in trials])

        return cls(
            dataset=dataset,
            model=model,
            compatibility_function=compatibility_function,
            activation_function=activation_function,
            serialization_dir=serialization_dir,
            feature_importance_measures=feature_importance_measures,
            attention_measures=attention_measures,
            correlation_measures=correlation_measures,
            feature_importance_results=feature_importance_results,
            correlation_results=correlation_results
        )

AttentionCorrelationExperiment.register("default", constructor="from_completed_trials")(AttentionCorrelationExperiment)
