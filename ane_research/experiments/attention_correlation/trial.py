from collections import defaultdict
import itertools
import logging
import math
import os
import random
from os import PathLike
from typing import Any, Dict, List, Generator, Optional, Tuple, Union

from allennlp.common import Lazy, Registrable, Tqdm
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.interpret import SaliencyInterpreter
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import numpy as np
import pandas as pd
import statistics
import torch

from ane_research.common.correlation_measures import CorrelationMeasure
import ane_research.common.utils as utils
from ane_research.config import Config
from ane_research.interpret.saliency_interpreters.attention import AttentionInterpreter


InstanceBatch = Tuple[List[int], List[Instance], List[LabelField]]


class AttentionCorrelationTrial(Registrable):

    default_implementation = "default"

    def __init__(
        self,
        seed: int,
        serialization_dir: PathLike,
        predictor: Predictor,
        instances: List[Instance],
        attention_interpreters: List[AttentionInterpreter],
        feature_importance_interpreters: List[SaliencyInterpreter],
        correlation_measures: List[CorrelationMeasure],
        batch_size: int,
        logger: Optional[logging.Logger] = None
    ):
        # Trial parameters
        self.seed = seed
        self.serialization_dir = serialization_dir
        self.predictor = predictor
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(Config.logger_name)
        self.field_names = self.predictor._model.get_field_names()

        # Interpreters
        self.attention_interpreters = attention_interpreters
        self.feature_importance_interpreters = feature_importance_interpreters

        # Correlation Measures
        self.correlation_measures = correlation_measures
        self.correlation_combos = list(
            itertools.combinations(
                [fi.id for fi in self.feature_importance_interpreters] \
                + [ai.id for ai in self.attention_interpreters], 2
            )
        )

        # Dataset
        self.dataset = self._batch_dataset(instances)
        self.num_batches = math.ceil(len(instances) / self.batch_size)

        # Dataframes
        self.feature_importance_results = None
        self.correlation_results = None


    def _batch_dataset(
        self,
        unlabeled_instances: List[Instance]
    ) -> Generator[InstanceBatch, None, None]:
        ids = iter(range(len(unlabeled_instances)))
        for b_id, instance_batch in enumerate(utils.batch(unlabeled_instances, self.batch_size)):

            batch_outputs = self.predictor._model.forward_on_instances(instance_batch)
            actual_labels = [instance["actual"] for instance in batch_outputs]

            batch_ids = [next(ids) for _ in range(len(instance_batch))]

            labeled_batch = [ \
                self.predictor.predictions_to_labeled_instances(instance, outputs)[0] \
                for instance, outputs in zip(instance_batch, batch_outputs) \
            ]

            yield (batch_ids, labeled_batch, actual_labels)


    def _calculate_feature_importance_batch(self, batch: InstanceBatch, progress_bar: Tqdm = None) -> None:
        feature_importance_df = defaultdict(list)

        ids, labeled_batch, actual_labels = batch
        batch_text = [[li[fn].tokens for fn in self.field_names] for li in labeled_batch]
        fields = [list(self.field_names) for _ in range(len(labeled_batch))]
        predicted_labels = [li['label'].label for li in labeled_batch]
        seed = [self.seed for _ in range(len(labeled_batch))]

        for interpreter in self.feature_importance_interpreters + self.attention_interpreters:

            if progress_bar:
                progress_bar.set_description(f"{interpreter.id}: interpreting {len(labeled_batch)} instances")

            # Some feature importance measures are too memory-intensive to run with larger batch sizes
            # These numbers are based on empirical tests with a standard 16GB gpu
            if 'shap' in interpreter.id or 'deep' in interpreter.id or 'intgrad' in interpreter.id:
                batch_scores = []
                for sub_batch in utils.batch(labeled_batch, 2):
                    batch_scores.extend(interpreter.saliency_interpret_instances(sub_batch).values())
            else:
                batch_scores = interpreter.saliency_interpret_instances(labeled_batch).values()

            # # There can be more than one array of scores for an instance (e.g. in the pair sequence case)
            scores = [[np.asarray(scoreset) for scoreset in v.values()] for v in batch_scores]

            feature_importance_df['scores'].extend(scores)
            feature_importance_df['seed'].extend(seed)
            feature_importance_df['instance_id'].extend(ids)
            feature_importance_df['instance_text'].extend(batch_text)
            feature_importance_df['instance_fields'].extend(fields)
            feature_importance_df['feature_importance_measure'].extend([interpreter.id for _ in range(len(labeled_batch))])
            feature_importance_df['predicted'].extend(predicted_labels)
            feature_importance_df['actual'].extend(actual_labels)

            if progress_bar:
                progress_bar.update(1)

        return feature_importance_df


    def calculate_feature_importance(self, force: bool = False) -> None:
        pkl_exists = os.path.isfile(os.path.join(self.serialization_dir, 'feature_importance.pkl'))

        if pkl_exists and not force:
            self.logger.info("Feature importance scores exist and force was not specified. Loading from disk...")
            self.feature_importance_results = pd.read_pickle(os.path.join(self.serialization_dir, 'feature_importance.pkl'))
        else:
            feature_importance_df = defaultdict(list)
            self.logger.info('Calculating feature importance scores...')

            num_interpreters = len(self.feature_importance_interpreters) + len(self.attention_interpreters)
            progress_bar = Tqdm.tqdm(total=self.num_batches * num_interpreters)

            for batch in self.dataset:
                importance_scores = self._calculate_feature_importance_batch(batch, progress_bar)
                for k, v in importance_scores.items():
                    feature_importance_df[k].extend(v)

            self.feature_importance_results = pd.DataFrame(feature_importance_df)
            utils.write_frame(self.feature_importance_results, self.serialization_dir, 'feature_importance')


    def _calculate_correlation_combo(
        self,
        key1: str,
        key2: str,
        correlation_kwargs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, List[int]]]:
        """Calculate the correlation between the scores of two interpreters in the feature importance dataframe

        Args:
            key1 (str):
                Id of the first interpreter
            key2 (str):
                Id of the second interpreter
            correlation_kwargs (Optional[Dict[str, Dict[str, Any]]]):
                A mapping of CorrelationMeasure.id -> kwargs for passing **kwargs to specific CorrelationMeasures

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]:
                A tuple of:
                    - combination_frame (Dict[str, Any])
                        A dictionary of results ready to be aggregated into the larger correlation dataframe
                    - unfair_fair_k_values (Dict[str, List[int]])
                       A mapping of the CorrelationMeasure.id to a list of k values used by CorrelationMeasures
                       which may be unfair when compared in isolation. These k values can be used to ensure a
                       fair comparison later on.
        """
        if correlation_kwargs is None:
            correlation_kwargs = {}

        corr_df = defaultdict(list)
        unfair_k_values = defaultdict(list)

        key1_mask = self.feature_importance_results['feature_importance_measure'].values == key1
        key2_mask = self.feature_importance_results['feature_importance_measure'].values == key2
        relevant_scores = self.feature_importance_results[key1_mask | key2_mask]
        relevant_scores = relevant_scores.groupby('instance_id').agg(lambda x: x.tolist())

        for row in relevant_scores.itertuples():
            instance_id, seed, text, fields = row.Index, row.seed[0], row.instance_text[0], row.instance_fields[0]
            predicted, actual = row.predicted[0], row.actual[0]
            measure_1, measure_2, key1_scores, key2_scores = *row.feature_importance_measure, *row.scores
            key1_scores, key2_scores = np.concatenate(key1_scores), np.concatenate(key2_scores)

            for measure in self.correlation_measures:
                kwargs = correlation_kwargs.get(measure.id) or {}
                corr_dict = measure.correlation(key1_scores, key2_scores, **kwargs)
                for name, result in corr_dict.items():
                    corr_df['instance_id'].append(instance_id)
                    corr_df['seed'].append(seed)
                    corr_df['instance_text'].append(text)
                    corr_df['instance_fields'].append(fields)
                    corr_df['predicted'].append(predicted)
                    corr_df['actual'].append(actual)
                    corr_df['feature_importance_measure_1'].append(measure_1)
                    corr_df['feature_importance_measure_2'].append(measure_2)
                    corr_df['correlation_measure'].append(name)
                    corr_df['correlation_value'].append(result.correlation)
                    corr_df['k'].append(result.k)

                    if measure.unfair_in_isolation:
                        unfair_k_values[measure.id].append(result.k)

        return corr_df, unfair_k_values


    def calculate_correlation(self, force: bool = False) -> None:
        pkl_exists = os.path.isfile(os.path.join(self.serialization_dir, 'correlation.pkl'))

        if pkl_exists and not force:
            self.logger.info("Correlations exist and force was not specified. Loading from disk...")
            self.correlation_results = pd.read_pickle(os.path.join(self.serialization_dir, 'correlation.pkl'))
        else:
            correlation_df = defaultdict(list)
            self.logger.info('Calculating correlations...')

            progress_bar = Tqdm.tqdm(total=len(self.correlation_combos))

            # We need to compare combinations with at least one attention interpreter first to get the k_values
            # for an apples to apples comparison with combinations where both interpreters are
            # feature importance measures
            unfair_k = defaultdict(lambda: defaultdict(list))
            for (key1, key2) in self.correlation_combos:
                if 'attn' in key1 or 'attn' in key2:
                    correlations, unfair_k_values = self._calculate_correlation_combo(key1, key2)

                    for key, values in correlations.items():
                        correlation_df[key].extend(values)

                    for measure, k in unfair_k_values.items():
                        unfair_k[key1][measure].extend(k)
                        unfair_k[key2][measure].extend(k)

                    progress_bar.update(1)

            # Now we can compare the feature importance measures to each other
            for (key1, key2) in self.correlation_combos:
                if 'attn' not in key1 and 'attn' not in key2:
                    correlation_kwargs = defaultdict(list)

                    # Unfair k strategy: take the average k used for each key
                    for name, k_values in unfair_k.get(key1, {}).items():
                        correlation_kwargs[name].extend(k_values)
                    for name, k_values in unfair_k.get(key2, {}).items():
                        correlation_kwargs[name].extend(k_values)
                    for name, k_values in correlation_kwargs.items():
                        correlation_kwargs[name] = {"k": math.floor(statistics.mean(k_values))}

                    correlations, _ = self._calculate_correlation_combo(key1, key2, correlation_kwargs=correlation_kwargs)

                    for k, v in correlations.items():
                        correlation_df[k].extend(v)
                    progress_bar.update(1)

            self.correlation_results = pd.DataFrame(correlation_df)
            utils.write_frame(self.correlation_results, self.serialization_dir, 'correlation')


    @classmethod
    def from_partial_objects(
        cls,
        seed: int,
        serialization_dir: PathLike,
        test_data_path: PathLike,
        feature_importance_measures: List[Lazy[SaliencyInterpreter]],
        correlation_measures: List[CorrelationMeasure],
        batch_size: int,
        attention_aggregator_methods: Optional[List[str]] = None,
        attention_analysis_methods: Optional[List[str]] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        nr_instances: Optional[int] = 0
    ):
        logger = logging.getLogger(Config.logger_name)

        archive = load_archive(os.path.join(serialization_dir, 'model.tar.gz'), cuda_device=cuda_device)
        predictor = Predictor.from_archive(archive, archive.config.params['model']['type'])

        test_instances = list(predictor._dataset_reader.read(test_data_path))
        if nr_instances:
            logger.info(f'Selecting a random subset of {nr_instances} for interpretation')
            random.seed(seed)
            test_instances = random.sample(test_instances, min(len(test_instances), nr_instances))

        attention_interpreters = []
        for aggregator_type in predictor.get_suitable_aggregators():
            for analysis in predictor._model.supported_attention_analysis_methods:
                aggregator = aggregator_type()
                if attention_aggregator_methods != None and aggregator.id not in attention_aggregator_methods:
                    logger.info(
                        f"Combination of aggregator method '{aggregator.id}' and analysis method "
                        f"'{analysis.value}' not requested and will be ignored"
                    )
                    continue
                if attention_analysis_methods != None and analysis.value not in attention_analysis_methods:
                    logger.info(
                        f"Combination of analysis method '{analysis.value}' and aggregator method "
                        f"'{aggregator.id}' not requested and will be ignored"
                    )
                    continue
                attention_interpreters.append(
                    AttentionInterpreter(
                        predictor=predictor,
                        analysis_method=analysis,
                        aggregate_method=aggregator
                    )
                )


        feature_importance_interpreters = [fi.construct(predictor=predictor) for fi in feature_importance_measures]

        return cls(
            seed=seed,
            serialization_dir=serialization_dir,
            predictor=predictor,
            instances=test_instances,
            attention_interpreters=attention_interpreters,
            feature_importance_interpreters=feature_importance_interpreters,
            correlation_measures=correlation_measures,
            batch_size=batch_size,
            logger=logger
        )

AttentionCorrelationTrial.register("default", constructor="from_partial_objects")(AttentionCorrelationTrial)
