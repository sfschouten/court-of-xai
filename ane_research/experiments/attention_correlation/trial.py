from collections import defaultdict
import itertools
import logging
import math
import os
import random
from os import PathLike
from typing import Any, Dict, List, Tuple, Generator

from allennlp.common import Registrable, Tqdm
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.interpret import SaliencyInterpreter
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import numpy as np
import pandas as pd

from ane_research.common.correlation_measures import CorrelationMeasure, KendallTauTopKNonZero
import ane_research.common.utils as utils
from ane_research.config import Config
from ane_research.interpret.saliency_interpreters.attention import AttentionInterpreter


class AttentionCorrelationTrial(Registrable):

    default_implementation = "default"

    def __init__(
        self,
        seed: int,
        serialization_dir: PathLike,
        predictor: Predictor,
        instances: List[Instance],
        feature_importance_interpreters: List[SaliencyInterpreter],
        correlation_measures: List[CorrelationMeasure],
        batch_size: int
    ):
        # Trial parameters
        self.seed = seed
        self.serialization_dir = serialization_dir
        self.predictor = predictor
        self.batch_size = batch_size
        self.logger = logging.getLogger(Config.logger_name)
        self.field_names = self.predictor._model.get_field_names()

        # Interpreters
        self.feature_importance_interpreters = feature_importance_interpreters
        self.attention_interpreters = self._get_suitable_attention_interpreters()

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
        self.dataset_length = len(instances)
        self.average_data_point_length  = self._calculate_average_datapoint_length(instances)
        self.num_batches = math.ceil(len(instances) / self.batch_size)

        # Dataframes
        self.feature_importance_results = None
        self.correlation_results = None


    def _calculate_average_datapoint_length(self, instances: List[Instance]) -> int:
        """
        Calculate the average length of a collection of instances. The length of an Instance with multiple TextFields
        is treated as the total length of all TextFields.

        Args:
            instances (List[Instance]): Collection of Instances

        Returns:
            int: Average length
        """
        num_tokens_per_datapoint = [ \
                sum( len(field) for field_name, field in instance.fields.items() if isinstance(field, TextField) ) \
                for instance in instances \
            ]
        mean = np.floor(np.mean(num_tokens_per_datapoint))
        return int(mean)


    def _batch_dataset(
        self,
        unlabeled_instances: List[Instance]
    ) -> Generator[List[Tuple[Instance, LabelField]], None, None]:

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


    def _get_suitable_attention_interpreters(self) -> List[AttentionInterpreter]:
        attention_interpreters = []
        for aggregator_type in self.predictor.get_suitable_aggregators():
            for analysis in self.predictor._model.supported_attention_analysis_methods:
                aggregator = aggregator_type()
                attention_interpreters.append(AttentionInterpreter(self.predictor, analysis, aggregator))
        return attention_interpreters


    def _calculate_feature_importance_batch(self, batch, progress_bar = None) -> None:

        feature_importance_df = {
            'seed': [],
            'instance_id': [],
            'instance_text': [],
            'instance_fields': [],
            'feature_importance_measure': [],
            'scores': [],
            'predicted': [],
            'actual': []
        }

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
            if 'shap' in interpreter.id or 'deep' in interpreter.id:
                batch_scores = []
                for sub_batch in utils.batch(labeled_batch, 2):
                    batch_scores.extend(interpreter.saliency_interpret_instances(sub_batch).values())
            elif 'intgrad' in interpreter.id:
                kwargs = {'internal_batch_size': min(len(labeled_batch), self.batch_size, 32)}
                batch_scores = interpreter.saliency_interpret_instances(labeled_batch, **kwargs).values()
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

    def _calculate_correlation_combo(self, key1, key2):

        corr_df = defaultdict(list)

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
                corr_dict = measure.correlation(key1_scores, key2_scores)
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

        return corr_df


    def calculate_correlation(self, force: bool = False) -> None:
        pkl_exists = os.path.isfile(os.path.join(self.serialization_dir, 'correlation.pkl'))

        if pkl_exists and not force:
            self.logger.info("Correlations exist and force was not specified. Loading from disk...")
            self.correlation_results = pd.read_pickle(os.path.join(self.serialization_dir, 'correlation.pkl'))
        else:
            correlation_df = defaultdict(list)
            self.logger.info('Calculating correlations...')

            progress_bar = Tqdm.tqdm(total=len(self.correlation_combos))

            for (key1, key2) in self.correlation_combos:
                correlations = self._calculate_correlation_combo(key1, key2)

                for k, v in correlations.items():
                    correlation_df[k].extend(v)

                progress_bar.update(1)

            self.correlation_results = pd.DataFrame(correlation_df)
            utils.write_frame(self.correlation_results, self.serialization_dir, 'correlation')


    @classmethod
    def from_partial_objects(
        cls,
        seed,
        serialization_dir,
        test_data_path,
        feature_importance_measures,
        correlation_measures,
        batch_size,
        cuda_device,
        nr_instances = 0
    ):
        archive = load_archive(os.path.join(serialization_dir, 'model.tar.gz'), cuda_device=cuda_device)
        predictor = Predictor.from_archive(archive, archive.config.params['model']['type'])

        test_instances = list(predictor._dataset_reader.read(test_data_path))
        if nr_instances:
            random.seed(seed)
            test_instances = random.sample(test_instances, min(len(test_instances), nr_instances))

        feature_importance_interpreters = [SaliencyInterpreter.from_params(params=fim, predictor=predictor) for fim in feature_importance_measures]
        correlation_measures = [CorrelationMeasure.from_params(cm) for cm in correlation_measures]

        return cls(
            seed=seed,
            serialization_dir=serialization_dir,
            predictor=predictor,
            instances=test_instances,
            feature_importance_interpreters=feature_importance_interpreters,
            correlation_measures=correlation_measures,
            batch_size=batch_size
        )

AttentionCorrelationTrial.register("default", constructor="from_partial_objects")(AttentionCorrelationTrial)
