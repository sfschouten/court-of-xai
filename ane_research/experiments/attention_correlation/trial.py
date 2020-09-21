from collections import defaultdict
import itertools
import logging
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

from ane_research.common.correlation_measures import CorrelationMeasure
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
        self.seed = seed
        self.serialization_dir = serialization_dir
        self.predictor = predictor
        self.batch_size = batch_size
        self.feature_importance_interpreters = feature_importance_interpreters
        self.correlation_measures = correlation_measures
        self.correlation_kwargs = {
            'average_length': self._calculate_average_datapoint_length(instances)
        }
        self.dataset = self._batch_dataset(instances)
        self.attention_scores = defaultdict(list)
        self.feature_importance_scores = defaultdict(list)
        self.feature_importance_results = None
        self.correlation_results = None
        self.logger = logging.getLogger(Config.logger_name)
        self.attention_interpreters = self._get_suitable_attention_interpreters()
        self.field_names = self.predictor._model.get_field_names()

    @staticmethod
    def load_results(serialization_dir) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feature_importance = pd.read_pickle(os.path.join(serialization_dir, 'feature_importance.pkl'))
        correlation = pd.read_pickle(os.path.join(serialization_dir, 'correlation.pkl'))
        return feature_importance, correlation

    @staticmethod
    def already_ran(serialization_dir) -> bool:
        feature_importance_exists = os.path.isfile(os.path.join(serialization_dir, 'feature_importance.pkl'))
        correlation_exists = os.path.isfile(os.path.join(serialization_dir, 'correlation.pkl'))
        return feature_importance_exists and correlation_exists

    def _calculate_average_datapoint_length(self, instances: List[Instance]):
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

    def _calculate_feature_importance_batch(self, batch) -> None:

        # Write to dataframe in case we want to generate heatmaps

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
            # we follow the Allennlp convention of returning a dictionary of {instance_id: dict of scores}
            # but there is only one feature importance key per dict of scores, eg. {'instance_x': {'loo_scores': [...]}}
            if 'shap' in interpreter.id or 'deep' in interpreter.id:
                batch_scores = []
                for sub_batch in utils.batch(labeled_batch, 2):
                    batch_scores.extend(interpreter.saliency_interpret_instances(sub_batch).values())
            elif 'intgrad' in interpreter.id:
                # NOTE: the 32 is based on empirical tests with a standard 16GB gpu
                kwargs = {'internal_batch_size': min(len(labeled_batch), self.batch_size, 32)}
                batch_scores = interpreter.saliency_interpret_instances(labeled_batch, **kwargs).values()
            else:
                batch_scores = interpreter.saliency_interpret_instances(labeled_batch).values()

            # # There can be more than one array of scores for an instance (e.g. in the pair sequence case)
            scores = [[np.asarray(scoreset) for scoreset in v.values()] for v in batch_scores]

            if isinstance(interpreter, AttentionInterpreter):
                self.attention_scores[interpreter.id].extend(scores)
            else:
                self.feature_importance_scores[interpreter.id].extend(scores)
            feature_importance_df['scores'].extend(scores)

            feature_importance_df['seed'].extend(seed)
            feature_importance_df['instance_id'].extend(ids)
            feature_importance_df['instance_text'].extend(batch_text)
            feature_importance_df['instance_fields'].extend(fields)
            feature_importance_df['feature_importance_measure'].extend([interpreter.id for _ in range(len(labeled_batch))])
            feature_importance_df['predicted'].extend(predicted_labels)
            feature_importance_df['actual'].extend(actual_labels)

        return feature_importance_df

    def _calculate_correlation_batch(self, batch) -> None:

        corr_df = {
            'seed': [],
            'instance_id': [],
            'instance_text': [],
            'instance_fields': [],
            'predicted': [],
            'actual': [],
            'measure_1': [],
            'measure_2': [],
        }

        ids, labeled_batch, actual_labels = batch

        def _get_scores_by_key_and_instance(key: str, instance_id: str):
            if key in self.attention_scores.keys():
                return np.concatenate(self.attention_scores[key][instance_id])
            else:
                return np.concatenate(self.feature_importance_scores[key][instance_id])

        for measure in self.correlation_measures:
            for field in measure.fields:
                corr_df[field] = []

        feature_importance_measures = list(self.attention_scores.keys()) + list(self.feature_importance_scores.keys())
        correlation_combos = list(itertools.combinations(feature_importance_measures, 2))

        for instance_id, labeled_instance, actual_label in zip(ids, labeled_batch, actual_labels):
            for (key1, key2) in correlation_combos:

                key1_score = _get_scores_by_key_and_instance(key1, instance_id)
                key2_score = _get_scores_by_key_and_instance(key2, instance_id)

                corr_df['seed'].append(self.seed)
                corr_df['instance_id'].append(instance_id)
                corr_df['instance_text'].append([labeled_instance[fn].tokens for fn in self.field_names])
                corr_df['instance_fields'].append(list(self.field_names))
                corr_df['predicted'].append(labeled_instance['label'].label)
                corr_df['actual'].append(actual_label)
                corr_df['measure_1'].append(key1)
                corr_df['measure_2'].append(key2)

                for measure in self.correlation_measures:
                    corr_dict = measure.correlation(key1_score, key2_score, **self.correlation_kwargs)
                    for k, v in corr_dict.items():
                        corr_df[k].append(v)

        return corr_df

    def calculate_scores(self):
        feature_importance_df = defaultdict(list)
        corr_df = defaultdict(list)

        for batch in Tqdm.tqdm(self.dataset):
            importance_scores = self._calculate_feature_importance_batch(batch)
            for k, v in importance_scores.items():
                feature_importance_df[k].extend(v)
            correlation_scores = self._calculate_correlation_batch(batch)
            for k, v in correlation_scores.items():
                corr_df[k].extend(v)
        
        self.correlation_results = pd.DataFrame(corr_df)
        self.feature_importance_results = pd.DataFrame(feature_importance_df)
        utils.write_frame(self.correlation_results, self.serialization_dir, 'correlation')
        utils.write_frame(self.feature_importance_results, self.serialization_dir, 'feature_importance')

    @classmethod
    def from_partial_objects(
        cls,
        seed,
        serialization_dir,
        feature_importance_measures,
        correlation_measures,
        batch_size,
        cuda_device,
        nr_instances = 0 
    ):
        archive = load_archive(os.path.join(serialization_dir, 'model.tar.gz'), cuda_device=cuda_device)
        predictor = Predictor.from_archive(archive, archive.config.params['model']['type'])

        test_instances = list(predictor._dataset_reader.read(archive.config.params['test_data_path']))
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
