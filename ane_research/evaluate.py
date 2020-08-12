from copy import deepcopy
import logging
import math
import os
import itertools
import random 
import statistics 
from typing import List, Tuple

from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.nn import util
from allennlp.data.fields import TextField

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import torch
from tqdm import tqdm

from ane_research.config import Config
from ane_research.models import JWAED
from ane_research.predictors import JWAEDPredictor
from ane_research.utils.correlation import Correlation, CorrelationMeasures
from ane_research.utils.kendall_top_k import kendall_top_k
import ane_research.utils.plotting as plotting


from ane_research.interpret.saliency_interpreters.leave_one_out import LeaveOneOut
from ane_research.interpret.saliency_interpreters.attention_interpreter import AttentionInterpreter 
from ane_research.interpret.saliency_interpreters.lime import LimeInterpreter 
from ane_research.interpret.saliency_interpreters.captum_interpreter import CaptumInterpreter, CaptumDeepLiftShap, CaptumGradientShap
from     allennlp.interpret.saliency_interpreters import SimpleGradient
from     allennlp.interpret.saliency_interpreters import IntegratedGradient

from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class Evaluator():
    '''
    Evaluator class to calculate, plot, and save correlation measures as defined in 'Attention is Not Explanation'
    (Jain and Wallace 2019 - https://arxiv.org/pdf/1902.10186.pdf), namely:
      - Gradient-based feature importance (see 'Attention is Not Explanation' p4. - Algorithm 1: Feature Importance Computations)
      - Feature erasure (a.k.a 'leave one out')-based feature importance
      - Attention weights

    Args
      - model_path (str): Path to archived model generated during an AllenNLP training run
      - calculate_on_init (bool, default: False): Calculate feature importance measures and correlations on class initialization
    '''
    def __init__(self, experiment_name: str, model_path: str, calculate_on_init: bool = False):
        self.precalculate = calculate_on_init
        self.logger = logging.getLogger(Config.logger_name)

        self.experiment_name = experiment_name
        # load a new AllenNLP predictor from the archive
        self.model_path = model_path
        self.model_base_path = os.path.split(self.model_path)[0]
        self.correlation_path = self.model_base_path + '/correlation/'

        ensure_dir(self.correlation_path)
        self.graph_path = self.model_base_path + '/graphs/'
        ensure_dir(self.graph_path)
        self.archive = load_archive(model_path)
        self.model = self.archive.model
        self.predictor = Predictor.from_archive(self.archive, self.archive.config.params['model']['type'])
        self.class_idx2names = self.model.vocab.get_index_to_token_vocabulary('labels')

        # load test instances and split into batches
        self.test_data_path = self.archive.config.params['test_data_path']
        self.test_instances = self.predictor._dataset_reader.read(self.test_data_path)

        random.seed(0)
        NR_INTERPRET_SAMPLES = 500
        subset = random.sample(list(self.test_instances), NR_INTERPRET_SAMPLES)
    
        vocab = self.test_instances.vocab
        self.test_instances = AllennlpDataset(subset, vocab)

        self.batch_size = self.archive.config.params['data_loader']['batch_sampler']['batch_size']
        self.batched_test_instances = list(batch(self.test_instances, self.batch_size))
        self.average_datapoint_length = self._calculate_average_datapoint_length()

        # saliency interpreters
        self.interpreters = {} 
        #TODO replace the following by a constructor parameter
        # self.interpreters['dl_shap'] = CaptumInterpreter(self.predictor, CaptumDeepLiftShap(self.predictor))
        self.interpreters['g_shap'] = CaptumInterpreter(self.predictor, CaptumGradientShap(self.predictor))
        # self.interpreters['lime'] = LimeInterpreter(self.predictor)
        # self.interpreters['loo']  = LeaveOneOut(self.predictor)
        self.interpreters['attn'] = AttentionInterpreter(self.predictor)
        # self.interpreters['grad'] = SimpleGradient(self.predictor)
        # self.interpreters['intgrad'] = IntegratedGradient(self.predictor)

        self.salience_scores = {}

        # measures
        self.labels = []

        # correlations
        self.correlations = {}
        for (key1, key2) in itertools.combinations(self.interpreters, 2):
            self.correlations[(key1, key2)] = Correlation(key1, key2, list(set(self.class_idx2names.values())))

        if self.precalculate:
            self.calculate_feature_importance_measures()
            self.calculate_correlations()

    def _calculate_average_datapoint_length(self):
        num_tokens_per_datapoint = [ sum( len(field) for field_name, field in instance.fields.items() if isinstance(field, TextField) ) for instance in self.test_instances ]
        mean = np.floor(np.mean(num_tokens_per_datapoint))
        return int(mean)

    def calculate_feature_importance_measures(self):
        for key, interpreter in self.interpreters.items():
            counter = iter(range(1, len(self.test_instances) + 1))
            self.salience_scores[key] = {}
            for instance_batch in tqdm(self.batched_test_instances):
                scores = interpreter.saliency_interpret_instances(instance_batch)
                for field, val in scores.items():
                    scoresets = [ np.asarray(list(scoreset)) for _, scoreset in val.items()]
                    self.salience_scores[key][f'instance_{next(counter)}'] = scoresets 

            for instance_batch in self.batched_test_instances:
                batch_outputs = self.model.forward_on_instances(instance_batch)
                batch_labels = [batch_output['label'] for batch_output in batch_outputs]
                self.labels.extend(batch_labels)
      
    def calculate_correlations(self):
        avg_length = self.average_datapoint_length
        non_zero_k = { (key1, key2) : [] for (key1,_), (key2,_) in itertools.combinations(self.salience_scores.items(), 2) }

        # Calculate kendalltau, kendall_tau_top_k_non_zero, and kendall_tau_top_k_average_length for each datapoint
        for i in tqdm(range(len(self.test_instances))):
            label = self.labels[i]
            class_name = self.class_idx2names[int(label)]

            # iterate over all possible pairs of saliency interpreters
            for (key1, scoresets1), (key2, scoresets2) in itertools.combinations(self.salience_scores.items(), 2):

                # get the current instance's scores for current 2 interpreters
                scoresets1 = scoresets1[f'instance_{i+1}']
                scoresets2 = scoresets2[f'instance_{i+1}']

                # chain the scoresets (one for each TextField in the instances).
                # (e.g. for pair-sequence classification there would be 2).
                score1 = np.concatenate(scoresets1)
                score2 = np.concatenate(scoresets2)

                if len(score1) != len(score2):
                    self.logger.error(f"List of scores for {key1} and {key2} were not equal length! ({len(score1)}) vs. ({len(score2)})")
                    self.logger.debug(f"Relevant instance: {self.test_instances[i]}")
                    continue

                self.correlations[(key1, key2)].calculate_kendall_tau_correlation(score1, score2, class_name=class_name)
                self.correlations[(key1, key2)].calculate_kendall_top_k_average_length_correlation(score1, score2, average_length=avg_length, p=0.5, class_name=class_name)
                _, k = self.correlations[(key1, key2)].calculate_kendall_top_k_non_zero_correlation(score1, score2, kIsNonZero=True, p=0.5, class_name=class_name)
                non_zero_k[(key1, key2)].append(k)
    
            # Important: 'apples to apples' comparison: ensure the correlation calculation between the various saliency interpreters
            # uses the same k value for the kendall_top_k calculation as was used for the previous correlation calculations
            recent_ks = { (key1,key2): ks[-1] for (key1, key2), ks in non_zero_k.items() }
      
            if len(set(recent_ks.values())) > 1:
                self.logger.warning(f"Not all k values used were the same across the different comparison pairs!")
                self.logger.info(recent_ks)

        canon_key = next(iter(non_zero_k.keys()))
        mean = statistics.mean(non_zero_k[canon_key])
        stdev = statistics.stdev(non_zero_k[canon_key])
        print(f'k-statistics (using {canon_key})')
        print(f'mean: {mean}')
        print(f'variance: {stdev}')

    def generate_and_save_correlation_data_frames(self):
        csvs = []
        for correlation in self.correlations.values():
            k_tau_df = correlation.generate_kendall_tau_data_frame()
            csvs.append((correlation.id, 'kendall_tau', k_tau_df))
            top_k_average_length = correlation.generate_kendall_top_k_data_frame(CorrelationMeasures.KENDALL_TOP_K_AVG_LEN)
            csvs.append((correlation.id, 'kendall_top_k_average_length', top_k_average_length))
            top_k_non_zero = correlation.generate_kendall_top_k_data_frame(CorrelationMeasures.KENDALL_TOP_K_NON_ZERO)
            csvs.append((correlation.id, 'kendall_top_k_non_zero', top_k_non_zero))
        for (correlation_id, correlation_name, df) in csvs:
            csv_path = f'{self.correlation_path}/{correlation_id}_{correlation_name}.csv'
            with open(csv_path, 'w') as out_path:
                df.to_csv(out_path, index=True)

    def generate_and_save_correlation_graphs(self):
        fig, ax = plotting.init_gridspec(3, 3, 3)
        for i, correlation_measure in enumerate(CorrelationMeasures):
            axes = ax[i]
            linestyles = ['-', '--', ':']
            correlations = [(correlation.id, correlation.get_total_correlation(correlation_measure)) for correlation in self.correlations.values()]
            plotting.generate_correlation_density_plot(ax=axes,correlations=correlations)
            plot_title = f'{self.experiment_name}_{correlation_measure.value}'
            plotting.annotate(ax=axes, title=plot_title)
            plotting.adjust_gridspec()
            plotting.save_axis_in_file(fig, axes, self.graph_path, plot_title)

