from copy import deepcopy
import logging
import math
import os
import itertools
from typing import List, Tuple

from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.nn import util

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
from ane_research.interpret.saliency_interpreters.captum_interpreter import CaptumInterpreter, CaptumDeepLiftShap
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

class JWAEDEvaluator():
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
  def __init__(self, model_path: str, calculate_on_init: bool = False):
    self.precalculate = calculate_on_init
    self.logger = logging.getLogger(Config.logger_name)

    # load a new AllenNLP predictor from the archive
    self.model_path = model_path
    self.model_base_path = os.path.split(self.model_path)[0]
    self.correlation_path = self.model_base_path + '/correlation/'

    model_specifics = os.path.split(os.path.split(self.model_base_path)[0])[1].split('_')
    self.dataset_name = model_specifics[0]
    self.attention_type = model_specifics[1]
    self.attention_activation_function = model_specifics[2]

    ensure_dir(self.correlation_path)
    self.graph_path = self.model_base_path + '/graphs/'
    ensure_dir(self.graph_path)
    self.archive = load_archive(model_path)
    self.model = self.archive.model
    self.predictor = Predictor.from_archive(self.archive, 'jain_wallace_attention_binary_classification_predictor')
    self.class_names = list(set(self.model.vocab.get_index_to_token_vocabulary('labels').values()))

    # load test instances and split into batches
    self.test_data_path = self.archive.config.params['test_data_path']
    self.test_instances = self.predictor._dataset_reader.read(self.test_data_path)
    
    # TODO remove this, just to speed up debugging.
    #subset = list(itertools.islice(self.test_instances, 100))
    #vocab = self.test_instances.vocab
    #self.test_instances = AllennlpDataset(subset, vocab)

    self.batch_size = self.archive.config.params['data_loader']['batch_sampler']['batch_size']
    self.batched_test_instances = list(batch(self.test_instances, self.batch_size))
    self.average_datapoint_length = self._calculate_average_datapoint_length()

    # saliency interpreters
    self.interpreters = {} 
    #TODO replace the following by a constructor parameter
    self.interpreters['dl_shap'] = CaptumInterpreter(self.predictor, CaptumDeepLiftShap(self.predictor))
    #self.interpreters['lime'] = LimeInterpreter(self.predictor)
    #self.interpreters['loo']  = LeaveOneOut(self.predictor)
    self.interpreters['attn'] = AttentionInterpreter(self.predictor)
    #self.interpreters['grad'] = SimpleGradient(self.predictor)
    self.interpreters['intgrad'] = IntegratedGradient(self.predictor)

    self.salience_scores = {}

    # measures
    self.labels = []

    # correlations
    self.correlations = {}
    for (key1, key2) in itertools.combinations(self.interpreters, 2):
      self.correlations[(key1, key2)] = Correlation(key1, key2, self.class_names)

    if self.precalculate:
      self.calculate_feature_importance_measures()
      self.calculate_correlations()

  def _calculate_average_datapoint_length(self):
    num_tokens_per_datapoint = [len(instance.fields['tokens']) for instance in self.test_instances]
    return math.floor(np.mean(num_tokens_per_datapoint))

  def calculate_feature_importance_measures(self):
    for key, interpreter in self.interpreters.items():
      self.salience_scores[key] = interpreter.saliency_interpret_dataset(self.test_instances, self.batch_size)

    instances = self.test_instances
    batches = [ instances[x:x+self.batch_size] for x in range(0, len(instances), self.batch_size) ]
    self.labels = []
    for batch in batches:
        batch_outputs = self.model.forward_on_instances(batch)
        batch_labels = [batch_output['label'] for batch_output in batch_outputs]
        self.labels.extend(batch_labels)
      

  def calculate_correlations(self):
    # Calculate kendalltau, kendall_tau_top_k_non_zero, and kendall_tau_top_k_average_length for each datapoint
    for i in tqdm(range(len(self.test_instances))):
      L = len(self.test_instances[i].fields['tokens']) # sequence length
      
      class_name = str(self.labels[i])

      for (key1, scoreset1), (key2, scoreset2) in itertools.combinations(self.salience_scores.items(), 2):
        score1 = scoreset1[f'instance_{i+1}']
        score2 = scoreset2[f'instance_{i+1}']

        score1 = next(iter(score1.values()))
        score2 = next(iter(score2.values()))
    
        if len(score1) != len(score2):
            self.logger.error(f"List of scores for {key1} and {key2} were not equal length!")
            self.logger.debug(f"Relevant instance: {self.test_instances[i]}")
            continue

        self.correlations[(key1, key2)].calculate_kendall_tau_correlation(score1, score2, class_name=class_name)

        avg_length = self.average_datapoint_length
        self.correlations[(key1, key2)].calculate_kendall_top_k_average_length_correlation(score1, score2, average_length=avg_length, p=0.5, class_name=class_name)
        
        self.correlations[(key1, key2)].calculate_kendall_top_k_non_zero_correlation(score1, score2, kIsNonZero=True, p=0.5, class_name=class_name)
        
      #TODO reimplement different k warnings.

      # Important: 'apples to apples' comparison: ensure the correlation calculation between the gradient and feature erasure measures
      # uses the same k value for the kendall_top_k calculation as was used for the previous correlation calculations
      #if k_af != k_ag:
      #  self.logger.warning(f'kendall_top_k_non_zero mismatched k_value {k_af} != {k_ag}')
      #_, k_fg = self.loo_gradient_correlation.calculate_kendall_top_k_non_zero_correlation(loo, gradients, k=k_af, kIsNonZero=False, class_name=class_name)
      #if k_fg != k_af:
      #  self.logger.warning(f'Used different k {k_fg}')

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
      plot_title = f'{self.dataset_name}_{self.attention_type}_{self.attention_activation_function}_{correlation_measure.value}'
      plotting.annotate(ax=axes, title=plot_title)
      plotting.adjust_gridspec()
      plotting.save_axis_in_file(fig, axes, self.graph_path, plot_title)
