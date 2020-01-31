from copy import deepcopy
import logging
import math
import os
from typing import List, Tuple

from allennlp.data.dataset import Batch
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
    self.batch_size = self.archive.config.params['iterator']['batch_size']
    self.batched_test_instances = list(batch(self.test_instances, self.batch_size))
    self.average_datapoint_length = self._calculate_average_datapoint_length()

    # measures
    self.labels = []
    self.predictions = []
    self.feature_erasure_predictions = []
    self.attention_weights = []
    self.gradient_feature_importance = []

    # correlations
    self.attention_gradient_correlation = Correlation('attention', 'gradient', self.class_names)
    self.feature_erasure_gradient_correlation = Correlation('feature_erasure', 'gradient', self.class_names)
    self.attention_feature_erasure_correlation = Correlation('attention', 'feature_erasure', self.class_names)
    self.correlations = [
      self.attention_gradient_correlation,
      self.feature_erasure_gradient_correlation,
      self.attention_feature_erasure_correlation
    ]

    if self.precalculate:
      self.calculate_feature_importance_measures()
      self.calculate_correlations()

  def _calculate_average_datapoint_length(self):
    num_tokens_per_datapoint = [len(instance.fields['tokens']) for instance in self.test_instances]
    return math.floor(np.mean(num_tokens_per_datapoint))

  def calculate_feature_importance_measures(self):
    self.predictions = []
    self.feature_erasure_predictions = []
    self.attention_weights = []
    self.gradient_feature_importance = []
    self.labels = []
    for instance_batch in tqdm(self.batched_test_instances):
      batch_outputs = self.model.forward_on_instances(instance_batch)
      batch_predictions = [batch_output['prediction'] for batch_output in batch_outputs]
      batch_attention_weights = [batch_output['attention'] for batch_output in batch_outputs]
      batch_labels = [batch_output['label'] for batch_output in batch_outputs]
      batch_feature_erasure_predictions = self.predictor.predict_batch_instances_with_feature_erasure(instance_batch)['prediction']
      batch_gradient_feature_importance = self.predictor.batch_instances_normalized_gradient_based_feature_importance(instance_batch)
      for i in range(len(instance_batch)):
        self.attention_weights.append(batch_attention_weights[i])
        self.predictions.append(batch_predictions[i])
        self.feature_erasure_predictions.append(batch_feature_erasure_predictions[i])
        self.gradient_feature_importance.append(batch_gradient_feature_importance[i])
        self.labels.append(batch_labels[i])

  def calculate_correlations(self):
    # Calculate kendalltau, kendall_tau_top_k_non_zero, and kendall_tau_top_k_average_length for each datapoint
    for i in tqdm(range(len(self.test_instances))):
      L = len(self.test_instances[i].fields['tokens']) # sequence length
      prediction_difference = np.abs(self.feature_erasure_predictions[i] - self.predictions[i]).mean(-1)[1:L-1]
      attention = list(self.attention_weights[i][1:L-1])
      gradients = list(self.gradient_feature_importance[i][1:L-1])
      class_name = str(self.labels[i])

      # f -> feature erasure, a -> attention, g -> gradients
      self.attention_feature_erasure_correlation.calculate_kendall_tau_correlation(attention, prediction_difference, class_name=class_name)
      self.attention_feature_erasure_correlation.calculate_kendall_top_k_average_length_correlation(attention, prediction_difference, average_length=self.average_datapoint_length, p=0.5, class_name=class_name)
      _, k_af = self.attention_feature_erasure_correlation.calculate_kendall_top_k_non_zero_correlation(attention, prediction_difference, kIsNonZero=True, p=0.5, class_name=class_name)
      

      self.attention_gradient_correlation.calculate_kendall_tau_correlation(attention, gradients, class_name)
      self.attention_gradient_correlation.calculate_kendall_top_k_average_length_correlation(attention, gradients, average_length=self.average_datapoint_length, p=0.5, class_name=class_name)
      _, k_ag = self.attention_gradient_correlation.calculate_kendall_top_k_non_zero_correlation(attention, gradients, kIsNonZero=True, p=0.5, class_name=class_name)

      self.feature_erasure_gradient_correlation.calculate_kendall_tau_correlation(prediction_difference, gradients, class_name=class_name)
      self.feature_erasure_gradient_correlation.calculate_kendall_top_k_average_length_correlation(prediction_difference, gradients, average_length=self.average_datapoint_length, p=0.5, class_name=class_name)
      # Important: 'apples to apples' comparison: ensure the correlation calculation between the gradient and feature erasure measures
      # uses the same k value for the kendall_top_k calculation as was used for the previous correlation calculations
      if k_af != k_ag:
        self.logger.warning(f'kendall_top_k_non_zero mismatched k_value {k_af} != {k_ag}')
      _, k_fg = self.feature_erasure_gradient_correlation.calculate_kendall_top_k_non_zero_correlation(prediction_difference, gradients, k=k_af, kIsNonZero=False, class_name=class_name)
      if k_fg != k_af:
        self.logger.warning(f'Used different k {k_fg}')

  def generate_and_save_correlation_data_frames(self):
    csvs = []
    for correlation in self.correlations:
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
      correlations = [(correlation.id, correlation.get_total_correlation(correlation_measure)) for correlation in self.correlations]
      plotting.generate_correlation_density_plot(ax=axes,correlations=correlations)
      plot_title = f'{self.dataset_name}_{self.attention_type}_{self.attention_activation_function}_{correlation_measure.value}'
      plotting.annotate(ax=axes, title=plot_title)
      plotting.adjust_gridspec()
      plotting.save_axis_in_file(fig, axes, self.graph_path, plot_title)
