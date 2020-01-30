from scipy.stats import kendalltau
from typing import List
from enum import Enum
from ane_research.utils.kendall_top_k import kendall_top_k
import pandas as pd
import numpy as np

class CorrelationMeasures(Enum):
  KENDALL_TAU = 'kendall_tau'
  KENDALL_TOP_K_NON_ZERO = 'kendall_top_k_non_zero'
  KENDALL_TOP_K_AVG_LEN = 'kendall_top_k_average_length'

class Correlation():

  def __init__(self, feature1:str, feature2: str, class_names: List[str]):
    self.feature1 = feature1
    self.feature2 = feature2
    self.id = feature1 + '_' + feature2
    self.class_names = class_names
    self.data_store = {}
    self._initialize_data_store()

  def _initialize_data_store(self):
    self.data_store = {}
    for correlation_measure in CorrelationMeasures:
      self.data_store.setdefault(correlation_measure.value, {})
      for class_name in self.class_names:
        self.data_store[correlation_measure.value][class_name] = []
      self.data_store[correlation_measure.value]['all_classes'] = []

  def clear_data_store(self):
    self._initialize_data_store()

  def calculate_kendall_tau_correlation(self, feature1values, feature2values, class_name: str = None):
    correlation = kendalltau(feature1values, feature2values)
    if class_name is not None:
      self.data_store[CorrelationMeasures.KENDALL_TAU.value][class_name].append(correlation)
    self.data_store[CorrelationMeasures.KENDALL_TAU.value]['all_classes'].append(correlation[0])
    return correlation

  def calculate_kendall_top_k_non_zero_correlation(self, feature1values, feature2values, k:int = None, kIsNonZero: bool = False,
                                          p:int = 0.5 , class_name: str = None):
    correlation, k = kendall_top_k(feature1values, feature2values, k, kIsNonZero, p)
    if class_name is not None:
      self.data_store[CorrelationMeasures.KENDALL_TOP_K_NON_ZERO.value][class_name].append(correlation)
    self.data_store[CorrelationMeasures.KENDALL_TOP_K_NON_ZERO.value]['all_classes'].append(correlation)
    return (correlation, k)

  def calculate_kendall_top_k_average_length_correlation(self, feature1values, feature2values, average_length:int = None, kIsNonZero: bool = False,
                                          p:int = 0.5 , class_name: str = None):
    correlation, k = kendall_top_k(feature1values, feature2values, k=average_length, kIsNonZero=kIsNonZero, p=p)
    if class_name is not None:
      self.data_store[CorrelationMeasures.KENDALL_TOP_K_AVG_LEN.value][class_name].append(correlation)
    self.data_store[CorrelationMeasures.KENDALL_TOP_K_AVG_LEN.value]['all_classes'].append(correlation)
    return (correlation, k)

  def get_total_correlation(self, correlation_measure: CorrelationMeasures):
    to_return = np.array(self.data_store[correlation_measure.value]['all_classes'])
    to_return = to_return[~np.isnan(to_return)]
    return to_return

  def generate_kendall_tau_data_frame(self) -> pd.DataFrame:
    measures = {'pval_sig' : {}, 'mean' : {}, 'std' : {}}
    total_p_values = []
    for class_name in self.class_names:
      correlation_values = []
      p_values = []
      for tup in self.data_store[CorrelationMeasures.KENDALL_TAU.value][class_name]:
        correlation_value, p_value = tup
        correlation_values.append(correlation_value)
        p_values.append(p_value)
      total_p_values += p_values
      p_values = np.array(p_values)
      pval_sig = (p_values <= 0.05).sum() / len(p_values)
      measures['pval_sig'][class_name] = pval_sig
      measures['mean'][class_name] = np.nanmean(correlation_values)
      measures['std'][class_name] = np.nanstd(correlation_values)
    total_p_values =  np.array(total_p_values)
    measures['pval_sig'] = (total_p_values <= 0.05).sum() / len(total_p_values)
    measures['mean']['all_classes'] = np.nanmean(self.data_store[CorrelationMeasures.KENDALL_TAU.value]['all_classes'])
    measures['std']['all_classes'] = np.nanstd(self.data_store[CorrelationMeasures.KENDALL_TAU.value]['all_classes'])
    return pd.DataFrame(measures)

  def generate_kendall_top_k_data_frame(self, correlation_measure: CorrelationMeasures) -> pd.DataFrame:
    measures = {'mean': {}, 'std': {}}
    for class_name in self.class_names:
      correlation_values = [val for val in self.data_store[correlation_measure.value][class_name]]
      measures['mean'][class_name] = np.nanmean(correlation_values)
      measures['std'][class_name] = np.nanstd(correlation_values)
    measures['mean']['all_classes'] = np.nanmean(self.data_store[correlation_measure.value]['all_classes'])
    measures['std']['all_classes'] = np.nanstd(self.data_store[correlation_measure.value]['all_classes'])
    return pd.DataFrame(measures)
