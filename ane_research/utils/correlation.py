from enum import Enum
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ane_research.utils.kendall_top_k import kendall_top_k


class CorrelationMeasures(Enum):
  '''The three correlation measures used in the paper'''
  KENDALL_TAU = 'kendall_tau'
  KENDALL_TOP_K_NON_ZERO = 'kendall_top_k_non_zero'
  KENDALL_TOP_K_AVG_LEN = 'kendall_top_k_average_length'

class Correlation():
  '''Data store class to hold the correlation values between two features 

    Support correlation values by predicted class if desired
  '''

  def __init__(self, feature1: str, feature2: str, class_names: List[str]) -> None:
    self.feature1 = feature1
    self.feature2 = feature2
    self.id = feature1 + '_' + feature2
    self.class_names = class_names
    self.data_store = {}
    self._initialize_data_store()

  def _initialize_data_store(self) -> None:
    self.data_store = {}
    for correlation_measure in CorrelationMeasures:
      self.data_store.setdefault(correlation_measure.value, {})
      for class_name in self.class_names:
        self.data_store[correlation_measure.value][class_name] = []
      self.data_store[correlation_measure.value]['all_classes'] = []

  def clear_data_store(self) -> None:
    '''Erase data held in data store'''
    self._initialize_data_store()

  def calculate_kendall_tau_correlation(self, feature1values: List[Any], feature2values: List[Any], 
                                        class_name: str = None) -> List[Tuple[float, float]]:
    '''Calculate Kendall tau correlation between two features

    Args:
      feature1values (List[Any]): The values of the first feature
      feature2values (List[Any]): The values of the second feature
      class_name (str, optional): Correlations can be stored by class if desired. Defaults to None.

    Returns:
      List[Tuple[float, float]]: List of (correlation value, p value) tuples
    '''
    correlation = kendalltau(feature1values, feature2values)
    if class_name is not None:
      self.data_store[CorrelationMeasures.KENDALL_TAU.value][class_name].append(correlation)
    # no need to store the p_values for the combination of all classes, we can aggregate later
    self.data_store[CorrelationMeasures.KENDALL_TAU.value]['all_classes'].append(correlation[0])
    return correlation

  def calculate_kendall_top_k_non_zero_correlation(self, feature1values: List[Any], feature2values: List[Any], k:int = None, kIsNonZero: bool = True,
                                          p:float = 0.5 , class_name: str = None) -> Tuple[List[Any], int]:
    '''Calculate Kendall top-k correlation between two features where k is the LEAST amount of non-zero values in the features

    Args:
      feature1values (List[Any]): The values of the first feature
      feature2values (List[Any]): The values of the second feature
      k (int): Specify the k value if needed. This will override kIsNonZero.
      kIsNonZero (bool): Set k to the number of nonzero elements. This may need to be explicitly overridden for some comparisons. Defaults to True.
      p (float): Parameter to influence whether a negative-neutral-positive assumption is made about the correlation of the part of the ranking that isn't in the top-k
      class_name (str, optional): Correlations can be stored by class if desired. Defaults to None.

    Returns:
      Tuple[List[Any], int]: Correlation values and the k value used, since it may have been adjusted in the lists are not long enough
    '''
    correlation, k = kendall_top_k(feature1values, feature2values, k, kIsNonZero, p)
    if class_name is not None:
      self.data_store[CorrelationMeasures.KENDALL_TOP_K_NON_ZERO.value][class_name].append(correlation)
    self.data_store[CorrelationMeasures.KENDALL_TOP_K_NON_ZERO.value]['all_classes'].append(correlation)
    return (correlation, k)

  def calculate_kendall_top_k_average_length_correlation(self, feature1values: List[Any], feature2values: List[Any],
                                                         average_length:int, p:float = 0.5, class_name: str = None) -> Tuple[List[Any], int]:
    '''Calculate Kendall top-k correlation between two features where k is the average length of a datapoint in your dataset

    Args:
      feature1values (List[Any]): The values of the first feature
      feature2values (List[Any]): The values of the second feature
      average_length (int): Average length of a datapoint in your dataset
      p (float): Parameter to influence whether a negative-neutral-positive assumption is made about the correlation of the part of the ranking that isn't in the top-k
      class_name (str, optional): Correlations can be stored by class if desired. Defaults to None.

    Returns:
      Tuple[List[Any], int]: Correlation values and the k value used, since it may have been adjusted in the lists are not long enough
    '''
    correlation, k = kendall_top_k(feature1values, feature2values, k=average_length, kIsNonZero=False, p=p)
    if class_name is not None:
      self.data_store[CorrelationMeasures.KENDALL_TOP_K_AVG_LEN.value][class_name].append(correlation)
    self.data_store[CorrelationMeasures.KENDALL_TOP_K_AVG_LEN.value]['all_classes'].append(correlation)
    return (correlation, k)

  def get_total_correlation(self, correlation_measure: CorrelationMeasures) -> np.array:

    to_return = np.array(self.data_store[correlation_measure.value]['all_classes'])
    to_return = to_return[~np.isnan(to_return)]
    return to_return

  def generate_kendall_tau_data_frame(self) -> pd.DataFrame:
    '''Fetch kendall tau statistics as a DataFrame

    Returns:
      pd.DataFrame: [description]
    '''
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
    '''Fetch kendall top-k statistics as a DataFrame

    Returns:
      pd.DataFrame: [description]
    '''
    measures = {'mean': {}, 'std': {}}
    for class_name in self.class_names:
      correlation_values = [val for val in self.data_store[correlation_measure.value][class_name]]
      measures['mean'][class_name] = np.nanmean(correlation_values)
      measures['std'][class_name] = np.nanstd(correlation_values)
    measures['mean']['all_classes'] = np.nanmean(self.data_store[correlation_measure.value]['all_classes'])
    measures['std']['all_classes'] = np.nanstd(self.data_store[correlation_measure.value]['all_classes'])
    return pd.DataFrame(measures)
