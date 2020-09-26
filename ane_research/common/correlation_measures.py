"""
Measures to calculate the correlation between the top-k elements of two identical length lists of scores
"""

from dataclasses import dataclass, field
import math
import numbers
from typing import Dict, List, Optional, Union

from allennlp.common import Registrable
import numpy as np
from overrides import overrides
from scipy.stats import kendalltau, spearmanr, pearsonr

from ane_research.common.kendall_top_k import kendall_top_k


def enforce_same_shape(func):
    def wrapper_enforce_same_shape(*args, **kwargs):
        rel_args = args[1:] # skip self
        all_same_shape = all(_arg.shape == rel_args[0].shape for _arg in rel_args)
        if not all_same_shape:
            raise ValueError(f"All arguments must have the same shape")
        return func(*args, **kwargs)
    return wrapper_enforce_same_shape


@dataclass
class CorrelationResult:
    correlation: float
    k: int = field(default=None)


CorrelationMap = Dict[str, CorrelationResult]


class CorrelationMeasure(Registrable):
    """
    A uniquely identifiable measure to calculate correlation(s) between the top-k elements of two identical
    length lists of scores

    Args:
        identifier (str):
            Unique name of the measure
        unfair_in_isolation (Optional[bool]):
            This metric uses a dynamic value for k which may produce correlations that cannot be compared with each
            other directly. If so, the caller is responsible for passing in a 'fair' override for k when required.

            For example: sparse attention distributions may produce scores of zero but feature importance measures
            do not. Thus, if we are calculating the KendallTauTopKNonZero metric and want an "apples to apples"
            comparison we must ensure the correlation calculation between two feature importance measures uses the
            average k value from correlation calculations with at least one attention interpreter.

            Defaults to False.
    """
    def __init__(self, identifier: str, unfair_in_isolation: Optional[bool] = False): 
        self._id = identifier
        self.unfair_in_isolation = unfair_in_isolation

    @property
    def id(self):
        return self._id

    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        """
        Calculates a number of uniquely identifiable correlations between the top-k elements of two
        identical length lists of scores. For example, if the measure calculates the correlation between the
        top 10% of scores in a and b and the top 20% of scores the return value would look something like:

        {
            '{id}_top_10_percent': ...
            '{id}_top_20_percent': ...
        }

        Args:
            a (np.ndarray): List of scores. Same length as b.
            b (np.ndarray): List of scores. Same length as a.

        Returns:
            CorrelationMap: Mapping of some identifier to a CorrelationResult
        """
        raise NotImplementedError("Implement correlation calculation")


@CorrelationMeasure.register("kendall_tau")
class KendallTau(CorrelationMeasure):

    def __init__(self):
        super().__init__(identifier="kendall_tau")

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        kt, _ = kendalltau(a, b)
        return {
            self.id: CorrelationResult(correlation=kt, k=len(a))
        }


@CorrelationMeasure.register("spearman_rho")
class SpearmanRho(CorrelationMeasure):

    def __init__(self):
        super().__init__(identifier="spearman_rho")

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        sr, _ = spearmanr(a, b)
        return {
            self.id: CorrelationResult(correlation=sr, k=len(a))
        }


@CorrelationMeasure.register("pearson_r")
class PearsonR(CorrelationMeasure):

    def __init__(self):
        super().__init__(identifier="pearson_r")

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        pr, _ = pearsonr(a, b)
        return {
            self.id: CorrelationResult(correlation=pr, k=len(a))
        }


@CorrelationMeasure.register("kendall_top_k_variable")
class KendallTauTopKVariable(CorrelationMeasure):

    def __init__(self, percent_top_k: List[float]):
        super().__init__(identifier="kendall_top_k_variable")
        self.variable_lengths = percent_top_k

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        results = {}
        for variable_length in self.variable_lengths:
            k = max(1, math.floor(len(a) * variable_length))
            kt_top_k, k = kendall_top_k(a=a, b=b, k=k)
            results[f"{self.id}_{variable_length}"] = CorrelationResult(correlation=kt_top_k, k=k)
        return results


@CorrelationMeasure.register("kendall_top_k_fixed")
class KendallTauTopKFixed(CorrelationMeasure):

    def __init__(self, fixed_top_k: List[int]):
        super().__init__(identifier="kendall_top_k_fixed")
        self.fixed_lengths = fixed_top_k

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        results = {}
        for fixed_length in self.fixed_lengths:
            kt_top_k, k = kendall_top_k(a=a, b=b, k=fixed_length)
            results[f"{self.id}_{fixed_length}"] = CorrelationResult(correlation=kt_top_k, k=k)
        return results


@CorrelationMeasure.register("kendall_top_k_non_zero")
class KendallTauTopKNonZero(CorrelationMeasure):

    def __init__(self):
        super().__init__(identifier="kendall_top_k_non_zero", unfair_in_isolation=True)

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:

        # k may be explicitly specified in some cases to ensure fair comparisons
        k = kwargs.get("k")
        print(f'got k: {k}')
        kIsNonZero = (k==None)

        kt_top_k, k = kendall_top_k(a=a, b=b, kIsNonZero=kIsNonZero, k=k)
        return {
            self.id: CorrelationResult(correlation=kt_top_k, k=k)
        }
