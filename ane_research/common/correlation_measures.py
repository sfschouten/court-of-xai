"""
Measures to calculate the correlation between the top-k elements of two identical length lists of scores
"""

from dataclasses import dataclass, field
import math
import numbers
from typing import Dict, List, Union

from allennlp.common import Registrable
import numpy as np
from overrides import overrides
from scipy.stats import kendalltau, spearmanr, pearsonr

from ane_research.common.kendall_top_k import kendall_top_k


def enforce_same_length(func):
    def wrapper_enforce_same_length(*args, **kwargs):
        lengths = list(map(len, args))
        all_same_length = all(_arg == args[0] for _arg in args)
        if not all_same_length:
            raise ValueError(f"All arguments must have the same length. Received lengths of: {lengths}")
    return wrapper_enforce_same_length


@dataclass
class CorrelationResult:
    correlation: float
    k: int = field(default=None)


ScoredList = Union[np.ndarray, List[numbers.Real]]
CorrelationMap = Dict[str, CorrelationResult]


class CorrelationMeasure(Registrable):
    """
    A uniquely identifiable measure to calculate correlation(s) between the top-k elements of two identical
    length lists of scores
    """
    def __init__(self, identifier: str):
        self._id = identifier

    @property
    def id(self):
        return self._id

    def correlation(self, a: ScoredList, b: ScoredList, **kwargs) -> CorrelationMap:
        """
        Calculates a variable number of uniquely identifiable correlations between the top-k elements of two
        identical length lists of scores. For example, if the measure calculates the correlation between the
        top 10% of scores in a and b and the top 20% of scores the return value would look something like:

        {
            '{id}_top_10_percent': ...
            '{id}_top_20_percent': ...
        }

        Args:
            a (ScoredList): List of scores. Same length as b.
            b (ScoredList): List of scores. Same length as a.

        Returns:
            CorrelationMap: Mapping of some identifier to a CorrelationResult
        """
        raise NotImplementedError("Implement correlation calculation")


@CorrelationMeasure.register("kendall_tau")
class KendallTau(CorrelationMeasure):

    def __init__(self):
        super().__init__(identifier="kendall_tau")

    @overrides
    @enforce_same_length
    def correlation(self, a: ScoredList, b: ScoredList, **kwargs) -> CorrelationMap:
        kt, _ = kendalltau(a, b)
        return {
            self.id: CorrelationResult(correlation=kt, k=len(a))
        }


@CorrelationMeasure.register("spearman_rho")
class SpearmanRho(CorrelationMeasure):

    def __init__(self):
        super().__init__(identifier="spearman_rho")

    @overrides
    @enforce_same_length
    def correlation(self, a: ScoredList, b: ScoredList, **kwargs) -> CorrelationMap:
        sr, _ = spearmanr(a, b)
        return {
            self.id: CorrelationResult(correlation=sr, k=len(a))
        }


@CorrelationMeasure.register("pearson_r")
class PearsonR(CorrelationMeasure):

    def __init__(self):
        super().__init__(identifier="pearson_r")

    @overrides
    @enforce_same_length
    def correlation(self, a: ScoredList, b: ScoredList, **kwargs) -> CorrelationMap:
        pr, _ = pearsonr(a, b)
        return {
            self.id: CorrelationResult(correlation=pr, k=len(a))
        }


@CorrelationMeasure.register("kendall_top_k_variable")
class KendallTauTopKVariable(CorrelationMeasure):

    def __init__(self, variable_lengths: List[float]):
        super().__init__(identifier="kendall_top_k_variable")
        self.variable_lengths = variable_lengths

    @overrides
    @enforce_same_length
    def correlation(self, a: ScoredList, b: ScoredList, **kwargs) -> CorrelationMap:
        results = {}
        for variable_length in self.variable_lengths:
            k = min(1, math.floor(len(a) * variable_length))
            kt_top_k, k = kendall_top_k(a=a, b=b, k=k)
            results[f"{self.id}_{variable_length}"] = CorrelationResult(correlation=kt_top_k, k=k)
        return results


@CorrelationMeasure.register("kendall_top_k_fixed")
class KendallTauTopKFixed(CorrelationMeasure):

    def __init__(self, fixed_lengths: List[float]):
        super().__init__(identifier="kendall_top_k_fixed")
        self.fixed_lengths = fixed_lengths

    @overrides
    @enforce_same_length
    def correlation(self, a: ScoredList, b: ScoredList, **kwargs) -> CorrelationMap:
        results = []
        for fixed_length in self.fixed_lengths:
            kt_top_k, k = kendall_top_k(a=a, b=b, k=fixed_length)
            results[f"{self.id}_{fixed_length}"] = CorrelationResult(correlation=kt_top_k, k=k)
        return results


@CorrelationMeasure.register("kendall_top_k_non_zero")
class KendallTauTopKNonZero(CorrelationMeasure):

    def __init__(self):
        super().__init__(identifier="kendall_top_k_non_zero")

    @overrides
    @enforce_same_length
    def correlation(self, a: ScoredList, b: ScoredList, **kwargs) -> CorrelationMap:

        # k may be explicitly specified in some cases to ensure fair comparisons
        k = kwargs.get('k')
        kIsNonZero = (k==None)

        kt_top_k, k = kendall_top_k(a=a, b=b, kIsNonZero=kIsNonZero, k=k)
        return {
            self.id: CorrelationResult(correlation=kt_top_k, k=k)
        }
