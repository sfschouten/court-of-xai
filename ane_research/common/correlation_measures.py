from scipy.stats import kendalltau
from overrides import overrides
from typing import Dict, List, Union
from allennlp.common import JsonDict, Registrable

from ane_research.common.kendall_top_k import kendall_top_k


class CorrelationMeasure(Registrable):

    def __init__(self, identifier: str, fields: List[str]):
        self._id = identifier
        self._fields = fields

    @property
    def id(self):
        return self._id

    @property
    def fields(self):
        return self._fields

    def correlation(self, a, b, **kwargs) -> JsonDict:
        # The correlation measure key must match the id
        raise NotImplementedError("Implement correlation calculation")


@CorrelationMeasure.register("kendall_tau")
class KendallTau(CorrelationMeasure):

    def __init__(self):
        fields = [
            "kendall_tau",
            "p_val"
        ]
        super().__init__(identifier="kendall_tau", fields=fields)

    @overrides
    def correlation(self, a, b, **kwargs) -> Dict[str, float]:
        correlation_kt, p_val = kendalltau(a, b)
        return {
            self.id: correlation_kt,
            "p_val": p_val
        }


@CorrelationMeasure.register("kendall_top_k")
class KendallTauTopK(CorrelationMeasure):

    def __init__(self, k: int = None):
        self.k = k

        k_name = str(k) if k else "length"
        id_ = f"kendall_top_k={k_name}"
        fields = [ id_, "k" ]

        super().__init__(identifier=id_, fields=fields)

    @overrides
    def correlation(self, a, b, **kwargs) -> Dict[str, Union[float, int]]:
        correlation_top_k, k = kendall_top_k(a=a, b=b, k=self.k)
        return {
            self.id: correlation_top_k,
            "k": k
        }


@CorrelationMeasure.register("kendall_top_k_average_length")
class KendallTauTopKAverageLength(CorrelationMeasure):

    def __init__(self, factor: float = 1):
        self.factor = factor
        id_ = f"kendall_top_k_{factor}_average_length"
        fields = [
            id_,
            "k_average_length"
        ]
        super().__init__(identifier=id_, fields=fields)

    @overrides
    def correlation(self, a, b, **kwargs) -> Dict[str, Union[float, int]]:
        average_length = kwargs['average_length']
        k = int(self.factor * average_length)
        correlation_top_k, k = kendall_top_k(a=a, b=b, k=k)
        return {
            self.id: correlation_top_k,
            "k_average_length": k
        }


@CorrelationMeasure.register("kendall_top_k_non_zero")
class KendallTauTopKNonZero(CorrelationMeasure):

    def __init__(self):
        fields = [
            "kendall_top_k_non_zero",
            "k_non_zero"
        ]
        super().__init__(identifier="kendall_top_k_non_zero", fields=fields)

    @overrides
    def correlation(self, a, b, **kwargs) -> Dict[str, Union[float, int]]:
        k = kwargs.get('k')
        kIsNonZero = (k==None)
        correlation_top_k, k = kendall_top_k(a=a, b=b, kIsNonZero=kIsNonZero, k=k)
        return {
            self.id: correlation_top_k,
            "k_non_zero": k
        }
