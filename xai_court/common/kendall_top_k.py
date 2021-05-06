"""
Top-k kendall-tau correlation metric for full (no ties) or partial (with ties) ranked lists.
The value of k can be provided or, optionally, set to the minimum number of non-zero elements in either list.

Returns a scalar in the range [-1, 1] where 1 means both rankings are identical and -1 means the rankings
are opposites.

Reference
--------
Ronald Fagin, Ravi Kumar, Mohammad Mahdian, D. Sivakumar, and Erik Vee. 2004. Comparing and aggregating rankings with ties.
In Proceedings of the twenty-third ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems (PODS '04).
Association for Computing Machinery, New York, NY, USA, 47–58. DOI:https://doi.org/10.1145/1055558.1055568
"""
from collections import Counter
import itertools
import math
import numbers
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import scipy.stats as stats


def unordered_cartesian_product(x: List[Any], y: List[Any]) -> Iterator[List[Tuple[Any, Any]]]:
    """Yield all unique unordered pairs from the Cartesian product of x and y"""
    seen = set()
    for (i, j) in itertools.product(x, y):
        if i == j or (i, j) in seen:
            continue
        seen.add((i, j))
        seen.add((j, i))

        yield (i, j)

def bucket_order(x: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Associate a top-k full (no ties) or partial ranking (with ties) with a bucket order per Fagin et al. (2004)

    Args:
        x (np.ndarray): A full or partial ranked list
        k (int): Only the indices elements of x in the top k buckets are returned. Defaults to the size of x.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Full bucket ranking, Indices of the elements of x in the top-k buckets
    """
    if k is None:
        k = x.size

    x_ranked = stats.rankdata(x, method='dense')
    rank_count = Counter(x_ranked)
    unique_ranks = sorted(list(set(x_ranked)), reverse=True)

    # The position of bucket B_i is the average location within bucket B_i
    bucket_sizes = [rank_count[rank] for rank in unique_ranks]
    bucket_positions = []
    def get_bucket_position(bucket_sizes, bucket_index):
        return sum(bucket_sizes[:bucket_index]) + (bucket_sizes[bucket_index] + 1) / 2
    bucket_positions = {rank: get_bucket_position(bucket_sizes, i) for i, rank in zip(range(len(bucket_sizes)), unique_ranks)}

    bucket_order = np.array([bucket_positions[i] for i in x_ranked])

    top_k_bucket_positions = sorted(bucket_positions.values())[:k]
    top_k = [i for i, bp in enumerate(x_ranked) if bucket_positions[bp] in top_k_bucket_positions]

    return bucket_order, top_k


def kendall_top_k(a: Any, b: Any, k: int = None, kIsNonZero: bool = False, p: float = 0.5) -> Tuple[float, int]:
    """
    Compute the top-k kendall-tau correlation metric for the given full (no ties) or partial (with ties) ranked lists

    Args:
        a (ArrayLike):
            The first ranked list. Can be any array-like type (e.g. list, numpy array or cpu-bound tensor)
        b (ArrayLike):
            The second ranked list. Can be any array-like type (e.g. list, numpy array or cpu-bound tensor)
        k (int, optional):
            Only the top "k" elements are compared. Defaults to the size of the first list
        kIsNonZero (bool, optional):
            If specified, overrides k to be the minimum number of non-zero elements in either list. Defaults to False
        p (float, optional):
            The penalty parameter in the range (0, 1]. This is a metric if p is in the range [1/2, 1] and a near metric
            if p is in the range (0, 1/2). Defaults to the neutral case, p = 1/2

    Raises:
        ValueError: If p is not defined as described or if the lists are not equal in length

    Returns:
        Tuple[float, int]: A tuple of the computed correlation and the value used for k
    """
    if not (isinstance(p, numbers.Real) and p > 0 and p <= 1):
        raise ValueError("The penalty parameter p must be numeric and in the range (0,1]")

    x = np.array(a).ravel()
    y = np.array(b).ravel()

    if x.size != y.size:
        raise ValueError("The ranked lists must have same lengths")

    if kIsNonZero:
        k = min(np.count_nonzero(x), np.count_nonzero(y))
    elif k is None:
        k = x.size

    k = min(k, x.size)

    x_bucket_order, x_top_k = bucket_order(x, k)
    y_bucket_order, y_top_k = bucket_order(y, k)

    kendall_distance = 0
    normalization_constant = 0

    for i, j in unordered_cartesian_product(x_top_k, y_top_k):

        normalization_constant += 1

        i_bucket_x = x_bucket_order[i]
        j_bucket_x = x_bucket_order[j]
        i_bucket_y = y_bucket_order[i]
        j_bucket_y = y_bucket_order[j]

        # Case 1: i and j are in different buckets in both x and y: penalty = 1
        if i_bucket_x != j_bucket_x and i_bucket_y != j_bucket_y:
            opposite_order_x = i_bucket_x > j_bucket_x and i_bucket_y < j_bucket_y
            opposite_order_y = i_bucket_x < j_bucket_x and i_bucket_y > j_bucket_y
            if opposite_order_x or opposite_order_y:
                kendall_distance += 1

        # Case 2: i and j are in the same bucket in both x and y: penalty = 0 (so we can ignore)

        # Case 3: i and j are in the same bucket in one of the partial rankings, but in different buckets in the other
        # penalty = p
        elif (i_bucket_x == j_bucket_x and i_bucket_y != j_bucket_y) or (i_bucket_y == j_bucket_y and i_bucket_x != j_bucket_x):
            kendall_distance += p


    # Normalize to range [-1, 1]
    correlation = kendall_distance / max(1, normalization_constant)
    correlation *= -2
    correlation += 1

    return (correlation, k)
