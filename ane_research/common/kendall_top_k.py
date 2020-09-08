"""
Top-k kendall-tau correlation metric for ranked lists. The value of k can be provided or, optionally, set to the minimum
number of non-zero elements in either list. Returns a scalar in the range [-1, 1] where 1 means both rankings are
identical and -1 means the rankings are opposites.

References
[1] Fagin, Ronald, Ravi Kumar, and D. Sivakumar. 'Comparing top k lists.' SIAM Journal on Discrete Mathematics 17.1 (2003): 134-160.
"""

# pylint: disable=E1101
# pylint incorrectly identifies some types as tuples

import math
import numpy as np
import numbers
import scipy.stats as stats
import scipy.special as special
from typing import Any, Tuple

def kendall_top_k(a: Any, b: Any, k: int = None, kIsNonZero: bool = False, p: float = 0.5) -> Tuple[float, int]:
    """Compute the top-k kendall-tau correlation metric for the given ranked lists.

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
            The penalty parameter in the range [0, 1]. Defaults to the neutral case, p = 0.5

    Raises:
        ValueError: If p is not defined as described or if the lists are not equal in length

    Returns:
        Tuple[float, int]: A tuple of the computed correlation and the value used for k
    """
    if not (isinstance(p, numbers.Real) and p >= 0 and p <= 1):
        raise ValueError("The penalty parameter p must be numeric and in the range [0,1]")

    x = np.array(a).flatten()
    y = np.array(b).flatten()

    if x.size != y.size:
        raise ValueError("The ranked lists must have same lengths")

    if kIsNonZero:
        k = min(np.count_nonzero(x), np.count_nonzero(y))
    elif k is None:
        k = x.size

    k = min(k, x.size)

    # indices of the top k arguments e.g., [1, 2, 3, 4] with k = 3 --> [1, 2 ,3]
    x_top_k = np.argpartition(x, -k)[-k:]
    # ranks of all arguments (projection from a list onto the domain [1...n]) e.g., [55, 42, 89, 100] --> [3, 4, 2, 1]
    x_ranks = np.full(x.size, x.size + 1) - stats.rankdata(x)

    y_top_k = np.argpartition(y, -k)[-k:]
    y_ranks = np.full(y.size, y.size + 1) - stats.rankdata(y)

    # Using the explicit notation of Fagin et al. with references to their equation numbers
    Z = np.intersect1d(x_top_k, y_top_k)
    S = np.setdiff1d(x_top_k, y_top_k)
    T = np.setdiff1d(y_top_k, x_top_k)
    z = Z.size

    # Equation 1: i and j appear in both top k lists. Penalize per the number of shared pairs that are discordant
    # Code partially taken from scipy.stats.kendalltau
    rx, ry = x[Z], y[Z]
    eqn1 = np.sum([((ry[i + 1:] < ry[i]) * (rx[i + 1:] > rx[i])).sum() for i in range(len(ry) - 1)], dtype=float)

    # Equation 2: i and j both appear in one top k list, and exactly one of i or j appears in the other
    eqn2 = (k - z) * (k + z + 1) - sum(x_ranks[S]) - sum(y_ranks[T])

    # Equation 3: i, but not j, appears in one top k list and j, but not i, appears in the other
    eqn3 = (k - z) ** 2

    # Equation 4: i and j both appear in one top k list, but neither i nor j appears in the other
    eqn4 = 2 * p * special.comb(k - z, 2)

    kendall_distance_with_penalty = eqn1 + eqn2 + eqn3 + eqn4

    # Normalize the distance to a correlation in the range [-1, 1]
    correlation = kendall_distance_with_penalty / special.comb(S.size + T.size + z, 2)
    correlation *= -2
    correlation += 1

    return (correlation, k)
