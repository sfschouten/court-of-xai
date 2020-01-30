"""Top-k kendall-tau distance.

This module generalise kendall-tau as defined in [1].
It returns a distance: 0 for identical (in the sense of top-k) lists and 1 if completely different.

Example:
    Simply call kendall_top_k with two same-length arrays of ratings (or also rankings), length of the top elements k (default is the maximum length possible), and p (default is 0, see [1]) as parameters:

        import kendall
        a = np.array([1,2,3,4,5])
        b = np.array([5,4,3,2,1])
        kendall.kendall_top_k(a,b,k=4)

Author: Alessandro Checco
    https://github.com/AlessandroChecco
References
[1] Fagin, Ronald, Ravi Kumar, and D. Sivakumar. "Comparing top k lists." SIAM Journal on Discrete Mathematics 17.1 (2003): 134-160.
"""
import math
import numpy as np
import scipy.stats as stats
import scipy.special as special

def kendall_top_k(a, b, k=None, kIsNonZero=False, p=0.5): 
    """
    kendall_top_k(np.array,np.array,k,p)
    This function generalise kendall-tau as defined in 
        [1] Fagin, Ronald, Ravi Kumar, and D. Sivakumar. "Comparing top k lists." SIAM Journal on Discrete Mathematics 17.1 (2003): 134-160.
    It returns a distance: 1 for identical (in the sense of top-k) lists and -1 if completely different.

    Example:
        Simply call it with two same-length arrays of ratings (or also rankings), 
        length of the top elements k (default is the maximum length possible), and p (default is 0, see [1]) as parameters:

            $ a = np.array([1,2,3,4,5])
            $ b = np.array([5,4,3,2,1])
            $ kendall_top_k(a,b,k=4)

    If the kIsNonZero option is True, k is set to the amount of non-zero values in a or b, depending on which has least.
    """

    a = np.array(a)
    b = np.array(b)

    if kIsNonZero:
        anz, bnz = np.count_nonzero(a), np.count_nonzero(b)
        k = min(np.count_nonzero(a), np.count_nonzero(b))
        #print("anz={}, bnz={}, k={}".format(anz, bnz, k))
    elif k is None:
        k = a.size

    if a.size != b.size:
        raise NameError('The two arrays need to have same lengths')

    k = min(k,a.size)
    a_top_k = np.argpartition(a,-k)[-k:]
    b_top_k = np.argpartition(b,-k)[-k:]
    common_items = np.intersect1d(a_top_k,b_top_k)
    only_in_a = np.setdiff1d(a_top_k, common_items)
    only_in_b = np.setdiff1d(b_top_k, common_items)

    # case 1
    kendall = (1 - (stats.kendalltau(a[common_items], b[common_items])[0] / 2 + 0.5)) * common_items.size**2
    #print(common_items)
    #print(kendall)

    if np.isnan(kendall): # degenerate case with only one item (not defined by Kendall)
        #print("DEGENERATE CASE <= 1 in common")
        kendall = 0

    #case 2 (& 3 ?)
    test = 0
    for i in common_items: 
        for j in only_in_a:
            if a[i] < a[j]:
                test += 1
        for j in only_in_b:
            if b[i] < b[j]:
                test += 1

    kendall += test

    # test
    #print(only_in_a)
    #print(only_in_b)
    #test2 = (k-common_items.size)*(k+common_items.size+1) - np.sum(only_in_a+1) - np.sum(only_in_b+1)
    #print(test2)
    #kendall += test2

    # case 4
    kendall += 2 * p * special.binom(k-common_items.size, 2)

    # case 3?
    #test3 = (k - common_items.size)**2
    #kendall += test3
    #print(kendall)
    kendall /= (only_in_a.size + only_in_b.size + common_items.size)**2  #normalization
    kendall = -2 * kendall + 1 # change to correct range

    return (kendall, k)
