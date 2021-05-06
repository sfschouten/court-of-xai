import unittest

from xai_court.common.kendall_top_k import *


class TestStringMethods(unittest.TestCase):

    # D: discordant pairs
    # E: pairs that are ties in one permutation, but not the other
    # P: set of unordered pairs of distinct elements

    def test_opposite_order(self):
        x = [1, 2, 3, 4, 5, 6, 7]
        y = [7, 6, 5, 4, 3, 2, 1]
        for k in reversed(range(1, len(x) + 1)):
            actual_correlation, actual_k = kendall_top_k(x, y, k=k)
            self.assertEqual(actual_correlation, -1.0)
            self.assertEqual(actual_k, k)

    def test_same_order(self):
        x = [1, 2, 3, 4, 5, 6, 7]
        y = [1, 2, 3, 4, 5, 6, 7]
        for k in reversed(range(1, len(x) + 1)):
            actual_correlation, actual_k = kendall_top_k(x, y, k=k)
            self.assertEqual(actual_correlation, 1.0)
            self.assertEqual(actual_k, k)

    def test_with_ties_same_bucket_sizes(self):
        x = [1, 2, 3, 1, 1, 2, 2]
        y = [3, 2, 1, 2, 1, 2, 1]
        p = 0.5

        # There are only 3 unique buckets so the correlation will be identical for k = 3 -> 7
        # D = {(0, 1), (0, 2), (0, 5), (0, 6), (1, 2), (2, 3), (2, 5), (3, 6)}, thus |D| = 8
        # E = {(0, 3), (0, 4), (1, 3), (1, 6), (2, 4), (2, 6), (3, 4), (3, 5), (4, 6), (5, 6)}, thus |E| = 10
        # |P| = 7 choose 2 = 21
        expected_correlation = (((8 + 10 * p) / 21) * -2) + 1
        for k in [3, 4, 5, 6, 7]:
            actual_correlation, actual_k = kendall_top_k(x, y, k=k)
            self.assertAlmostEqual(actual_correlation, expected_correlation)
            self.assertEqual(actual_k, k)

        # x top k indexes -> [1, 2, 5, 6]
        # y top k indexes -> [0, 1, 3, 5]
        # D = {(0, 1), (0, 2), (0, 5), (0, 6), (1, 2), (2, 3), (2, 5), (3, 6)}, thus |D| = 8
        # E = {(1, 3), (1, 6), (3, 5), (5, 6)}, thus |E| = 4
        # P = {(0, 1), (0, 2), (0, 5), (0, 6), (1, 2), (1, 3), (1, 5), (1, 6), (2, 3), (2, 5), (3, 5), (3, 6),
        # (5, 6)}, thus |P| = 13
        k = 2
        expected_correlation = (((8 + 4 * p) / 13) * -2) + 1
        actual_correlation, actual_k = kendall_top_k(x, y, k=k)
        self.assertAlmostEqual(actual_correlation, expected_correlation)
        self.assertEqual(actual_k, k)

        k = 1
        expected_correlation = -1.0
        actual_correlation, actual_k = kendall_top_k(x, y, k=k)
        self.assertAlmostEqual(actual_correlation, expected_correlation)
        self.assertEqual(actual_k, k)

    def test_with_ties_different_bucket_sizes_larger_p(self):
        x = [1, 2, 3, 1, 2, 2, 2]
        y = [3, 2, 1, 2, 1, 2, 1]
        p = 1.0

        # There are only 3 unique buckets so the correlation will be identical
        # D = {(0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (1, 2), (2, 3), (2, 5), (3, 4), (3, 6)}, thus |D| = 10
        # E = {(0, 3), (1, 3), (1, 4), (1, 6), (2, 4), (2, 6), (3, 5), (4, 5), (5, 6)}, thus |E| = 9
        # |P| = 7 choose 2 = 21
        expected_correlation = (((10 + 9 * p) / 21) * -2) + 1
        for k in [3, 4, 5, 6, 7]:
            actual_correlation, actual_k = kendall_top_k(x, y, k=k, p=p)
            self.assertAlmostEqual(actual_correlation, expected_correlation)
            self.assertEqual(actual_k, k)

        # x top k indexes -> [1, 2, 4, 5, 6]
        # y top k indexes -> [0, 1, 3, 5]
        # D = {(0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (1, 2), (2, 3), (2, 5), (3, 4), (3, 6)} thus |D| = 10
        # E = {(1, 3), (1, 4), (1, 6), (3, 5), (4, 5), (5, 6)}, thus |E| = 6
        # P = {(0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 5),
        # (3, 4), (3, 5), (3, 6), (4, 5), (5, 6)}, thus |P| = 17
        k = 2
        expected_correlation = (((10 + 6 * p) / 17) * -2) + 1
        actual_correlation, actual_k = kendall_top_k(x, y, k=k, p=p)
        self.assertAlmostEqual(actual_correlation, expected_correlation)
        self.assertEqual(actual_k, k)

        k = 1
        expected_correlation = -1.0
        actual_correlation, actual_k = kendall_top_k(x, y, k=k, p=p)
        self.assertAlmostEqual(actual_correlation, expected_correlation)
        self.assertEqual(actual_k, k)
