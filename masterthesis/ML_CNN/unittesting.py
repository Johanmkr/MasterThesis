import numpy as np
import torch
import unittest

import train_utils as tutils

"""
def return1():
    return 1

class MyTestCase(unittest.TestCase):
    def test_return1(self):
        result = return1()
        self.assertEqual(result, 1)
"""


class TestMetrics(unittest.TestCase):
    def test_all_ones(self):
        N = 20
        y_true = torch.ones(N)
        y_pred = torch.ones(N)
        TP, TN, FP, FN = tutils.confusion_metrics(y_pred, y_true)
        self.assertEqual(TP, N)
        self.assertEqual(TN, 0)
        self.assertEqual(FP, 0)
        self.assertEqual(FN, 0)

    def test_all_zeros(self):
        N = 20
        y_true = torch.zeros(N)
        y_pred = torch.zeros(N)
        TP, TN, FP, FN = tutils.confusion_metrics(y_pred, y_true)
        self.assertEqual(TP, 0)
        self.assertEqual(TN, N)
        self.assertEqual(FP, 0)
        self.assertEqual(FN, 0)

    def test_half_half(self):
        N = 20
        y_true = torch.cat((torch.zeros(N // 2), torch.ones(N // 2)))
        y_pred = torch.cat((torch.zeros(N // 2), torch.ones(N // 2)))
        TP, TN, FP, FN = tutils.confusion_metrics(y_pred, y_true)
        self.assertEqual(TP, N // 2)
        self.assertEqual(TN, N // 2)
        self.assertEqual(FP, 0)
        self.assertEqual(FN, 0)

    def test_half_half_wrong(self):
        N = 20
        y_true = torch.cat((torch.zeros(N // 2), torch.ones(N // 2)))
        y_pred = torch.cat((torch.ones(N // 2), torch.zeros(N // 2)))
        TP, TN, FP, FN = tutils.confusion_metrics(y_pred, y_true)
        self.assertEqual(TP, 0)
        self.assertEqual(TN, 0)
        self.assertEqual(FP, N // 2)
        self.assertEqual(FN, N // 2)

    def test_half_half_wrong2(self):
        N = 20
        y_true = torch.cat((torch.zeros(N // 2), torch.ones(N // 2)))
        y_pred = torch.cat((torch.zeros(N // 2), torch.zeros(N // 2)))
        TP, TN, FP, FN = tutils.confusion_metrics(y_pred, y_true)
        self.assertEqual(TP, 0)
        self.assertEqual(TN, N // 2)
        self.assertEqual(FP, 0)
        self.assertEqual(FN, N // 2)

    def test_half_half_wrong3(self):
        N = 20
        y_true = torch.cat((torch.zeros(N // 2), torch.ones(N // 2)))
        y_pred = torch.cat((torch.ones(N // 2), torch.ones(N // 2)))
        TP, TN, FP, FN = tutils.confusion_metrics(y_pred, y_true)
        self.assertEqual(TP, N // 2)
        self.assertEqual(TN, 0)
        self.assertEqual(FP, N // 2)
        self.assertEqual(FN, 0)

    def test_calc_metrics_all_correct(self):
        N = 20
        y_true = torch.cat((torch.zeros(N // 2), torch.ones(N // 2)))
        y_pred = torch.cat((torch.zeros(N // 2), torch.ones(N // 2)))
        TP, TN, FP, FN = tutils.confusion_metrics(y_pred, y_true)
        acc, prec, rec, f1, TPR, FPR = tutils.calculate_metrics(TP, TN, FP, FN)
        self.assertAlmostEqual(acc, 1.0)
        self.assertAlmostEqual(prec, 1.0)
        self.assertAlmostEqual(rec, 1.0)
        self.assertAlmostEqual(f1, 1.0)
        self.assertAlmostEqual(TPR, 1.0)
        self.assertAlmostEqual(FPR, 0.0)

    def test_calc_metrics_all_wrong(self):
        N = 20
        y_true = torch.cat((torch.zeros(N // 2), torch.ones(N // 2)))
        y_pred = torch.cat((torch.ones(N // 2), torch.zeros(N // 2)))
        TP, TN, FP, FN = tutils.confusion_metrics(y_pred, y_true)
        acc, prec, rec, f1, TPR, FPR = tutils.calculate_metrics(TP, TN, FP, FN)
        self.assertAlmostEqual(acc, 0.0)
        self.assertAlmostEqual(prec, 0.0)
        self.assertAlmostEqual(rec, 0.0)
        self.assertAlmostEqual(f1, 0.0)
        self.assertAlmostEqual(TPR, 0.0)
        self.assertAlmostEqual(FPR, 1.0)

    def test_calc_metrics_half_negative_correct(self):
        N = 20
        y_true = torch.cat((torch.zeros(N // 2), torch.ones(N // 2)))
        y_pred = torch.cat((torch.zeros(N // 2), torch.zeros(N // 2)))
        TP, TN, FP, FN = tutils.confusion_metrics(y_pred, y_true)
        acc, prec, rec, f1, TPR, FPR = tutils.calculate_metrics(TP, TN, FP, FN)
        self.assertAlmostEqual(acc, 0.5)
        self.assertAlmostEqual(prec, 0.0)
        self.assertAlmostEqual(rec, 0.0)
        self.assertAlmostEqual(f1, 0.0)
        self.assertAlmostEqual(TPR, 0.0)
        self.assertAlmostEqual(FPR, 0.0)

    def test_calc_metrics_half_positive_correct(self):
        N = 20
        y_true = torch.cat((torch.zeros(N // 2), torch.ones(N // 2)))
        y_pred = torch.cat((torch.ones(N // 2), torch.ones(N // 2)))
        TP, TN, FP, FN = tutils.confusion_metrics(y_pred, y_true)
        acc, prec, rec, f1, TPR, FPR = tutils.calculate_metrics(TP, TN, FP, FN)
        self.assertAlmostEqual(acc, 0.5)
        self.assertAlmostEqual(prec, 0.5)
        self.assertAlmostEqual(rec, 1.0)
        self.assertAlmostEqual(f1, (2 * (0.5 * 1.0)) / (0.5 + 1.0))
        self.assertAlmostEqual(TPR, 1.0)
        self.assertAlmostEqual(FPR, 1.0)
