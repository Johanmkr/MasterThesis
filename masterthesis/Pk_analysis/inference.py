import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, accuracy_score

from IPython import embed

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import NEW_CNN.train_utils as tutils

# Load data and create dataframes
S1_data = np.loadtxt("S1_results.txt", skiprows=1)
S2_data = np.loadtxt("S2_results.txt", skiprows=1)
S1_frame = pd.DataFrame(S1_data, columns=["score", "prediction", "true_values", "loss"])
S2_frame = pd.DataFrame(S2_data, columns=["score", "prediction", "true_values", "loss"])

# Apply tolerances to the predictions
TOLERANCE = 0.5
S1_frame["bool_pred"] = (S1_frame["prediction"] > TOLERANCE).astype(bool)
S1_frame["int_pred"] = (S1_frame["prediction"] > TOLERANCE).astype(int)
S1_frame["bool_true"] = (S1_frame["true_values"] > TOLERANCE).astype(bool)
S2_frame["bool_pred"] = (S2_frame["prediction"] > TOLERANCE).astype(bool)
S2_frame["int_pred"] = (S2_frame["prediction"] > TOLERANCE).astype(int)
S2_frame["bool_true"] = (S2_frame["true_values"] > TOLERANCE).astype(bool)

# For S1
cfm1 = confusion_matrix(S1_frame["int_pred"], S1_frame["true_values"])
tn1, fp1, fn1, tp1 = cfm1.ravel()
accuracy1, precision1, recall1, F11, TPR1, FPR1 = tutils.calculate_metrics(TP=tp1, TN=tn1, FP=fp1, FN=fn1)
fpr1, tpr1, thresholds1 = roc_curve(S1_frame["true_values"], S1_frame["prediction"])

# For S1
cfm2 = confusion_matrix(S2_frame["int_pred"], S2_frame["true_values"])
tn2, fp2, fn2, tp2 = cfm2.ravel()
accuracy2, precision2, recall2, F12, TPR2, FPR2 = tutils.calculate_metrics(TP=tp2, TN=tn2, FP=fp2, FN=fn2)
fpr2, tpr2, thresholds2 = roc_curve(S2_frame["true_values"], S2_frame["prediction"])


