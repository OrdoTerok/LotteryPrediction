"""
DEPRECATED: Moved to visualization/plot_utils_std.py
"""

import logging
import matplotlib.pyplot as plt
...
import logging
# Remove or comment out the following block to avoid overwriting the main log file
# logging.basicConfig(
#     filename='log.rtf',
#     filemode='a',
#     format='%(asctime)s %(levelname)s: %(message)s',
#     level=logging.INFO
# )


import matplotlib.pyplot as plt
import numpy as np

def plot_multi_round_true_std(y_true, rounds_pred_list, prev_pred=None, num_balls=5, round_labels=None, prev_label='Previous'):
    

def plot_multi_round_pred_std():
    

def kl_divergence(p, q, n_classes=None):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # If n_classes is provided, treat p and q as integer arrays and convert to distributions
    if n_classes is not None:
        p = np.bincount(p.astype(int)-1, minlength=n_classes) / len(p)
        q = np.bincount(q.astype(int)-1, minlength=n_classes) / len(q)
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    return np.sum(p * np.log(p / q))

def plot_multi_round_kl_divergence():
    
