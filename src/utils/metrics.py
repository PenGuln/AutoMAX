"""
Metrics and visualization utilities for AutoAUC.
"""

import numpy as np
import matplotlib.pyplot as plt


def show_diagram(target, trainlog):
    """
    Show training progress diagram.
    
    Args:
        target: Target metric name
        trainlog: List of training logs
    """
    x = np.arange(len(trainlog))
    curves = trainlog[0].keys()
    plt.figure()
    for c in curves:
        if c == "loss":
            continue
        y = [item[c][target] for item in trainlog]
        plt.plot(x, y, label=c, linewidth=3)
    plt.legend(fontsize=15)
    plt.ylabel(target, fontsize=25)
    plt.xlabel('Epoch', fontsize=25)
    plt.show()
