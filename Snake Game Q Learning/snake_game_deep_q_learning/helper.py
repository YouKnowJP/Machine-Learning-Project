#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:11:36 2025

@author: youknowjp
"""

import matplotlib.pyplot as plt
from IPython import display
from typing import List, Union, Optional

# Enable interactive mode for live plot updates.
plt.ion()

def plot(scores: List[Union[int, float]], mean_scores: List[Union[int, float]], title: Optional[str] = 'Training Progress') -> None:
    """
    Plots the training progress, including scores and mean scores.

    Args:
        scores (List[Union[int, float]]): List of scores.
        mean_scores (List[Union[int, float]]): List of mean scores.
        title (Optional[str]): optional title of the plot. Defaults to 'Training Progress'.
    """
    display.clear_output(wait=True)
    plt.clf()
    plt.title(title)
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score', color='blue')
    plt.plot(mean_scores, label='Mean Score', color='orange')
    plt.ylim(bottom=0)

    # Add text annotations for the latest scores and mean scores.
    if scores:
        plt.text(len(scores) - 1, scores[-1], f'{scores[-1]:.2f}', fontsize=9)
    if mean_scores:
        plt.text(len(mean_scores) - 1, mean_scores[-1], f'{mean_scores[-1]:.2f}', fontsize=9)

    plt.legend()
    display.display(plt.gcf())
    plt.pause(0.1)
