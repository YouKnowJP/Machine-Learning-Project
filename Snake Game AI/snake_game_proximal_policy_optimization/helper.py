# helper.py

import matplotlib.pyplot as plt
from IPython import display
from typing import List, Union

plt.ion()  # Enable interactive mode for live plot updates

def plot(scores: List[Union[int, float]], mean_scores: List[Union[int, float]]) -> None:
    display.clear_output(wait=True)
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score', color='blue')
    plt.plot(mean_scores, label='Mean Score', color='orange')
    plt.ylim(bottom=0)
    plt.text(len(scores)-1, scores[-1], f'{scores[-1]:.2f}', fontsize=9)
    plt.text(len(mean_scores)-1, mean_scores[-1], f'{mean_scores[-1]:.2f}', fontsize=9)
    plt.legend()
    display.display(plt.gcf())
    plt.pause(0.1)
