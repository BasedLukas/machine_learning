from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show


def plot_3d_heatmap(
        matrix: np.ndarray, 
        x_values,
        y_values, 
        x_label = "", 
        y_label = "", 
        title = ""
    ) -> None:
    """
    Plot a 3D heatmap of the given matrix with appropriate axes and scales.
    
    Args:
        matrix (np.ndarray): The matrix to be plotted, with shape (m, n)
        x_values (list): Labels for the x-axis ticks.
        y_values (list): Labels for the y-axis ticks.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): The title of the plot.
    """
    fig = plt.figure(figsize=(10, 7)) 
    ax = fig.add_subplot(111, projection='3d')
    

    Y, X = np.meshgrid(y_values, x_values)
    surf = ax.plot_surface(X, Y, matrix, cmap='viridis')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.draw()

def plot_2d_heatmap(data: np.ndarray, title="", xtick_labels=None, ytick_labels=None):
    """
    Create a 2D heatmap from an array.
    
    Args:
        data (np.ndarray): 2D array to visualize as a heatmap.
        title (str): Title of the plot.
        xtick_labels (list): Labels for the x-axis ticks. len==ncols
        ytick_labels (list): Labels for the y-axis ticks. len==nrows
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    heatmap = ax.imshow(data, cmap='viridis', origin='upper', aspect='equal')
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    if xtick_labels: ax.set_xticklabels(xtick_labels)
    if ytick_labels: ax.set_yticklabels(ytick_labels)

    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    plt.draw()

def plot_policy(policy: np.ndarray, title: str):
    """
    Create a policy visualization from an array.
    Args:
        policy (np.ndarray): 2D array representing the policy, where 1 indicates a move to the right and 0 indicates a move down.
        title (str): Title of the plot.
    """
    nrows, ncols = policy.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.set_xlim(-0.5, ncols-0.5)
    ax.set_ylim(-0.5, nrows-0.5)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    
    for i in range(nrows):
        for j in range(ncols):
            dx, dy = (0.8, 0) if policy[i, j] == 1 else (0, 0.8)
            ax.arrow(j, i, dx, dy, head_width=0.2, head_length=0.1, fc='k', ec='k')
    plt.draw()
