from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show




def plot_3d_heatmap(matrix: np.ndarray, title: str = "") -> None:
    """
    Plot a 3D heatmap of the given matrix
    Args:
        matrix (np.ndarray): The matrix to be plotted, with shape (m, n).
        title (str): The title of the plot.
    """
    m, n = matrix.shape  # m rows, n columns
    x = np.arange(0, m)
    y = np.arange(0, n)
    X, Y = np.meshgrid(y, x)  # Create a meshgrid for Y, X to match matrix's shape (m, n)
    
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, matrix, cmap='viridis')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.text2D(0.05, 0.95, title, transform=ax.transAxes)    
    plt.draw()

def plot_2d_heatmap(data: np.ndarray, title: str):
    """
    Create a 2D heatmap from an array.
    
    Args:
        data (np.ndarray): 2D array to visualize as a heatmap.
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    heatmap = ax.imshow(data, cmap='viridis', origin='upper', aspect='equal')
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
