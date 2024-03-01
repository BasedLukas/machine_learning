
import numpy as np
from utils.plotting import plot_2d_heatmap, show




def search(i,j,dp={})->int:
    cached = dp.get((i,j), None)
    if cached: return cached
    if (i,j) == (m-1,n-1):
        dp[(i, j)] = matrix[i, j]
        return matrix[i, j]

    if i < (m-1):
        down = search(i+1, j, dp)
    else:
        down = -float("inf")
    if j < (n-1):
        right = search(i, j+1, dp)
    else:
        right = -float("inf")
    
    dp[(i, j)] = matrix[i, j] + max(down, right)
    return dp[(i, j)]

if __name__ == "__main__":


    m,n = 15, 20 # matrix dim
    value = np.random.random(size=(m,n))
    matrix = np.random.choice([-1, 0, 1, 3], size=(m,n),p=[0.15,0.5,0.3, 0.05])

    dp = {}

    search(0,0, dp)
    for k,v in dp.items():
        value[k] = v

    plot_2d_heatmap(value, "value map")
    plot_2d_heatmap(matrix, "environment")
    show()


