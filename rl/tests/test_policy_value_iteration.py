
import numpy as np

from policy_iteration import policy_iteration
from value_iteration import value_iteration


def test_value():
    """just test that it runs"""
    m,n = 15, 20 # matrix dim
    policy = np.random.randint(0,2, size=(m,n)) # 0 for down, 1 for right
    value = np.random.random(size=(m,n))
    matrix = np.random.choice([-1, 0, 1, 3], size=(m,n),p=[0.15,0.5,0.3, 0.05])
    
    value_iteration(matrix, value, policy)



def test_policy():
    """just test that it runs"""
    m,n = 15, 20 # matrix dim
    policy = np.random.randint(0,2, size=(m,n)) # 0 for down, 1 for right
    value = np.random.random(size=(m,n))
    matrix = np.random.choice([-1, 0, 1, 3], size=(m,n),p=[0.15,0.5,0.3, 0.05])
    
    policy_iteration(matrix, value, policy)


