## Currently still work in progress

## Algos
These files can be run directly (as `__main__`)

#### Policy Iteration / Value Iteration / Dynamic Programming
`policy_iteration.py` and `value_iteration.py` and `dynamic_programming.py`

![policy](assets/policy.png "Policy")

Task: Move from (0,0) to (m-1, n-1) by only moving down or right. Collect the value of each square that you land on. This is a finite, discrete and deterministic task. Your choices are either down or right, the episode always comes to an end and there is no randomness involved.

![value function](assets/value_3d.png "Value Function")

- [ ] policy iteration doesn't seem to converge to optimal solution ???

## Games
#### Tree
Tree is a simple tree that can be followed or traversed, where each node has a fixed underlying value [-1,1] where all daughter nodes values sum to 0.
However on each visit to the node the current value is drawn from a distribution centred around that value, to add some uncertainty. This means there is an optimal way to follow the tree, but it is not immediately apparent.