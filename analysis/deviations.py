"""
Analysis methods based on deviations from the shortest path.
"""
import numpy as np

def stupid_moves(trial):
    """
    dg = distance to goal
    spl = shortest path length.
    """
    dg_trajectory = [trial.maze.distances_dictionary[n][trial.goal] for n
            in trial.path]

    delta = np.diff(dg_trajectory)

    sigma = np.zeros(dg_trajectory[0])

    for i, d in enumerate(delta):
        if d == 1:
            position = dg_trajectory[0] - dg_trajectory[i]
            sigma[position] += 1

    return np.arange(dg_trajectory[0], 0, -1), sigma
