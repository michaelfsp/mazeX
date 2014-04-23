"""
Analysis methods based on deviations from the shortest path.
"""

def deviation_from_shortest(trial):
    """
    dg = distance to goal
    spl = shortest path length.
    """
    dg_trajectory = [trial.maze.distances_dictionary[trial.start][trial.goal]]
    excess_steps = [0]
    sigma_trajectory = [0]

    for i, n in enumerate(trial.path[1:]):
        dg_trajectory.append(trial.maze.distances_dictionary[n][trial.goal])
        #i+1 is the number of steps taken so far
        excess_steps.append(dg_trajectory[-1] - dg_trajectory[0]-(i+1))
        sigma_trajectory = (excess_steps[-1] - excess_steps[-2])/2

    return dg_trajectory, sigma_trajectory




