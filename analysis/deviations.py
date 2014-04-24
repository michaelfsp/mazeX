"""
Analysis methods based on deviations from the shortest path.
"""

def deviation_from_shortest(trial):
    """
    dg = distance to goal
    spl = shortest path length.
    """
    dg_trajectory = [trial.maze.distances_dictionary[n][trial.goal] for n
            in trial.path]

    sigma_trajectory = []
    while len(sigma_trajectory) != len(dg_trajectory)-1:
        i = len(sigma_trajectory)

        if dg_trajectory[i] > dg_trajectory[i+1]:
            # if the distance drecreased, the agent acted optimally
            sigma_trajectory.append(0)
        elif dg_trajectory[i] < dg_trajectory[i+1]:
            # otherwise we need to find how big the detour was
            # so we search the distances list starting from the point the
            # distance increased until we find the same distance again. The
            # index of that occurrence is the number of extra steps the agent
            # will have to do because of its stupidity here, as compared to an
            # optimal agent
            extra_steps = dg_trajectory[i+1:].index(dg_trajectory[i]) + 1

            # we define the deviation sigma
            sigma_trajectory.append(int(extra_steps/2))

            del dg_trajectory[i+1:i+extra_steps+1]

    return dg_trajectory[:-1], sigma_trajectory
