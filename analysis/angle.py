"""
Analysis methods based on angle of edge chosen.
"""


def prob_angle_choice(session, plot=True, **kwargs):
    """Calculate choice probability by cos of angle formed between edge and
    goal."""
    increase = []
    decrease = []

    for t in session.trials:
        for i,s in enumerate(t.path[:-1]):
            if t.maze.distances_dictionary[s][t.goal] > t.maze.distances_dictionary[t.path[i+1]][t.goal]:
                decrease.append(utils.cos_between_nodes(s, t.path[i+1], t.goal))
            elif t.maze.distances_dictionary[s][t.goal] < t.maze.distances_dictionary[t.path[i+1]][t.goal]:
                increase.append(utils.cos_between_nodes(s, t.path[i+1], t.goal))
            else:
                print('Unexpected: distance to goal did not change in this step!')

    increase.sort()
    decrease.sort()

    if plot:
        pl.figure()
        pl.title('Cumulative frequency of choice vs cos($\\theta$)')
        pl.xlabel('cos($\\theta$)')
        pl.ylabel('Cumulative frequency')
        yaxis_decrease = np.arange(0, len(decrease))
        yaxis_increase = np.arange(0, len(increase))
        pl.plot(decrease, yaxis_decrease/len(decrease), label='decrease in distance', **kwargs)
        pl.plot(increase, yaxis_increase/len(increase), label='increase in distance', **kwargs)
        pl.legend()

    else:
        return increase, decrease


