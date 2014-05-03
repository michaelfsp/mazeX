"""
Analysis methods based on angle of edge chosen.
"""
import numpy as np
import pylab as pl

import mazex.utils as utils

def cum_prob_angle_choice(session, first_exit_only=True, plot=True, **kwargs):
    """Calculate choice probability by cos of angle formed between edge and
    goal."""
    increase = []
    decrease = []

    for t in session.trials:
        if first_exit_only:
            step_indices = [t.path.index(n) for n in set(t.path[:-1])]

            for i in step_indices:
                if t.maze.distances_dictionary[t.path[i]][t.goal] > t.maze.distances_dictionary[t.path[i+1]][t.goal]:
                    decrease.append(utils.cos_between_nodes(t.path[i], t.path[i+1], t.goal))
                elif t.maze.distances_dictionary[t.path[i]][t.goal] < t.maze.distances_dictionary[t.path[i+1]][t.goal]:
                    increase.append(utils.cos_between_nodes(t.path[i], t.path[i+1], t.goal))
                else:
                    print('Unexpected: distance to goal did not change in this step!')

        else:
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
        pl.ylabel('Cumulative probability')
        yaxis_decrease = np.arange(0, len(decrease))
        yaxis_increase = np.arange(0, len(increase))
        pl.plot(decrease, yaxis_decrease/len(decrease), label='decrease in distance', **kwargs)
        pl.plot(increase, yaxis_increase/len(increase), label='increase in distance', **kwargs)
        pl.legend(loc=2)

    else:
        return increase, decrease

def prob_angle_choice(session, first_exit_only=True, plot=True, **kwargs):
    """Calculate choice probability by cos of angle formed between edge and
    goal."""
    increase = []
    decrease = []

    for t in session.trials:
        if first_exit_only:
            step_indices = [t.path.index(n) for n in set(t.path[:-1])]

            for i in step_indices:
                if t.maze.distances_dictionary[t.path[i]][t.goal] > t.maze.distances_dictionary[t.path[i+1]][t.goal]:
                    decrease.append(utils.cos_between_nodes(t.path[i], t.path[i+1], t.goal))
                elif t.maze.distances_dictionary[t.path[i]][t.goal] < t.maze.distances_dictionary[t.path[i+1]][t.goal]:
                    increase.append(utils.cos_between_nodes(t.path[i], t.path[i+1], t.goal))
                else:
                    print('Unexpected: distance to goal did not change in this step!')

        else:
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
        pl.title('Probability of choice vs cos($\\theta$)')
        pl.xlabel('cos($\\theta$)')
        pl.ylabel('Choice Probability')
        d_hist, d_bin_edges = np.histogram(decrease)
        i_hist, i_bin_edges = np.histogram(increase)

        # find the midpoints between the bin edges
        d_xaxis = [(d_bin_edges[i]+d_bin_edges[i+1])/2 for i in range(len(
            d_bin_edges)-1)]

        i_xaxis = [(i_bin_edges[i]+i_bin_edges[i+1])/2 for i in range(len(
            i_bin_edges)-1)]

        pl.plot(d_xaxis, d_hist/np.sum(d_hist), label='decrease in distance', **kwargs)
        pl.plot(i_xaxis, i_hist/np.sum(i_hist), label='increase in distance', **kwargs)
        pl.legend(loc=2)

    else:
        return (d_xaxis, d_hist), (i_xaxis, i_hist)

