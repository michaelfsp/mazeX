"""
Analysis methods that rely on information content.
"""
import numpy as np
import pylab as pl
import scipy.optimize
import networkx as nx

import mazex
from mazex.exceptions import *
import mazex.utils as utils

def surprisal(step, null='random'):
    """Surprisal associated with movement out of node at Step.loc.
    
    Parameters:
    -----------
    step : Step
        The step for which surprisal is to be computed

    null : 'random', 'manhattan', 'vonmises'
    """
    #TODO: Give surprisal per direction
    if null == 'random':
        return -np.log2(mazex.analysis.probs.prob(step, null=null))

    else:
        raise NotSupported

psi_fun = lambda t, m, x: (np.exp(t*(x/m))-1)/(np.exp(t)-1)
# we use this function for fsolve to find the right value of t
psi_fun_solve = lambda t, m, x: psi_fun(t, m, x)-(1/2)

def psi(Ic, Ic_exp, Ic_max):
    #the value of the initial guesses for t
    if Ic_exp/Ic_max < 1/2:
        t0 = -1
    else:
        t0 = 1
    t = scipy.optimize.fsolve(psi_fun_solve, t0, args=(Ic_max, Ic_exp))

    return psi_fun(float(t), Ic_max, Ic)

def dist_psi(session, point, first_exit_only=True):
    """Calculate the average direction of choices in distances metric for a
    single point on the distances space.

    It outputs 'normalized surprisal'

    Parameters:
    -----------
        first_exit_only : Boolean
            Consider for the analysis only the first transition out of a node
    """
    #TODO: Spit an error if the point does not exist
    #FIXME: These storage of maze and goal in a step will be deprecated
    maze = session.trial.maze

    Ic = np.zeros(4)
    Ic_max = np.zeros(4)
    Ic_exp = np.zeros(4)

    availability = np.zeros(4)

    for t in session.trials:
        try:
            nodes_on_point = maze.distances_to_coord[point][t.goal]
        except KeyError: # this point doesn't exist on this trial
            continue # ignores the remaining calculations for the current
        # iteration of this for loop

        if first_exit_only:
            step_indices = [t.path.index(n) for n in nodes_on_point if n in
                    t.path]
        else:
            step_indices = []
            for n in nodes_on_point:
                # This list comprehension finds all occurrences of n in list 
                # t.path, unlinke t.path.index() which returns only the first
                # occurrence
                # += 'appends' it to the list step_indices
                step_indices += [i for i, x in enumerate(
                    t.path) if x == n]

        # just to save on text we will save here the parameters we need to
        # call the function to_dist_coord()

        for i in step_indices:
            step = t.steps[i]
            try:
                # will fail if it is the last in the path
                next_step = t.steps[i+1]
            except IndexError:
                continue # ignore current step and go to next...
            # makes sense to not break instead in case the trial was aborted
            # before goal was reached.
            dist_params = (step.loc, t.maze, t.goal)

            available_directions = []
            for a in step.available_nodes:
                available_directions.append(utils.to_dist_direction(a, *dist_params))
            # list() so I can use output as an index
            # set() gets only the unique indices in the list
            # NOTE: if in the same step two maze arms represent the same
            # direction in the distances space, that direction is counted as
            # available only once
            availability_indices = list(set(available_directions))
            # [[]] makes use of advanced indexing
            availability[[availability_indices]] += 1

            index_chosen = utils.to_dist_direction(step.subsequent.loc, *dist_params)

            surpr = surprisal(step)
            # this trick is to avoid problems with sums of nans and Nones
            surpr[surpr==np.inf] = 0
            p = mazex.analysis.probs.prob(step)

            Ic[index_chosen] += surpr[index_chosen]
            Ic_exp += p*surpr
            Ic_max += surpr

    psi_values = np.zeros(4)

    for i in [0,1,2,3]:
        psi_values[i] = psi(Ic[i], Ic_exp[i], Ic_max[i])

    return psi_values, availability


def per_dist_psi(session, first_exit_only=True, plot=True):
    max_dists = np.array([utils.max_distances(t.maze, t.goal) for t in session.trials]
            ).max(0)

    dir_surprisals_dims = max_dists + 1
    dir_surprisals_dims = dir_surprisals_dims.tolist()
    dir_surprisals_dims.append(4)
    dir_surprisals = np.zeros(dir_surprisals_dims)

    x, y = np.meshgrid(range(dir_surprisals.shape[0]), range(dir_surprisals.shape[1]))
    xy = np.array([x.flatten(), y.flatten()]).transpose()

    for dist_coord in xy:
        surpr = dist_psi(session, tuple(dist_coord),
                first_exit_only=first_exit_only)
        dir_surprisals[dist_coord[0], dist_coord[1], :] = surpr

    if not plot:
        return dir_surprisals
    else:
        mazex.analysis.distances.distances_plot(session.trial.maze,
                [t.goal for t in session.trials], plot=True, zorder=10)
        pl.hold(True)
        dx_sign = (1, 1, -1, -1)
        dy_sign = (1, -1, -1, 1)
        # scale factor that will make the maximum length of the arrow 0.5 when
        # projected to either x or y axis. 0.75 = np.sqrt(1**2/2), and 1 is
        # obviously the maximum...
        #FIXME

        max_proj = 1
        # this is simply a scaling factor so that the arrows don't overlap on
        # the plot
        scale = 0.5/max_proj

        for direction in range(0, 4):
            surprisals_xy_proj = scale*np.sqrt(dir_surprisals[:,:,direction
                ]**2/2).transpose().flatten()

            pl.quiver(xy[:,0], xy[:,1], dx_sign[direction]*surprisals_xy_proj,
                   dy_sign[direction]*surprisals_xy_proj, units='xy', scale=1,
                   color='g', width=0.15, headwidth=1, headlength=0.5, zorder=2) 

            # now we overlay only the part of the 'arrow' that goes up to 0.5
            # by passing it through a thresholding function

            threshold = lambda x: 0.5 if 0.5<x<np.inf else x 
            threshold = np.vectorize(threshold)
            
            # here we calculate the length of the vector that is going up to 0.5
            surprisals_xy_proj_05 = scale*np.sqrt(threshold(dir_surprisals[:,:,direction
                ])**2/2).transpose().flatten()

            # for visualization purposes let's plot the vectors of length 0.5
            # and we call them scale vectors
            scale_vector_xy_proj = scale*np.sqrt(np.where(dir_surprisals[
                :,:,direction]>0, 0.5, np.nan)**2/2).transpose().flatten()

            # import pdb; pdb.set_trace()

            pl.quiver(xy[:,0], xy[:,1], dx_sign[direction]*scale_vector_xy_proj,
                    dy_sign[direction]*scale_vector_xy_proj, units='xy', scale=1,
                color='r', width=0.15, headwidth=1, headlength=0, zorder=4) 

            pl.quiver(xy[:,0], xy[:,1], dx_sign[direction]*surprisals_xy_proj_05,
                    dy_sign[direction]*surprisals_xy_proj_05, units='xy', scale=1,
                    color='b', width=0.15, headwidth=1, headlength=0, zorder=5) 

        pl.title('$\psi_{d,l}(\kappa)$', fontsize=30)


