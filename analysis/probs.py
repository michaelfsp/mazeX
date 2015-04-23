"""
Analysis methods that don't rely on fitting should come here
"""
from mazex.exceptions import *
from mazex.analysis.iterators import first_exit, first_edge_crossing
import mazex.utils as utils
import networkx as nx
import numpy as np
import pylab as pl
import scipy.optimize

def prob(step, null='random'):
    if null == 'random':
        # now we need to count how many edges correspond to the chosen
        # direction, but only in case there are more edges available than
        # directions in the distances space
        edges_per_direction = np.zeros(4)
        for n in step.available_nodes:
            edges_per_direction[utils.to_dist_direction(n, step.loc,
                step.trial.maze, step.trial.goal)] += 1

        return edges_per_direction/len(step.available_nodes)

    else:
        raise NotSupported

def avg_vector_difference(session, geometric_distance='euclidian',
        exclusion_criterium='node', plot=False, **kwargs):
    chosen_vectors = []
    available_mean_vectors = []
    sum_vector = np.zeros(2)
    step_count = 0

    if exclusion_criterium == 'node':
        step_iterator = first_exit(session)

    elif exclusion_criterium == 'edge':
        step_iterator = first_edge_crossing(session)

    if geometric_distance == 'manhattan':
        unit_vectors = np.array([[1,1], [1,-1], [-1,-1], [-1,1]])

    for s in step_iterator:
        if s.previous: 
            forward_nodes = [a for a in s.available_nodes if a != s.previous.loc]
        else:
            forward_nodes = []

        if len(forward_nodes) > 1:
            step_count +=1

            if geometric_distance == 'manhattan':
                available = np.array([np.nan]*4) 
                for a in forward_nodes:
                    available[utils.to_dist_direction(a, s.loc, session.maze,
                        s.trial.goal)] = np.nansum((available[utils.to_dist_direction(
                            a, s.loc, session.maze, s.trial.goal)], 1))

                available_vectors = available[..., np.newaxis] * unit_vectors
                chosen_vector = unit_vectors[utils.to_dist_direction(s.subsequent.loc,
                    s.loc, session.maze, s.trial.goal)]

            elif geometric_distance == 'euclidian':
                current_geodesic_dist = session.maze.distances_dictionary[s.loc][
                        s.trial.goal]
                current_euclidian_dist = utils.euclidian_distance(
                        s.loc, s.trial.goal)
                available_vectors = []
                for a in forward_nodes:
                    choice_geodesic_dist = session.maze.distances_dictionary[
                            a][s.trial.goal]
                    choice_euclidian_dist = utils.euclidian_distance(
                            a, s.trial.goal)

                    available_vectors.append((
                        choice_geodesic_dist - current_geodesic_dist,
                        choice_euclidian_dist - current_euclidian_dist))
                available_vectors = np.array(available_vectors)

                # we compute the chosen vector
                chosen_geodesic_dist = session.maze.distances_dictionary[
                            s.subsequent.loc][s.trial.goal]
                chosen_euclidian_dist = utils.euclidian_distance(
                            s.subsequent.loc, s.trial.goal)

                d_geodesic_dist = chosen_geodesic_dist - current_geodesic_dist
                d_euclidian_dist = chosen_euclidian_dist - current_euclidian_dist

                chosen_vector = np.array([d_geodesic_dist, d_euclidian_dist])

            available_mean_vector = np.nanmean(available_vectors, axis=0)
            available_mean_vectors.append(list(available_mean_vector))

            chosen_vectors.append(list(chosen_vector))

            sum_vector += chosen_vector - available_mean_vector

    normed_sum_vector = sum_vector/step_count
    if plot:
        pl.arrow(0, 0, *normed_sum_vector, **kwargs)
        # we compute the right xlim and ylim
        lower_bound = min(np.floor(normed_sum_vector.min()), -1)
        upper_bound = max(np.ceil(normed_sum_vector.max()), 1)

        pl.ylim(lower_bound, upper_bound)
        pl.xlim(lower_bound, upper_bound)
        ax = pl.gca()
        ax.set_aspect(1, adjustable='box')
        pl.grid(True)
        pl.tight_layout()

        #return chosen_vectors, available_mean_vectors, sum_vector/step_count
    return normed_sum_vector

def avg_fraction(session, exclusion_criterium='node'):
    available = np.zeros(4)
    chosen = np.zeros(4)

    if exclusion_criterium == 'node':
        step_iterator = first_exit(session)

    elif exclusion_criterium == 'edge':
        step_iterator = first_edge_crossing(session)

    for s in step_iterator:
        for a in s.available_nodes:
            available[utils.to_dist_direction(a, s.loc, session.maze,
                s.trial.goal)] += 1
        chosen[utils.to_dist_direction(s.subsequent.loc, s.loc, session.maze,
            s.trial.goal)] += 1
    return chosen, available

def fraction_dir_vector(session, exclusion_criterium='node'):
    chosen, available = avg_fraction(session, exclusion_criterium)
    fraction = chosen / available

    unit_vectors = np.array([[1,1], [1,-1], [-1,-1], [-1,1]])
    fraction_vectors = fraction[..., np.newaxis] * unit_vectors
    fraction_vector = np.sum(fraction_vectors, axis=0)

    tan_vector = fraction_vector[1]/fraction_vector[0]
    angle = np.arctan(tan_vector)

    return fraction_vector, angle

def prob_m_vs_prob_g(steps):
    unit_vectors = np.array([[1,1], [1,-1], [-1,-1], [-1,1]])
    S_d_M = 0
    available_M = 0
    S_d_G = 0
    available_G = 0
    step_count = 0

    for s in steps:
        if s.previous: 
            forward_nodes = [a for a in s.available_nodes if a != s.previous.loc]
            import ipdb; ipdb.set_trace()
        else:
            forward_nodes = s.available_nodes

        if len(forward_nodes) > 1:
            directions = np.array([unit_vectors[utils.to_dist_direction(
                a, s.loc, s.trial.maze, s.trial.goal)] for a in forward_nodes])

            # We check if there were options along the Geodesical axis
            if np.any(directions[:,0] == -1) and np.any(directions[:,0] == 1):
                available_G += 1
                d_G, d_M = unit_vectors[utils.to_dist_direction(s.subsequent.loc, s.loc, 
                        s.trial.maze, s.trial.goal)]
                S_d_G += d_G

            if np.any(directions[:,1] == -1) and np.any(directions[:,1] == 1):
                available_M += 1
                d_G, d_M = unit_vectors[utils.to_dist_direction(s.subsequent.loc, s.loc, 
                        s.trial.maze, s.trial.goal)]
                S_d_M += d_M
    
    return np.array([S_d_G/available_G, S_d_M/available_M]), (
            available_G, available_M)


#TODO: take the common core of dist_dir_prob() nd avg_dist_dir_prob() into
#   separate function
def dist_dir_prob(session, point, first_exit_only=True):
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

    surprisals = np.zeros(4)
    availability = np.zeros(4)

    surprisals_counter = np.zeros(4)
    availability_counter = np.zeros(4)

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
            availability_counter[[availability_indices]] += 1
            availability[[availability_indices]] += surprisal(step)

            surprisals_counter[utils.to_dist_direction(next_step.loc, *dist_params)] += 1

            surprisals[utils.to_dist_direction(next_step.loc, *dist_params)] += surprisal(
                    step)

    """
    # These don't really make as much sense anymore but will leave them here
    # for now
    if normalized == 'simple':
        prob /= prob.max()

    elif normalized == 'moves':
        prob /= surprisals_counter

    elif normalized == 'prob':
        prob /= prob.sum()
    """
    return surprisals/surprisals_counter, availability/availability_counter

def avg_dist_dir_prob(session, plot=False):
    """Calculate the average direction of choices in distances metric.
    """
    moves = np.zeros(4)
    availability = np.zeros(4)

    for t in session.trials:
        # just to save on text we will save here the parameters we need to
        # call the function to_dist_coord()
        for i in range(len(t.steps)-1): # len(t.steps)-1 to exclude the goal
            step = t.steps[i]
            next_step = t.steps[i+1]
            dist_params = (step.loc, t.maze, t.goal)

            availability_indices = []
            for a in step.available_nodes:
                availability_indices.append(utils.to_dist_direction(a, *dist_params))
            availability_indices = list(set(availability_indices))
            # [[]] activates advanced indexing
            availability[[availability_indices]] += 1

            moves[utils.to_dist_direction(next_step.loc, *dist_params)] += 1
    prob = moves/availability

    if plot:
        # we will use these signs to make the arrows point in the right
        # direction
        dx_sign = (1, 1, -1, -1)
        dy_sign = (1, -1, -1, 1)
        # Will be used for the plotting: hold until the last arrow is drawn
        # and then stop the holding
        hold = (True, True, True, False)
        for i in range(len(prob)):
            # we calculate the distance on x/y axis
            proj_axis = np.sqrt(prob[i]**2/2)
            dx = dx_sign[i] * proj_axis
            dy = dy_sign[i] * proj_axis
            pl.arrow(0, 0, dx, dy, hold[i], lw=2)

        pl.xlim(-1.2, 1.2)
        pl.ylim(-1.2, 1.2)
        ax = pl.gca()
        ax.set_aspect('equal')
        pl.xlabel('$\Delta$Z')
        pl.ylabel('$\Delta$M')
        pl.tight_layout()

    else:
        return prob

def per_dist_avg_dir_prob(session, first_exit_only=True,
        availability='plot', plot=False):
    """Calculate the probability of direction choice on the distances space.
    Right now the way the algorithm is the following:
    
       For every point in the distances space, calculate for every direction
       number of times it is available (availability) and the number of times
       it is taken (moves) and return the 'probability' as moves/availability.
    """
    # session.maze will return a list if not all trials use the same maze
    # in which case it will be iterable
    max_dists = np.array([utils.max_distances(t.maze, t.goal) for t in session.trials]
            ).max(0)

    dir_surprisals_dims = max_dists + 1
    dir_surprisals_dims = dir_surprisals_dims.tolist()
    dir_surprisals_dims.append(4)
    dir_surprisals = np.zeros(dir_surprisals_dims)
    dir_availabilities = np.zeros(dir_surprisals_dims)

    x, y = np.meshgrid(range(dir_surprisals.shape[0]), range(dir_surprisals.shape[1]))
    xy = np.array([x.flatten(), y.flatten()]).transpose()

    for dist_coord in xy:
        #import pdb; pdb.set_trace()
        surpr, avail = dist_dir_prob(session, tuple(dist_coord),
                first_exit_only=first_exit_only)
        dir_surprisals[dist_coord[0], dist_coord[1], :] = surpr
        dir_availabilities[dist_coord[0], dist_coord[1], :] = avail

    if not plot:
        return dir_surprisals, dir_availabilities
    else:
        distances_plot(session.trial.maze, [t.goal for t in session.trials],
                plot=True, zorder=10)
        pl.hold(True)
        dx_sign = (1, 1, -1, -1)
        dy_sign = (1, -1, -1, 1)
        # scale factor that will make the maximum length of the arrow 0.5 when
        # projected to either x or y axis. 0.75 = np.sqrt(1**2/2), and 1 is
        # obviously the maximum...
        #FIXME

        if availability == 'normalizer':
            # maximum would be 1, since the surprisals are normalized
            # this is simply the value of the projection of a vector of length
            # 1 and angle 45 degr. onto x or y axes...
            max_proj = np.sqrt(1/2)
        else:
            # if not normalized the maximum surprisal is 2
            max_proj = np.sqrt((2**2)/2)
        
        # this is simply a scaling factor so that the arrows don't overlap on
        # the plot
        scale = 0.5/max_proj

        for direction in range(0, 4):
            surprisals_xy_proj = scale*np.sqrt(dir_surprisals[:,:,direction
                ]**2/2).transpose().flatten()
            pl.quiver(xy[:,0], xy[:,1], dx_sign[direction]*surprisals_xy_proj,
                   dy_sign[direction]*surprisals_xy_proj, units='xy', scale=1,
                   color='b', zorder=2) 

            if availability == 'plot':
                availabilities_xy_proj = scale*np.sqrt(dir_availabilities[:,:,direction
                    ]**2/2).transpose().flatten()
                pl.quiver(xy[:,0], xy[:,1], dx_sign[direction]*availabilities_xy_proj,
                       dy_sign[direction]*availabilities_xy_proj, units='xy', scale=1,
                       color='r', zorder=1) 

def heatmap(session, **kwargs):
    plot=True
    goals = [t.step.goal for t in session.trials]

    heatarray_shape = np.array(utils.max_distances(session.trial.step.maze, goals)) + 1

    heatarray = np.zeros(heatarray_shape)

    for t in session.trials:
        for s in t.path[:-1]:
            heatarray[utils.to_dist_coord(s, t.step.maze, t.step.goal)] += 1

    maze = session.trial.step.maze
    max_distances_all = utils.max_distances(maze, goals)

    dist_shape = np.array(max_distances_all) + 1
    distances = np.zeros(dist_shape)

    for goal in goals:
        for n in maze.nodes():
           new_coord = utils.to_dist_coord(n, maze, goal)
           distances[new_coord] += 1

    if len(goals) == 1:
        # This turns every element of d into an int
        distances = distances.astype(int)

    else:
        distances = distances / len(goals)

    # it is a bit hacky but it produces the plot we want... good for now
    mesh = np.meshgrid(range(dist_shape[1]), range(dist_shape[0]))
    y, x = mesh
    x = x.flatten()
    y = y.flatten()

    if plot:
        ax = pl.gca()
        ax.set_aspect('equal')
        # multiplies the value to obtain number of pixels in the surface
        area_factor = 30* 15 * np.pi
        pl.scatter(x.flatten(), y.flatten(),
                    s=(area_factor*heatarray.flatten())/heatarray.sum(), c='k', **kwargs)

        # Very ugly hack just to get some legend working
        bullet_sizes = np.linspace(0.001, np.max(heatarray/heatarray.sum()), 4)
        for b in bullet_sizes:
            pl.scatter(-1,-1, s=area_factor*b, c='k', label='%f' %b, **kwargs)
        pl.legend(loc='upper left', scatterpoints=1, title='<nodes>')

        pl.xlim([-0.5, max_distances_all[0]+0.5])
        pl.ylim([-0.5, max_distances_all[1]+0.5])
        pl.xticks(range(0, max_distances_all[0]+1))
        pl.yticks(range(0, max_distances_all[1]+1))
        pl.grid()
        ax.set_axisbelow(True) # sends the grid to the 'bottom layer'
        pl.xlabel('Maze distance to goal')
        pl.ylabel('Manhattan distance to goal')

    else:
        return distances, x, y

