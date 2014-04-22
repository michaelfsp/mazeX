"""
Analysis methods that don't rely on fitting should come here
"""
from mazex.exceptions import *
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
            edges_per_direction[to_dist_direction(n, step.loc,
                step.trial.maze, step.trial.goal)] += 1

        return edges_per_direction/len(step.available_nodes)

    else:
        raise NotSupported

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
        return -np.log2(prob(step, null=null))

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

    return psi_fun(t, Ic_max, Ic)

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
                available_directions.append(to_dist_direction(a, *dist_params))
            # list() so I can use output as an index
            # set() gets only the unique indices in the list
            # NOTE: if in the same step two maze arms represent the same
            # direction in the distances space, that direction is counted as
            # available only once
            availability_indices = list(set(available_directions))
            # [[]] makes use of advanced indexing
            availability[[availability_indices]] += 1

            index_chosen = to_dist_direction(step.subsequent.loc, *dist_params)

            surpr = surprisal(step)
            # this trick is to avoid problems with sums of nans and Nones
            surpr[surpr==np.inf] = 0
            p = prob(step)

            Ic[index_chosen] += surpr[index_chosen]
            Ic_exp += p*surpr
            Ic_max += surpr

    Ic = Ic/availability
    Ic_exp = Ic_exp/availability
    Ic_max = Ic_max/availability

    psi_values = np.zeros(4)

    for i in [0,1,2,3]:
        if Ic[i]:
            psi_values[i] = (psi(Ic[i], Ic_exp[i], Ic_max[i]))

    return psi_values



def to_dist_direction(n, loc, maze, goal):
    """Calculate the direction in the distances space of a given transition
    from loc to n.
    
    I am not sure if it is possible to have something other than
    the 4 possible transitions I have identified so I will make
    conditions very explicit in order to detect eventual
    exceptions
    """
    loc = to_dist_coord(loc, maze, goal)
    n = to_dist_coord(n, maze, goal)

    if n[0]>loc[0] and n[1]>loc[1]:
        "NE transition in our convention for the axis on this diagram"
        return 0
    elif n[0]>loc[0] and n[1]<loc[1]:
        "SE transition in our convention for the axis on this diagram"
        return 1
    elif n[0]<loc[0] and n[1]<loc[1]:
        "SW transition in our convention for the axis on this diagram"
        return 2
    elif n[0]<loc[0] and n[1]>loc[1]:
        "NW transition in our convention for the axis on this diagram"
        return 3
    else:
        raise WeirdDirection

def to_dist_coord(n, maze, goal):
    """Returns the coordinates of node n in distance coordinates in a tuple
    of the shape (maze_distance, manhattan_distance).
    """
    return (int(maze.distances_dictionary[n][goal]),
            abs(goal[0] - n[0]) + abs(goal[1] - n[1]))

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
                available_directions.append(to_dist_direction(a, *dist_params))
            # list() so I can use output as an index
            # set() gets only the unique indices in the list
            # NOTE: if in the same step two maze arms represent the same
            # direction in the distances space, that direction is counted as
            # available only once
            availability_indices = list(set(available_directions))
            # [[]] makes use of advanced indexing
            availability_counter[[availability_indices]] += 1
            availability[[availability_indices]] += surprisal(step)

            surprisals_counter[to_dist_direction(next_step.loc, *dist_params)] += 1

            surprisals[to_dist_direction(next_step.loc, *dist_params)] += surprisal(
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
                availability_indices.append(to_dist_direction(a, *dist_params))
            availability_indices = list(set(availability_indices))
            # [[]] activates advanced indexing
            availability[[availability_indices]] += 1

            moves[to_dist_direction(next_step.loc, *dist_params)] += 1
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
    max_dists = np.array([max_distances(t.maze, t.goal) for t in session.trials]
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

def max_distances(maze, goals=None):
    # the output of this function is the maximum of an array called
    # max_distances_all
    # we first need to create that array based on the three possible scenarios
    # no goal, single goals, multiple goals
    # for each goal calculate the maximum Maze and Manhattan distances, store
    # that in a list, turn list into array and calculate the maximum over the
    # first axis
    if not goals:
        max_maze_distance = nx.algorithms.distance_measures.diameter(maze)
        max_manhattan_distance = 2 * (maze.size-1)
        return (max_maze_distance, max_manhattan_distance)

    # In case only one goal was passed as argument
    # TODO: This is very hacky. improve...
    try:
        utils.assert_is_node(goals)
        goals = [goals]
    except InvalidNode:
        pass

    max_distances_all = []
    
    for goal in goals:
        max_maze_distance = nx.algorithms.distance_measures.eccentricity(maze,
                goal)
        # find largest x distance to this goal
        max_x_dist = max((goal[0], maze.size-1 - goal[0]))
        # same for y
        max_y_dist = max((goal[1], maze.size-1 - goal[1]))
        max_manhattan_distance = max_x_dist + max_y_dist

        max_distances_all.append([max_maze_distance, max_manhattan_distance])

    max_distances_all = np.array(max_distances_all)
    return max_distances_all.max(0)

def heatmap(session, **kwargs):
    plot=True
    goals = [t.step.goal for t in session.trials]

    heatarray_shape = np.array(max_distances(session.trial.step.maze, goals)) + 1

    heatarray = np.zeros(heatarray_shape)

    for t in session.trials:
        for s in t.path[:-1]:
            heatarray[to_dist_coord(s, t.step.maze, t.step.goal)] += 1

    maze = session.trial.step.maze
    max_distances_all = max_distances(maze, goals)

    dist_shape = np.array(max_distances_all) + 1
    distances = np.zeros(dist_shape)

    for goal in goals:
        for n in maze.nodes():
           new_coord = to_dist_coord(n, maze, goal)
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


def distances_plot(maze, goals=None, plot=True, **kwargs):
    """Produce a distance vs shortest path distance for a given maze and goal
    location.
    """
    # If no goal is specified, consider every node as a possible goal location
    if not goals:
        goals = maze.nodes()
        max_distances_all = max_distances(maze)

    else:
        # In case only one goal was passed as argument
        # TODO: This is very hacky. improve...
        try:
            utils.assert_is_node(goals)
            goals = [goals]
        except InvalidNode:
            pass

        max_distances_all = max_distances(maze, goals)

    dist_shape = np.array(max_distances_all) + 1
    distances = np.zeros(dist_shape)

    for goal in goals:
        for n in maze.nodes():
           new_coord = to_dist_coord(n, maze, goal)
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
        area_factor = 15 * np.pi
        pl.scatter(x.flatten(), y.flatten(),
                    s=area_factor*distances.flatten(), c='k', **kwargs)

        # Very ugly hack just to get some legend working
        bullet_sizes = np.arange(1, distances.max()+1, 1)
        for b in bullet_sizes:
            pl.scatter(-1,-1, s=area_factor*b, c='k', label='%i' %b, **kwargs)
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

def distances_plot_path(trial):
    """Plot paths overlayed onto the distances plot.
    """
    distances_plot(trial.step.maze, trial.step.goal, zorder=10)
    for i in range(len(trial.path)-2):
        first_node = to_dist_coord(trial.path[i], trial.step.maze,
                trial.step.goal)
        second_node = to_dist_coord(trial.path[i+1], trial.step.maze,
                trial.step.goal)
        # zorder=1 sends the lines to the 'bottom layer'
        pl.plot([first_node[0], second_node[0]], [first_node[1],
            second_node[1]], 'g', linewidth=4, zorder=1)

