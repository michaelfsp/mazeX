"""
Analysis methods that don't rely on fitting should come here
"""
from mazex.exceptions import *
import mazex.utils as utils
import networkx as nx
import numpy as np
import pylab as pl

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
    """Returns the coordinates of node n in distance coordinates.
    """
    maze_dist = nx.algorithms.shortest_path_length(maze, n,
            goal) 
    manhattan_dist = abs(goal[0] - n[0]) + abs(goal[1] - n[1])

    return (maze_dist, manhattan_dist)

#TODO: take the common core of dist_dir_prob() nd avg_dist_dir_prob() into
#   separate function
def dist_dir_prob(session, point):
    """Calculate the average direction of choices in distances metric for a
    single point on the distances space.

    It outputs n_times_choice / n_times_available
    """
    moves = np.zeros(4)
    availability = np.zeros(4)

    for t in session.trials:
        # just to save on text we will save here the parameters we need to
        # call the function to_dist_coord()
        for i in range(len(t.steps)-1): # len(t.steps)-1 to exclude the goal
            step = t.steps[i]
            if to_dist_coord(step.loc, t.maze, t.goal) == point:
                next_step = t.steps[i+1]
                dist_params = (step.loc, t.maze, t.goal)

                availability_indices = []
                for a in step.available_nodes:
                    availability_indices.append(to_dist_direction(a, *dist_params))
                availability_indices = list(set(availability_indices))
                # [[]] makes use of advanced indexing
                availability[[availability_indices]] += 1
                moves[to_dist_direction(next_step.loc, *dist_params)] += 1
        
    prob = moves/availability

    return prob

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

def per_dist_avg_dir_prob(session, plot=False):
    # session.maze will return a list if not all trials use the same maze
    # in which case it will be iterable
    max_dists = np.array([max_distances(t.maze, t.goal) for t in session.trials]
            ).max(0)

    dir_probs_dims = max_dists + 1
    dir_probs_dims = dir_probs_dims.tolist()
    dir_probs_dims.append(4)
    dir_probs = np.zeros(dir_probs_dims)
    x, y = np.meshgrid(range(dir_probs.shape[0]), range(dir_probs.shape[1]))
    xy = np.array([x.flatten(), y.flatten()]).transpose()

    for dist_coord in xy:
        dir_probs[dist_coord[0], dist_coord[1], :] = dist_dir_prob(session,
                tuple(dist_coord))

    if not plot:
        return dir_probs
    else:
        distances_plot(session.trial.maze, [t.goal for t in session.trials], plot=True)
        pl.hold(True)
        dx_sign = (1, 1, -1, -1)
        dy_sign = (1, -1, -1, 1)
        # scale just so that arrows don't overlap
        scale = 1
        for direction in range(0, 4):
            xy_proj = scale * np.sqrt(dir_probs[:,:,direction]**2/2).transpose().flatten()
            pl.quiver(xy[:,0], xy[:,1], dx_sign[direction]*xy_proj,
                   dy_sign[direction]*xy_proj, color='b') 



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

