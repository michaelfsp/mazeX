"""
Analysis methods that don't rely on fitting should come here
"""
from mazex.exceptions import *
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

class Map:
    """Mapping between X,Y coordinates into Manhattan vs Shortest
    Path Distance coordinates.
    
    Not in use at the moment.
    """
    def __init__(self, maze, goal):
        self.maze = maze
        self.goal = goal
        self._mapping = None


def avg_dist_dir_prob(session):
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
            for a in step.available_nodes:
                                availability[to_dist_direction(a,
                                    *dist_params)] += 1
            moves[to_dist_direction(next_step.loc, *dist_params)] += 1
        
        prob = moves/availability

    return prob


def distances_plot(maze, goal, **kwargs):
    """Produce a distance vs shortest path distance for a given maze and goal
    location.
    """
    # Finds the longest distance to the goal in this particular maze and goal
    max_maze_distance = nx.algorithms.distance_measures.eccentricity(
            maze, goal)
    max_manhattan_distance = 2 * (maze.size-1)
    
    distances = np.zeros((max_maze_distance+1, max_manhattan_distance+1))
    
    for n in maze.nodes():
       maze_dist = nx.algorithms.shortest_path_length(maze, n, goal) 
       manhattan_dist = abs(goal[0] - n[0]) + abs(goal[1] - n[1])
       point = (manhattan_dist, maze_dist) 

       new_coord = to_dist_coord(n, maze, goal)
       #import pdb; pdb.set_trace()
       distances[new_coord] += 1

    # This turns every element of d into an int
    distances = distances.astype(int)

    shape = distances.shape
    mesh = np.meshgrid(range(shape[1]), range(shape[0]))
    y, x = mesh
    x = x.flatten()
    y = y.flatten()
    ax = pl.gca()
    ax.set_aspect('equal')
    # we create a mask that selects only the nonzero elements of the distances
    # matrix so that the "0" points don't ruin the legend
    for i in np.unique(distances)[np.unique(distances)>0]:
        mask = distances.flatten()==i
        pl.scatter(x.flatten()[mask], y.flatten()[mask],
                s=15*np.pi*distances.flatten()[mask], c='k', label='%i'%i,
                **kwargs)
    pl.xlim([-0.5, max_maze_distance+0.5])
    pl.ylim([-0.5, max_manhattan_distance+0.5])
    pl.xticks(range(0, max_maze_distance+1))
    pl.yticks(range(0, max_manhattan_distance+1))
    pl.grid()
    ax.set_axisbelow(True) # sends the grid to the 'bottom layer'
    pl.xlabel('Maze distance to goal')
    pl.ylabel('Manhattan distance to goal')
    pl.legend(loc='upper left', scatterpoints=1)
    return distances, mesh

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

