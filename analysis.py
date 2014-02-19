"""
Analysis methods that don't rely on fitting should come here
"""
import networkx as nx
import numpy as np
import pylab as pl

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
    """
    def __init__(self, maze, goal):
        self.maze = maze
        self.goal = goal
        self._mapping = None

def distances_plot(maze, goal):
    """Produce a distance vs shortest path distance for a given maze and goal
    location.
    """
    # Finds the longest distance to the goal in this particular maze and goal
    max_maze_distance = nx.algorithms.distance_measures.eccentricity(
            maze, goal)
    max_manhattan_distance = 2 * (maze.size-1)
    distances = np.zeros((max_manhattan_distance+1, max_maze_distance+1))

    
    for n in maze.nodes():
       maze_dist = nx.algorithms.shortest_path_length(maze, n, goal) 
       manhattan_dist = abs(goal[0] - n[0]) + abs(goal[1] - n[1])
       point = (manhattan_dist, maze_dist) 
       distances[point] += 1

    # This turns every element of d into an int
    distances = distances.astype(int)

    shape = distances.shape
    mesh = np.meshgrid(range(shape[1]), range(shape[0]))
    x, y = mesh
    x = x.flatten()
    y = y.flatten()
    ax = pl.gca()
    ax.set_aspect('equal')
    for i in np.unique(distances)[np.unique(distances)>0]:
        mask = distances.flatten()==i
        pl.scatter(x.flatten()[mask], y.flatten()[mask],
                s=15*np.pi*distances.flatten()[mask], c='k', label='%i'%i)
    pl.xlim([-0.5, max_maze_distance+0.5])
    pl.ylim([-0.5, max_manhattan_distance+0.5])
    pl.xticks(range(0, max_maze_distance+1))
    pl.yticks(range(0, max_manhattan_distance+1))
    pl.grid()
    pl.xlabel('Maze distance to goal')
    pl.ylabel('Manhattan distance to goal')
    pl.legend(loc='upper left', scatterpoints=1)
    return distances, mesh

def distances_plot_path(trial):
    """Plot paths overlayed onto the distances plot.
    """
    distances_plot(trial.step.maze, trial.step.goal)
    for i in range(len(trial.path)):
        if trial.path[i+1]:
            pl.plot([trial.path[i][0], trial.path[i+1][0]],
                    [trial.path[i][1], trial.path[i+1][1]])
