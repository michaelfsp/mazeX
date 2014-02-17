"""
Analysis methods that don't rely on fitting should come here
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x,y),w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

def distances_plot(maze, goal):
    """Produce a distance vs shortest path distance for a given maze and goal
    location.
    """
    # Finds the longest distance to the goal in this particular maze and goal
    max_shortest_distance = nx.algorithms.distance_measures.eccentricity(
            maze, goal)
    max_manhattan_distance = 2 * (maze.size-1)
    distances = np.zeros((max_manhattan_distance+1, max_shortest_distance+1))

    for n in maze.nodes():
       shortest_dist = nx.algorithms.shortest_path_length(maze, n, goal) 
       manhattan_dist = abs(goal[0] - n[0]) + abs(goal[1] - n[1])
       point = (manhattan_dist, shortest_dist) 
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
    pl.xlim([-0.5, max_shortest_distance+0.5])
    pl.ylim([-0.5, max_manhattan_distance+0.5])
    pl.xticks(range(0, max_shortest_distance+1))
    pl.yticks(range(0, max_manhattan_distance+1))
    pl.grid()
    pl.xlabel('Maze distance to goal')
    pl.ylabel('Manhattan distance to goal')
    pl.legend(loc='upper left', scatterpoints=1)
    return distances, mesh
