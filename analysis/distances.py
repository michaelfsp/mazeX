"""Basic analysis tools for operations on the distances space.
"""
import numpy as np
import pylab as pl

from mazex.exceptions import *
import mazex.utils as utils

def distances_plot(maze, goals=None, plot=True, node_size=False, **kwargs):
    """Produce a distance vs shortest path distance for a given maze and goal
    location.
    """
    # If no goal is specified, consider every node as a possible goal location
    if not goals:
        goals = maze.nodes()
        max_distances_all = utils.max_distances(maze)

    else:
        # In case only one goal was passed as argument
        # TODO: This is very hacky. improve...
        try:
            utils.assert_is_node(goals)
            goals = [goals]
        except InvalidNode:
            pass

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

        if node_size:
            # multiplies the value to obtain number of pixels in the surface
            area_factor = 15 * np.pi
            pl.scatter(x.flatten(), y.flatten(),
                        s=area_factor*distances.flatten(), c='k', **kwargs)

            # Very ugly hack just to get some legend working
            bullet_sizes = np.arange(1, distances.max()+1, 1)
            for b in bullet_sizes:
                pl.scatter(-1,-1, s=area_factor*b, c='k', label='%i' %b, **kwargs)

                pl.legend(loc='upper left', scatterpoints=1, title='<nodes>')

        else:
            area_factor = 15 * np.pi
            pl.scatter(x.flatten(), y.flatten(),
                        s=np.ones_like(distances.flatten()), c='k', **kwargs)

        pl.xlim([-0.5, max_distances_all[0]+0.5])
        pl.ylim([-0.5, max_distances_all[1]+0.5])
        pl.xticks(range(0, max_distances_all[0]+1))
        pl.yticks(range(0, max_distances_all[1]+1))
        pl.grid()
        ax.set_axisbelow(True) # sends the grid to the 'bottom layer'
        pl.xlabel('Maze distance to goal', fontsize=22)
        pl.ylabel('Manhattan distance to goal', fontsize=22)

    else:
        return distances, x, y

def distances_plot_path(trial):
    """Plot paths overlayed onto the distances plot.
    """
    distances_plot(trial.step.maze, trial.step.goal, zorder=10)
    for i in range(len(trial.path)-2):
        first_node = utils.to_dist_coord(trial.path[i], trial.step.maze,
                trial.step.goal)
        second_node = utils.to_dist_coord(trial.path[i+1], trial.step.maze,
                trial.step.goal)
        # zorder=1 sends the lines to the 'bottom layer'
        pl.plot([first_node[0], second_node[0]], [first_node[1],
            second_node[1]], 'g', linewidth=4, zorder=1)

