"""
Visually represent mazes and trials
"""
import numpy as np
import pylab as pl
import networkx as nx
from mazex.mazes import SquareMaze
from mazex.exceptions import *
from mazex.datamodel import Step, Trial
from mazex.analysis.distances import distances_plot

# other interesting node colors '#848599'

def draw_maze(M, ax=None, hold=None, **kwds):
    """Draw a maze configuration.
    """
    if not isinstance(M, SquareMaze):
        raise DrawException
    if not ax:
        ax = pl.gca()
    # Makes sure the drawings are square and not rectangular
    ax.set_aspect('equal')

    if isinstance(M, SquareMaze):
        nx.draw(M, pos=M.pos, ax=ax, hold=True, node_color='#797b8c', linewidths=0,
                node_size=800, width=6, **kwds)
        nx.draw_networkx_nodes(M, pos=M.pos, ax=ax, hold=hold, node_color='white', linewidths=0,
                node_size=350)
        nx.draw_networkx_nodes(M, pos=M.pos, ax=ax, hold=hold, node_color='#797b8c', linewidths=0,
                node_size=100)
        pl.tight_layout()

def draw_step(S, ax=None, hold=None, **kwds):
    """Represent a maze configuration.
    
    Parameters:
    -----------
    M : maze
       A maze object
    
    **kwds : dictionary
       Additional parameters to be passed onto 
       newtworkx.draw()
    """
    if not isinstance(S, Step):
        raise DrawException
    # First we obtain a list of the nodes that are neither current
    # nor goal location
    normal_nodes = S.trial.maze.nodes()
    # And then we remove current and goal location nodes because
    # those will be plotted differently
    if S.trial.goal != S.loc:
        normal_nodes.remove(S.trial.goal)

    draw_maze(S.trial.maze, ax=ax, hold=True, nodelist=normal_nodes, **kwds)
    # Then we draw the nodes correspoding to current location
    #  we use the heading to shape the arrow, if present
    if S.heading:
        shape = S.heading
    else:
        shape = '8'
    nx.draw_networkx_nodes(S.trial.maze, pos=S.trial.maze.pos,
            nodelist=[S.loc], node_color='#6161f2', node_size=350,
            linewidths=1.5, node_shape=shape)
  # ... and goal location 
    nx.draw_networkx_nodes(S.trial.maze, pos=S.trial.maze.pos,
            nodelist=[S.trial.goal], node_color='#f26161', linewidths=0,
            node_size=800, hold=True)
    nx.draw_networkx_nodes(S.trial.maze, pos=S.trial.maze.pos,
            nodelist=[S.trial.goal], node_color='white', linewidths=0,
            node_size=350, hold=hold)
    nx.draw_networkx_nodes(S.trial.maze, pos=S.trial.maze.pos,
            nodelist=[S.trial.goal], node_color='#f26161', linewidths=0,
            node_size=100, hold=hold)

def draw_trial(T, ax=None, hold=None, **kwds):
    """Draw a Trial. Will represent the taken path if that is available.
    """
    edges = []

    for i in range(len(T.path)-1):
        edges.append((T.path[i], T.path[i+1]))

    draw_step(T.steps[0], ax=ax, hold=True, **kwds)

    nx.draw_networkx_edges(T.maze, pos=T.maze.pos,
            edgelist=edges, edge_color='#61f261', width=3, style='dashed')

def draw(M, ax=None, hold=None, **kwds):
    """Meta function. Calls the write draw function.
    """
    if isinstance(M, SquareMaze):
        draw_maze(M, ax=ax, hold=hold, **kwds)
    
    elif isinstance(M, Step):
        draw_step(M, ax=ax, hold=hold, **kwds)
    
    elif isinstance(M, Trial):
        draw_trial(M, ax=ax, hold=hold, **kwds)
    else:
        raise DrawException
