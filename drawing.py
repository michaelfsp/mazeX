"""
Visually represent mazes and trials
"""
import numpy as np
import pylab as pl
import networkx as nx
from mazex.mazes import SquareMaze
from mazex.exceptions import *
from mazex.task import *

def draw(M, ax=None, hold=None, **kwds):
	"""Represent a maze configuration.
	
	Parameters:
	-----------
	M : maze
	   A maze object
	
	**kwds : dictionary
	   Additional parameters to be passed onto 
	   newtworkx.draw()
	"""
	#if not ax:
		#		ax = pl.gca()
		# Makes sure the drawings are square and not rectangular
		#ax.set_aspect('equal')

	if isinstance(M, SquareMaze):
		nx.draw(M, pos=M.pos, ax=ax, hold=hold, node_color='gray', **kwds)
	
	elif isinstance(M, Step):
		# First we obtain a list of the nodes that are neither current
		# nor goal location
		normal_nodes = M.maze.nodes()
		# And then we remove current and goal location nodes because
		# those will be plotted differently
		normal_nodes.remove(M.loc)
		if M.goal != M.loc:
			normal_nodes.remove(M.goal)
		nx.draw(M.maze, pos=M.maze.pos, ax=ax, hold=True, node_color='gray',
				nodelist=normal_nodes, kwds=kwds)
		# Then we draw the nodes correspoding to current location
		#  we use the heading to shape the arrow, if present
		if M.heading:
			shape = M.heading
		else:
			shape = '8'
		nx.draw_networkx_nodes(M.maze, pos=M.maze.pos,
				nodelist=[M.loc], node_color='b',
				node_shape=shape)
		# ... and goal location 
		nx.draw_networkx_nodes(M.maze, pos=M.maze.pos,
				nodelist=[M.goal], node_color='r',
				node_shape='h') 

	else:
		raise DrawException

	ax = pl.gca()

	ax.set_aspect('equal')

def draw_path(trial):
	path = [s.loc for s in trial.steps]
	path.append(trial.step.goal)

	edges = []

	for i in np.arange(len(path) -1):
		edges.append((path[i], path[i+1]))

	edges.append((path[-1], trial.step.goal))

	draw(trial.steps[0])

	nx.draw_networkx_edges(trial.step.maze, pos=trial.step.maze.pos,
			edgelist=edges, edge_color='g', width=3)

