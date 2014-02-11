"""
Visually represent mazes and trials
"""
import numpy as np
import pylab as pl
import networkx as nx
from mazex.mazes import SquareMaze
from mazex.exceptions import *
from mazex.task import Step, Trial

import pdb

def draw_maze(M, ax=None, hold=None, **kwds):
	"""Draw a maze configuration.
	"""
	if not isinstance(M, SquareMaze):
		raise DrawException
	if not ax:
		ax = pl.gca()
	# Makes sure the drawings are square and not rectangular
	ax.set_aspect('equal')

	# Passing kwds from function to function causes kwds to look like
	# kwds = {'kwds': ...}
	# Here we try to unzip that
	try:
		kwds = kwds['kwds']
	except KeyError:
		pass

	#pdb.set_trace()

	kwds['node_color'] = 'gray'
 	
	if isinstance(M, SquareMaze):
		nx.draw(M, pos=M.pos, ax=ax, hold=hold,	**kwds)


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
	normal_nodes = S.maze.nodes()
	# And then we remove current and goal location nodes because
	# those will be plotted differently
	normal_nodes.remove(S.loc)
	if S.goal != S.loc:
		normal_nodes.remove(S.goal)

	# Unzip kwds in case it is necessary from another function calling it
	try:
		kwds = kwds['kwds']
	except KeyError:
		pass

	#pdb.set_trace()	

	kwds['nodelist'] = normal_nodes
	draw_maze(S.maze, ax=ax, hold=True, kwds=kwds)
	# Then we draw the nodes correspoding to current location
	#  we use the heading to shape the arrow, if present
	if S.heading:
		shape = S.heading
	else:
		shape = '8'
	nx.draw_networkx_nodes(S.maze, pos=S.maze.pos,
			nodelist=[S.loc], node_color='b',
			node_shape=shape)
	# ... and goal location 
	nx.draw_networkx_nodes(S.maze, pos=S.maze.pos,
			nodelist=[S.goal], node_color='r',
			node_shape='h') 

def draw_trial(T, ax=None, hold=None, **kwds):
	"""Draw a Trial. Will represent the taken path if that is available.
	"""
	edges = []

	for i in range(len(T.path)-1):
		edges.append((T.path[i], T.path[i+1]))

	#pdb.set_trace()
	draw_step(T.steps[0], ax=ax, hold=True, kwds=kwds)

	nx.draw_networkx_edges(T.step.maze, pos=T.step.maze.pos,
			edgelist=edges, edge_color='g', width=4)


def draw(M, ax=None, hold=None, **kwds):
	"""Meta function. Calls the write draw function.
	"""
	if isinstance(M, SquareMaze):
		draw_maze(M, ax=ax, hold=hold, kwds=kwds)
	
	elif isinstance(M, Step):
		draw_step(M, ax=ax, hold=hold, kwds=kwds)
	
	elif isinstance(M, Trial):
		draw_trial(M, ax=ax, hold=hold, kwds=kwds)
	else:
		raise DrawException
