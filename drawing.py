"""
Visually represent mazes and trials
"""

import networkx as nx
from mazex.mazes import SquareMaze
from mazex.exceptions import *
from mazex.task import *

import pdb

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
	if isinstance(M, SquareMaze):
		nx.draw(M, pos=M.pos, ax=ax, hold=hold, node_color='gray', **kwds)
	
	elif isinstance(M, Step):
		# First we obtain a list of the nodes that are neither current
		# nor goal location
		normal_nodes = M.maze.nodes()
		# And then we remove current and goal location nodes because
		# those will be plotted differently
		normal_nodes.remove(M.current_loc)
		normal_nodes.remove(M.goal_loc)
		nx.draw(M.maze, pos=M.maze.pos, ax=ax, hold=True, node_color='gray',
				nodelist=normal_nodes, kwds=kwds)
		# Then we draw the nodes correspoding to current location
		#  we use the heading to shape the arrow, if present
		if M.heading:
			shape = M.heading
		else:
			shape = '8'
		nx.draw_networkx_nodes(M.maze, pos=M.maze.pos,
				nodelist=[M.current_loc], node_color='b',
				node_shape=shape)
		# ... and goal location 
		nx.draw_networkx_nodes(M.maze, pos=M.maze.pos,
				nodelist=[M.goal_loc], node_color='r',
				node_shape='h') 


	else:
		raise DrawException

