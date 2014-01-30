"""
Visually represent mazes and trials
"""

import networkx as nx
from mazex.mazes import *
from exceptions import *

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
		nx.draw(M, pos=M.pos, ax=ax, hold=hold, **kwds)
	else:
		raise DrawException

