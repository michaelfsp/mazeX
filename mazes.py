import numpy as np
import networkx as nx
import random

import pdb

class SquareMaze(nx.classes.graph.Graph):
	"""Generic class of for square mazes.
	"""
	def __init__(self, size):
		"""Generates a square maze of given size and no connections.
		Total number of nodes will size**2.
		"""

		# TODO: Check whether the graph instance is a possible maze
		# configuration
		G = nx.Graph()
		
		for i in np.ndindex((size, size)):
			G.add_node(i, attr_dict={'pos':i})

		super(SquareMaze, self).__init__(G)
		self.maze_size = size
	
	@property
	def pos(self):
		return nx.get_node_attributes(self, 'pos')

	@property
	def possible_edges(self):
		"""List possible connections for given node.

		Parameters:
		-----------
		node : node
			Node to check for possible edges
		"""
		possible_edges = []

		for n in self.nodes():
			for nbr in self.adjacent_neighbors(n):
				if nbr[0]>n[0] or nbr[1]>n[1]:
					possible_edges.append((n, nbr))
		return possible_edges

	def adjacent_neighbors(self, n):
		"""Returns a list of nodes located in the immediate physical
		vicinity of node n. It does not include "diagonal" neighbours.

		Parameters
		----------
		n : node
			A node in the graph

		Returns
		-------
		nlist : list
			A list of nodes physically adjacent to n
		"""
		adj_list = ((n[0]-1, n[1]), (n[0], n[1]-1), (n[0]+1, n[1]),
				(n[0], n[1]+1))
		adj_neighbors = []	

		for n in adj_list:
			if n in self.nodes():
				adj_neighbors.append(n)

		return adj_neighbors


class RandomSquareMaze(SquareMaze):
	"""Generator of SuqareMaze with random connections.
	"""
	def __new__(self, size, connectedness):
		"""Generates a SquareMaze with random connectivity of
		given connectedness.

		Parameters:
		-----------
		size : int
			Maze size. Number of nodes will be size*size.

		connectedness : float [0,1]
			Fraction of all possible connections that actually exist.
		"""
		maze = SquareMaze(size)

		possible_edges = maze.possible_edges

		edges_to_add = random.sample(possible_edges, round(len(
			possible_edges)*connectedness)) 

		maze.add_edges_from(edges_to_add)

		return maze

