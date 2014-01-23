import numpy as np
import networkx as nx

class SquareMaze(nx.classes.graph.Graph):
	
	def __init__(self, size):
		
		# TODO: Check whether the graph instance is a possible maze
		# configuration
		G = nx.Graph()
		
		for i in np.ndindex((size, size)):
			G.add_node(G.number_of_nodes(), attr_dict={'pos':i})

		super(SquareMaze, self).__init__(G)
	
	@property
	def pos(self):
		return nx.get_node_attributes(self, 'pos')

	@property
	def possible_edges(self):
		#TODO: to be implemented	
		pass
	
	def adjacent_neighbors(self, n):
		"""Returns a list of nodes located in the immediate physical
		vicinity of node n.

		Parameters
		----------
		n : node
			A node in the graph

		Returns
		-------
		nlist : list
			A list of nodes physically adjacent to n
		"""
		adj_list = []
		nodes = [c for c in self.pos.items()]

		for n in nodes:
			pass	
			
		return adj_list
		

