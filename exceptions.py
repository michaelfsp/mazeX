import networkx as nx

class MazeXException(nx.exception.NetworkXException):
	"""Base class for exceptions in Maze."""
	pass

class DrawException(MazeXException):
	def __init__(self):
		self.message = "Don't know how to draw this object"
