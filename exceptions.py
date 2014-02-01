import networkx as nx

class MazeXException(nx.exception.NetworkXException):
	"""Base class for exceptions in Maze."""
	pass

class DrawException(MazeXException):
	def __str__(self):
		return "Don't know how to draw this object."

class NumberEdgesException(MazeXException):
	def __str__(self):
		return "Invalid number of edges. Please review your input."
