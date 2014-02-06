from networkx.exception import NetworkXException 

class MazeXException(NetworkXException):
	"""Base class for exceptions in Maze."""
	pass

class DrawException(MazeXException):
	def __str__(self):
		return "Don't know how to draw this object."

class NumberEdgesException(MazeXException):
	def __str__(self):
		return "Invalid number of edges. Please review your input."

class NoHeading(MazeXException):
	def __str__(self):
		return "No Heading is specified."

class InvalidHeading(MazeXException):
	def __str__(self):
		return ("The heading specified is incorrect! Please use only "
		"'^', '>', 'v', '<'.")

class IlegalStep(MazeXException):
	def __str__(self):
		return ("Sorry about that but I can't quite teleport myself"
		"... Next time move me to an adjacent node please!")

class InvalidNode(MazeXException):
	def __str__(self):
		return ("Invalid node specification. Nodes are of the format "
				"(x, y).")
