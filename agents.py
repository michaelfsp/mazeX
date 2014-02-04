import numpy as np

def softmax(beta, values):
	"""A Softmax function.

	Parameters:
	-----------
	beta : float, [0, inf+]
		The inverse temperature parameter of the softmax function. A
		very high value causes greedy action, whereas a low value very
		stochastic action selection (more exploration).

	values : list or numpy.array
		The list of values of the actions available to the agent.
	"""
	# Reduce dimensionality of values to 1, if it is an array, else
	# transform it into an array
	try:
		values.ravel() #will fail if it is not a numpy array
	except AttributeError:
		values = np.array([values])

	#this is the calculation of the numerator of the equation
	exp_elements = np.exp(beta*values)

	#this is the calculation of the denominator of the equation
	denominator = np.sum(exp_elements)	

	#and here we compute the action probabilities
	probs = exp_elements/denominator 	

	return probs

def edge_angle(edge_vec, loc_coord, goal_coord):
	"""A common function that determines the angle between a given edge and
	the goal.
	
	Parameters:
	-----------
	edge_vec : list
		List of edge vector x,y lengths with respect to the agent

	loc_coord : tuple or list with 2 elements
		The x,y coordinates of the current location.	
	
	goal_coord : tuple or list with 2 elements
		The x,y coordinates of the goal location.
	"""
	loc_coord = np.array(loc_coord)
	goal_coord = np.array(goal_coord)
	
	#reference goal location with respect to the agent's current position
	goal_vec = goal_coord - loc_coord	
	
	return np.cross(edge_vec, goal_vec)/(np.linalg.norm(edge_vec)*
			np.linalg.norm(goal_vec)) 


class Agent:
	def __init__(self):
		self._step_cache = []

	def edge_angle_from_heading(self, loc, heading, edge):
		"""
		"""	
	
class DirectedRandomAgent(Agent):
	def __init__(self, conc, direction=0):
		"""Creates an agent that follows semi-random motion.
		The choice probabilities of this agent follow
		the circular von Mises distribution.
		
		Parameters:
		-----------
		conc : float
			Concentration parameter for the von Mises distribution.

		direction : float , optional, [0, 2*pi]
			Direction (location parameter for the von Mises
			distribution in radians off the center.
		"""
		self.direction = direction
		self.conc = conc

	def edge_value(self, edges):
		"""Calculates the value of each edge available to the agent.

		Parameters:
		-----------

		edges : list
			List of edges to evaluate
		"""
	def next_path_seg(self, loc, goal, mazemodel, memory=None):
		"""Calculates the position the agent wants to visit next.
		"""

class DirectionalHeuristicAgent(Agent):
	def __init__(self, kappa, gamma, beta):
		"""Creates an agent that follows a directional heuristic.

		Parameters:
		-----------
		kappa : float
			Concentration parameter for the von Mises distribution.

		gamma : float
			Cost parameter for recently performed actions
	
