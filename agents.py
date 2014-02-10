import numpy as np
import random
from scipy.stats import vonmises
from collections import OrderedDict

from mazex.exceptions import *

def _get_loc(self):
	return self._loc

def _set_loc(self, node):
	if assert_nodes_adjacent(self.loc, node):
		if node[1] > self.loc[1]:
			self.heading = '^'
		elif node[1] < self.loc[1]:
			self.heading = 'v'
		elif node[0] > self.loc[0]:
			self.heading = '>'
		elif node[0] < self.loc[0]:
			self.heading = '<'
		self._loc = node

def angle_between_nodes(loc, node1, node2):
	"""A common function that determines the angle between a given edge and
	the goal.

	Parameters:
	-----------
	node : node or tuple
		Node on the maze we want to evaluate the angle of
	
	loc : node or tuple
		The current node
	
	goal : node or turple
		The node of the ,y coordinates of the goal location.
	"""
	node1_vec = np.array(node1) - np.array(loc)
	node2_vec = np.array(node2) - np.array(loc)
	
	return np.arccos(np.dot(node1_vec, node2_vec) / (np.linalg.norm(
		node1_vec)*np.linalg.norm(node2_vec)))

def assert_is_node(node):
	# test the node follows our node naming convention
	if not len(node) == 2:
		raise InvalidNode
	elif not type(node[0]) is int or not type(node[1]) is int:
		raise InvalidNode
	elif node[0] < 0 or node[1] < 0:
		raise InvalidNode
	else:
		return True

def assert_nodes_adjacent(node1, node2):
	"""Determine whether node1 and node2 are in the immediate vicinity,
	simply using Euclidian distance. Note that it does not check whether
	this node exists on the maze nor whether the path between node1 and
	node2 is free.
	Note that transition to same node (staying in place) is not allowed.
	"""
	assert_is_node(node1)
	assert_is_node(node2)
	if np.sqrt((node2[0] - node1[0])**2 + (node2[1] - node1[1])**2) == 1:
		return True
	else:
		raise IlegalStep

def softmax(beta, values):
	"""A Softmax function.

	Parameters:
	-----------
	beta : float, [0+, inf+]
		The inverse temperature parameter of the softmax function. A
		very high value causes greedy action, whereas a low value very
		stochastic action selection (more exploration).

	values : list or numpy.array
		The list of values of the actions available to the agent.
	"""
	# Reduce dimensionality of values to 1, if it is an array, else
	# transform it into an array
	try:
		values = values.ravel() #will fail if it is not a numpy array
	except AttributeError:
		values = np.array(values)

	if beta == float('inf'):
		probs = np.zeros(len(values))
		probs[np.argmax(values)] = 1

	elif beta == float('-inf'):
		probs = np.ones(len(values)) * 1/(len(values))
	
	else:
		#this is the calculation of the numerator of the equation
		exp_elements = np.exp(beta*values)

		#this is the calculation of the denominator of the equation
		denominator = np.sum(exp_elements)	

		#and hnndore we compute the action probabilities
		probs = exp_elements/denominator 	

	return probs


class Agent:
	def __init__(self, loc, beta, heading=None):
		"""Parameters:
		-----------
		beta : float, [0+, inf+]
			The inverse temperature parameter of the softmax function. A
			very high value causes greedy action, whereas a low value very
			stochastic action selection (more exploration).

		heading : string
			this is the direction the agent is heading toward. It
			should be derived from the last agent's position by some
			external method. It can assume values '^', '>', '<',
			'v'
		"""
		self._step_cache = [] #for an agent that plans multiple steps
		# at once
		self.beta = beta
		self.heading = heading
		# we will use the property decorator to turn self.loc into
		# a smart property that agents with position memory will make
		# use of
		if assert_is_node(loc):
			self._loc = loc 
	
	loc = property(_get_loc, _set_loc)

	def __reset_loc_to__(self, loc):
		"""Since setting the agent's location for proximity checks for
		adjacency and complains if the nodes are not adjacent, this
		method can be called to force reset location to a specific one.

		Parameters:
		-----------

		loc : node
			New location
		"""
		if assert_is_node(loc):
			self._loc = loc
			
	def angle_to_goal(self, node, goal):
		"""A common function that determines the angle between a given edge and
		the goal.
		
		Parameters:
		-----------
		node : node or tuple
			Node on the maze we want to evaluate the angle of

		goal : node or turple
			The node of the ,y coordinates of the goal location.
	
		Output:
		-------
		angle : float
			In radians
		"""
		
		return angle_between_nodes(self.loc, node, goal)

	def angle_to_heading(self, node):
		"""Determines the angle formed between the vector pointing to
		the the node 'node' and the vector pointing toward heading
		direction.

		Parameters:
		-----------
		node : node or tuple
			The node for which direction the angle is to be
			calculated.

		Output:
		-------
		angle : float
			In radians
		"""	
		#TODO: verify that this is the way coordinates are specified
		
		"""We will create a virtual heading_node which corresponds to
		the node the agent is facing in order to facilitate
		calculation by reusing the method anglet_between_nodes().
		However do note that heading_note might have coordinates that 
		are outside of the maze. This should still be OK.
		"""
		if not self.heading:
			raise NoHeading
		elif self.heading == '^':
			heading_node = (self.loc[0], self.loc[1]+1)
		elif self.heading == 'v':
			heading_node = (self.loc[0], self.loc[1]-1)
		elif self.heading == '<':
			heading_node = (self.loc[0]-1, self.loc[1])
		elif self.heading == '>':
			heading_node = (self.loc[0]+1, self.loc[1])
		else:
			raise InvalidHeading

		return angle_between_nodes(self.loc, node, heading_node)

	def choose_edge_index(self, values):
		"""Choose one edge using softmax decision rule based on the
		edges values

		Output:
		-------
		index : int
			Index of the chosen edge
		"""
		choice_prob = softmax(self.beta, values)

		cum_prob = np.cumsum(choice_prob)
		
		random_number = random.uniform(0, cum_prob[-1])

		return np.argmax(cum_prob >= random_number)

		
class HeuristicAgent(Agent):
	def __init__(self, loc, beta, kappa, xi, heading=None):
		"""Creates an agent that follows a directional heuristic.

		Parameters:
		-----------
		kappa : float
			Concentration parameter for the von Mises distribution.

		xi : float
			Cost parameter for recently performed actions. From
			the greek xanazõ, to relive ;)
		"""
		super(HeuristicAgent, self).__init__(loc, beta, heading)
		self.kappa = kappa
		self.xi = xi
		# we are using an OrderedDict to store the last visited edges
		# because it will make it easier to make the agent forget the
		# older entries by using self.edge_visit_cache.popitem(
		# last=False)
		self.edge_visit_cache = OrderedDict()
		self.step_count = 0

	def set_loc(self, node):
		old_loc = self.loc
		new_loc = node

		_set_loc(self, new_loc)

		# I am calling it a directed edge because our edges normally
		# have a specific node order specification but here we care
		# about node order in that it specifies a direction
		directed_edge = (old_loc, new_loc)

		if directed_edge in self.edge_visit_cache:
			# popitem() deletes the entry AND spits an output we
			# want to silence with the ;
			# it's necessary to delete the item with popitem()
			# and readd it because only then it assumes the
			# desired order in the ordered dictionary
			self.edge_visit_cache.popitem(directed_edge);

		self.edge_visit_cache[directed_edge] = self.step_count

		self.step_count += 1
	
	loc = property(_get_loc, set_loc)

	def evaluate(self, nodeslist, goal):
		"""Returns a vector with the edge values of each edge given as
		input.
		
		Parameters:
		-----------
		nodeslist : list of np.array, dim=1
			Nodes for which the transition to is to be evaluated.

		goal : node
			Location of goal.
		"""
		# this little trick makes the method work even if not a list
		# but a single node is passed
		try:
			assert_is_node(nodeslist)
			nodeslist = [nodeslist]
		except InvalidNode:
			pass

		angles = [angle_between_nodes(
			self.loc, goal, n) for n in nodeslist] 
		
		values = np.array(vonmises.pdf(angles, kappa=self.kappa))

		edgeslist = [(self.loc, n) for n in nodeslist]
		
		step_count_at_last_visit = []

		for e in edgeslist:
			if e in self.edge_visit_cache:
				step_count_at_last_visit.append(
						self.edge_visit_cache[e])
			else:
				# nodes that have never been visited we will
				# assume they have been visited in "step -1"
				# the value -1 doesn't play any actual role
				# as these steps will have no penalization
				step_count_at_last_visit.append(-1)

		# This is the number of edges for which there is a visit
		# record
		
		step_count_at_last_visit = np.array(step_count_at_last_visit)

		n_visited_edges = np.count_nonzero(
				step_count_at_last_visit > 0)

		penalties = np.zeros(len(edgeslist))


		while n_visited_edges > 0:
			max_step_count = max(step_count_at_last_visit)
			index_max_step_count = np.where(step_count_at_last_visit
					== max_step_count)

			penalties[index_max_step_count] = n_visited_edges * self.xi
			# very dumb way of having max() finding the second
			# max in the next iteration
			step_count_at_last_visit[index_max_step_count] = -1
			
			n_visited_edges -= 1

		values -= penalties

		return values
		
class DirectedRandomAgent(Agent):
	def __init__(self, loc, beta, conc, heading=None, direction=0):
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
		super(DirectedRandomAgent, self).__init__(loc, beta, heading)
		self.direction = direction
		self.conc = conc

	def edge_value(self, edges):
		"""Calculates the value of each edge available to the agent.

		Parameters:
		-----------

		edges : list
			List of edges to evaluate
		"""
		pass

	def next_path_seg(self, loc, goal, mazemodel, memory=None):
		"""Calculates the position the agent wants to visit next.
		"""
		pass
