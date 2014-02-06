from mazex.mazes import SquareMaze

class Step:
	def __init__(self, maze, current_loc, goal_loc, heading=None):
		"""Initializes the Step.
		
		Parameters:
		-----------		
		heading : string
			this is the direction the agent is heading toward. It
			should be derived from the last agent's position by some
			external method. It can assume values '^', '>', '<',
			'v'
		"""
		if isinstance(maze, SquareMaze):
			self.maze = maze

		self.current_loc = current_loc
		self.goal_loc = goal_loc
		self.heading = heading


class Trial:
	def __init__(self, maze, start, goal, agent, max_steps=0, goal_reached=True):
		self.maze = maze
		self.goal = goal
		self.start = start
		self.loc = start
		self.steps = []
		self.step = Step(maze, start, goal, agent.heading)
		self.max_steps = max_steps
		self.goal_reached = goal_reached

	def __iter__(self):
		return self

	def __next__(self):
		if self._termination_condition():
			raise StopIteration

		else:
			next_pos = self.agent.choose(self.step)	
			self.step = Step(self.maze, next_pos, self.goal,
				self.agent.heading)
			return self.step	

	def solve(self, agent):
		loc = self.loc
		path = []
		while loc != goal:
			next_path_seg = agent.next_path_seg(loc, goal)
			path.append(next_path_seg)
			loc = next_path_seg[-1]

	def _termination_condition(self):
		"""This is a helper function for the simmulation methods.

		Parameters:
		-----------
		max_steps : int
			Maximum number of steps the agent has until the Trial
			is over. 0 means number of steps is not used as
			termination criterion

		goal_reached : bool
			In the most usual scenario, goal_reached=True, means
			the trial is over when the agent has reached the goal
		"""
		if max_steps and len(self.steps) >= max_steps:
			return True

		elif goal_reached and self.loc == self.goal:
			return True

		else:
			return False


class Session:
	def __init__(self, agent, trials):
		"""Session initiation method.

		Parameters:
		-----------
		agent : Agent
			Agent that will solve the trials of the session

		trials : list
			List of trials to be presented to the Agent
		"""

		self.agent = agent
		self.trials = trials
		self.trial = trials[0]
		self.step = self.trial.step

	def __iter__(self):
		return self	

	def __next__(self):
		if self.trial == self.trials[-1]:
			raise StopIteration
		
		else:
			self.step = self.trial.__next__()
			return self.step

