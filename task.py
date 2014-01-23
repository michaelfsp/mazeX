class Trial:
	def __init__(self, maze, goal, start):
		self.maze = maze
		self.goal = goal
		self.start = start
		self.loc = start
		self.steps = []

	def check_path_seg(self, loc, path_seg):
		"""Check if the path the agent wants to take is a valid one,
		i.e. doesn't violate the maze configuration
		"""

	def solve(self, agent):
		loc = self.loc
		path = []
		while loc != goal:
			next_path_seg = agent.next_path_seg(loc, goal)
			path.append(next_path_seg)
			loc = next_path_seg[-1]

class Step:
	def __init__(self):
		pass


class Session:
	def __init__(self, start, agent, trials=None):
		self.loc = start
		self.data = None

	def iterate(Session, Agent):
		pass


