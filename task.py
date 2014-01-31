from mazex.mazes import SquareMaze

class Trial:
	def __init__(self, maze, goal, start):
		self.maze = maze
		self.goal = goal
		self.start = start
		self.loc = start
		self.steps = []

	def solve(self, agent):
		loc = self.loc
		path = []
		while loc != goal:
			next_path_seg = agent.next_path_seg(loc, goal)
			path.append(next_path_seg)
			loc = next_path_seg[-1]

class Step:
	def __init__(self, maze, current_loc, goal_loc, heading=None):
		"""

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

class Session:
	def __init__(self, start, agent, trials=None):
		self.loc = start
		self.data = None

	def iterate(Session, Agent):
		pass


