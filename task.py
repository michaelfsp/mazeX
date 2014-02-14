from mazex.mazes import SquareMaze
import copy
from random import choice

class Step:
    def __init__(self, maze, loc, goal, heading=None):
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

        self.loc = loc
        self.goal = goal
        self.heading = heading


class Trial:
    def __init__(self, steps, agent, max_steps=0, goal_reached=True):
        self.steps = [steps]

        self.agent = agent

        # Trial termination conditions
        self.max_steps = max_steps
        self.goal_reached = goal_reached
    
    @property
    def loc(self):
        return self.agent.loc

    @property
    def step(self):
        return self.steps[-1]

    @property
    def path(self):
        return [s.loc for s in self.steps]

#   def __iter__(self):
#       return self
#
#   def __next__(self):
#       if self._termination_condition():
#           raise StopIteration

#       else:
#           next_pos = self.agent.choose(self.step) 
#           self.step = Step(self.maze, next_pos, self.goal,
#               self.agent.heading)
#           return self.step    

    def go_to_next_step(self):
        available_nodes = self.step.maze.neighbors(self.agent.loc)

        values = self.agent.evaluate(available_nodes, self.step.goal)

        next_node = available_nodes[self.agent.choose_edge_index(
            values)]

        self.agent.loc = next_node

        self.steps.append(Step(copy.deepcopy(self.step.maze),
            self.agent.loc, self.step.goal, self.agent.heading))

    def solve(self):
        while not self.termination_condition():
            self.go_to_next_step()
        
    def termination_condition(self):
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
        if self.max_steps and len(self.steps) >= self.max_steps:
            return True

        elif self.goal_reached and self.agent.loc == self.step.goal:
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

    @property
    def trial(self):
        return self.trials[-1]

#   def __iter__(self):
#       return self 

#   def __next__(self):
#       if self.trial == self.trials[-1]:
#           raise StopIteration
#       
#       else:
#           self.step = self.trial.__next__()
#           return self.step

class SessionWithRandomTrials(Session):
    def __init__(cls, agent, trials, trial_limit):
        """

        Parameters:
        -----------
        number : int
            Total number or trials
        """
        super(SessionWithRandomTrials, self).__init__(agent, trials)
        self.trial_limit = trial_limit

    def go_to_new_random_trial(self):
        new_trial = copy.deepcopy(self.trial)

        new_goal = choice(new_trial.maze.nodes)

        new_trial.goal = new_goal
        new_trial = Trial(self.agent.loc, new_goal, self.agent,
                self.Trial.max_steps, self.Trial.goal_reached)
        self.trials.append(new_trial)

