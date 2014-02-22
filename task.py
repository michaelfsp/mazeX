from mazex.mazes import SquareMaze
from mazex.exceptions import *
from mazex.utils import check_elements_equal
import random
import numpy as np

def choose(probs):
    """Choose one edge using softmax decision rule based on the
    edges values

    Output:
    -------
    index : int
        Index of the chosen edge
    """
    # This function is 5X faster than numpy.random.choice!!
    cum_prob = np.cumsum(probs)

    random_number = random.uniform(0, cum_prob[-1])

    return np.argmax(cum_prob >= random_number)

class Step:
    def __init__(self, maze, loc, goal, heading=None, previous=None):
        """Initializes the Step.
        
        Parameters:
        -----------     
        heading : string
            this is the direction the agent is heading toward. It
            should be derived from the last agent's position by some
            external method. It can assume values '^', '>', '<',
            'v'

        previous : Step
            Previous step.
        """
        if isinstance(maze, SquareMaze):
            self.maze = maze
        else:
            raise InvalidMaze

        self.loc = loc
        self.goal = goal
        self.heading = heading
        self.values = []
        self.probs = []
        self.previous = previous

    @property
    def available_nodes(self):
        return self.maze.neighbors(self.loc)

    @property
    def path(self):
        """Calculates  path until current step by iterating over the previous
        step's locations.
        """
        path = [self.loc]
        previous_step = self.previous

        while previous_step:
            path.insert(0, previous_step.loc)
            previous_step = previous_step.previous
        return path


class Trial:
    def __init__(self, steps, agent, max_steps=0, goal_reached=True):
        self.steps = steps
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
        return self.step.path

    @property
    def maze(self):
        """If all steps have the same maze, return that maze, else return a
        list of mazes. This might not be after some refactoring to be done in 
        a future incarnation.
        """
        mazes = [s.maze for s in self.steps]
        if check_elements_equal(mazes):
            return mazes[0]
        else:
            return mazes

    @property
    def goal(self):
        """If all steps have the same goal, return that goal, else return a
        list of goals. This might not be after some refactoring to be done in 
        a future incarnation.
        """
        goals = [s.goal for s in self.steps]
        if check_elements_equal(goals):
            return goals[0]
        else:
            return goals

    def go_to_next_step(self):
        self.step.values = self.agent.evaluate(
            self.step.available_nodes, self.step.goal)
        
        self.step.probs = self.agent.choice_probs(self.step.values)

        next_node = self.step.available_nodes[choose(self.step.probs)]

        self.agent.loc = next_node

        self.steps.append(Step(self.step.maze, self.agent.loc,
            self.step.goal, self.agent.heading, self.step))

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

        elif self.goal_reached and self.step.loc == self.step.goal:
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

        new_goal = random.choice(new_trial.maze.nodes)

        new_trial.goal = new_goal
        new_trial = Trial(self.agent.loc, new_goal, self.agent,
                self.Trial.max_steps, self.Trial.goal_reached)
        self.trials.append(new_trial)

