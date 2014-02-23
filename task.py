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
    def __init__(self, loc, heading=None, previous=None, trial=None):
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
        self.loc = loc
        self.heading = heading
        self.values = []
        self.probs = []
        self.previous = previous
        self.trial = trial

    @property
    def available_nodes(self):
        try:
            return self.trial.maze.neighbors(self.loc)
        except AttributeError:
            raise NotBoundToParent

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
    def __init__(self, steps, maze, goal, max_steps=0, goal_reached=True,
            session=None):
        self.steps = steps
        self._current_step = len(steps)-1

        if isinstance(maze, SquareMaze):
            self.maze = maze
        else:
            raise InvalidMaze

        self.goal = goal
 
        # Trial termination conditions
        self.max_steps = max_steps
        self.goal_reached = goal_reached
        self.session = session

    
    @property
    def step(self):
        return self.steps[self._current_step]

    @property
    def path(self):
        return self.step.path

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

        elif self.goal_reached and self.step.loc == self.goal:
            return True

        else:
            return False


class SimulatedTrial(Trial):
    def __init__(self, steps, maze, goal, session, max_steps=0,
            goal_reached=True):
        super(SimulatedTrial, self).__init__(steps, maze, goal, max_steps,
                goal_reached, session)

    def go_to_next_step(self):
        self.step.values = self.session.agent.evaluate(
            self.step.available_nodes, self.step.goal)
        
        self.step.probs = self.session.agent.choice_probs(self.step.values)

        next_node = self.step.available_nodes[choose(self.step.probs)]

        self.agent.loc = next_node

        self.steps.append(Step(self.step.maze, self.agent.loc,
            self.step.goal, self.agent.heading, self.step))

    def solve(self):
        while not self.termination_condition():
            self.go_to_next_step()
 

class Session:
    def __init__(self, trials):
        """Session initiation method.

        Parameters:
        -----------
        agent : Agent
            Agent that will solve the trials of the session

        trials : list
            List of trials to be presented to the Agent
        """
        self.trials = trials
        self._current_trial = len(trials)-1

    @property
    def trial(self):
        return self.trials[self._current_trial]

class SimulatedSession(Session):
    def __init__(self, trials, agent):
        super(SimulatedSession, self).__init__(trials)
        self._current_trial = 0
        self.agent = agent

    def next_trial(self):
        return self.trials[self._current_trial+1]

    def run(self):
        """Runs the Session's trials sequentially. Note that it will solve
        it again even if it has already been solved.
        """
        self.current_trial = 0
        
        # This statement only checks if the pointer self.current_trial is the
        # same as the index of the current trial, and not for object identity
        # (==) because it will allow for trials to be created on the fly.
        while not self.current_trial == len(self.trial)-1:
            self.trial.solve()
            self.current_trial += 1

        # solve the last trial
        self.trial.solve()


class SimulatedSessionWithRandomGoals(SimulatedSession):
    def __init__(self, agent, n_trials, first_trial, gen_func):
        """

        Parameters:
        -----------
        number : int
            Total number or trials
        """
        trials = [first_trial] + (n_trials-1)*[None]
        super(SessionWithRandomTrials, self).__init__(agent, trials)
        self.gen_func = gen_func

    def go_to_new_random_trial(self):
        new_trial = copy.deepcopy(self.trial)

        new_goal = random.choice(new_trial.maze.nodes)

        new_trial.goal = new_goal
        new_trial = Trial(self.agent.loc, new_goal, self.agent,
                self.Trial.max_steps, self.Trial.goal_reached)
        self.trials.append(new_trial)

