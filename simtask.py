from mazex.mazes import SquareMaze
from mazex.exceptions import *
from mazex.utils import check_elements_equal
from mazex.datamodel import *
import random
import numpy as np

from numba import jit

@jit(nopython=True, nogil=True)
def choose(probs):
    #if np.any(probs>1) or np.any(probs<0):
    #    print(probs)
    """Choose one option based on given decision probabilities.

    Output:
    -------
    index : int
        Index of the picked choice
    """
    # This function is 5X faster than numpy.random.choice!!
    #
    # Alternatives
    # return np.random.multinomial(1, probs).argmax() #Faster alternative
    #
    # return np.random.choice(len(probs), p=probs) #Slower alternative

    cum_prob = np.cumsum(probs)

    random_number = random.uniform(0, cum_prob[-1])

    return np.argmax(cum_prob >= random_number)

class SimStep(Step):
    def __init__(self, loc, trial=None, previous=None):
        # trial=None is only here to allow the creation of the initial step
        # for the initial step this will be altered later by
        # SimTrial.__init__() a bit of a hugly hack but will do it for now
        super(SimStep, self).__init__(loc, trial, previous = previous)
        
        
class SimTrial(Trial):
    def __init__(self, maze, goal, initial_step, agent, max_steps=0,
            goal_reached=True):
        super(SimTrial, self).__init__(maze, goal, max_steps,
                goal_reached)
        self.agent = agent
        self.steps = [initial_step]

        initial_step.trial = self

    def go_to_next_step(self):
        #TODO: do not calculate penalties once in order to store them and then
        #  once again when evaluate is called.
        self.step.penalties = self.agent.penalty_factor(
                self.step.available_nodes)

        self.step.values = self.agent.evaluate(
            self.step.available_nodes, self.goal)
        
        self.step.probs = self.agent.choice_probs(self.step.values)

        
        next_node = self.step.available_nodes[choose(self.step.probs)]

        self.agent.loc = next_node
        
        current_step = self.step

        self.steps.append(SimStep(next_node, self, previous = current_step))

    def solve(self):
        while not self.termination_condition():
            self.go_to_next_step()

class SimSession(Session):
    def __init__(self):
        """
        Parameters:
        -----------
        agent : Agent
            Agent that will solve the trials of the session
        """
        super(SimSession, self).__init__()

    def run(self):
        """Runs the Session's trials sequentially. Note that it will solve
        it again even if it has already been solved.
        """
        self.current_trial_index = 0
      
        # This statement only checks if the pointer self.current_trial is the
        # same as the index of the current trial, and not for object identity
        # (==) because it will allow for trials to be created on the fly.
        while not self.current_trial_index == len(self.trials)-1:
            self.trial.solve()
            self.current_trial_index += 1

        # solve the last trial
        self.trial.solve()

class SimSessionRandomGoal(SimSession):
    def __init__(self, maze, agent, n_trials, initial_step=None):
        """
        Parameters:
        -----------
        number : int
            Total number or trials
        """
        if not initial_step:
            initial_step = SimStep(loc, trial, previous=None)
        initial_trial = SimTrial(maze, goal, initial_step, agent, max_steps=0,
            goal_reached=True)
        super(SimSessionRandomGoal, self).__init__()
        self.trials.append(initial_trial)

    def run(self):
        while not len(self.trials) == n_trials:
            maze_nodes = self.maze.nodes
            # exclude current location from the list from which the new goal
            # is picked
            maze_nodes.remove(self.trial.step.loc)
            new_goal = random.choice(maze_nodes)

            initial_step = copy.deepcopy(self.trial.step)
            self.trials.append(SimTrial(maze, new_goal, initial_step, agent,
                max_steps=0, goal_reached=True))

            super(SimSessionRandomGoal, self).run()
