from mazex.mazes import SquareMaze
from mazex.exceptions import *
from mazex.utils import check_elements_equal
from mazex.datamodel import *
import random
import inspect
import numpy as np

from numba import jit
from tqdm import tqdm

import ray

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
    def __init__(self, **kwargs):
        """TODO: Specify the **kwargs.
        """
        super(SimSession, self).__init__(**kwargs)

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


class SimSessionGivenGoals(SimSession):
    def __init__(self, agent, maze, goals, max_steps=0, max_total_steps=0,
                 loc_start=(0,0), continue_from_goal_after_max_steps=False,
                 **kwargs):
        """
        Parameters:
        -----------
        agent : Agent
            Agent that will solve the trials of the session
        """
        self.agent = agent
        #TODO: change sim_maze to maze and refactor Session.maze
        self.sim_maze = maze
        self.goals = goals
        self.max_steps = max_steps
        self.max_total_steps = max_total_steps
        self.continue_from_goal_after_max_steps = continue_from_goal_after_max_steps
        self.total_steps = 0
        self.loc_start = loc_start
        self.trials = []

        super(SimSessionGivenGoals, self).__init__(**kwargs)


    def run(self, verbose=False, progress_bar=True):
        """Runs the Session's trials sequentially. Note that it will solve
        it again even if it has already been solved.
        """
        self.agent.loc = self.loc_start
        trials = []
        if self.max_total_steps > 0:
            termination_condition = lambda self: self.total_steps >= self.max_total_steps
        else:
            termination_condition = lambda self: False
        
        if progress_bar:
            for goal in tqdm(self.goals, leave=False):
                if not termination_condition(self):
                    initial_step = SimStep(self.agent.loc)
                    self.agent.node_visit_cache.clear()
                    self.agent.edge_visit_cache.clear()
                    # we set loc to current loc such that loc gets added to edge and
                    #  node_visit_cache
                    self.agent.loc = self.agent.loc
                    self.trials.append(SimTrial(self.sim_maze, goal, initial_step, self.agent,
                                               max_steps=self.max_steps))

                    self.trials[-1].solve()
                    if (
                        self.continue_from_goal_after_max_steps
                        and self.max_steps
                        and self.trials[-1].step.loc != goal
                        and len(self.trials[-1].steps) >= self.max_steps
                    ):
                        self.agent.__reset_loc_to__(goal)
                    if verbose:
                        print('Solved trial %i' %len(self.trials))
                    self.total_steps += len(self.trials[-1].steps) - 1
                else:
                    pass
        else:
            for goal in self.goals:
                if not termination_condition(self):
                    initial_step = SimStep(self.agent.loc)
                    self.agent.node_visit_cache.clear()
                    self.agent.edge_visit_cache.clear()
                    # we set loc to current loc such that loc gets added to edge and
                    #  node_visit_cache
                    self.agent.loc = self.agent.loc
                    self.trials.append(SimTrial(self.sim_maze, goal, initial_step, self.agent,
                                             max_steps=self.max_steps))

                    self.trials[-1].solve()
                    if (
                        self.continue_from_goal_after_max_steps
                        and self.max_steps
                        and self.trials[-1].step.loc != goal
                        and len(self.trials[-1].steps) >= self.max_steps
                    ):
                        self.agent.__reset_loc_to__(goal)
                    if verbose:
                        print('Solved trial %i' %len(self.trials))
                    self.total_steps += len(self.trials[-1].steps) - 1
                else:
                    pass



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
            maze_nodes = list(self.maze.nodes)
            # exclude current location from the list from which the new goal
            # is picked
            maze_nodes.remove(self.trial.step.loc)
            new_goal = random.choice(maze_nodes)

            initial_step = copy.deepcopy(self.trial.step)
            self.trials.append(SimTrial(maze, new_goal, initial_step, agent,
                max_steps=0, goal_reached=True))

            super(SimSessionRandomGoal, self).run()


class SimParallelExperiment(object):
    def __init__(self, maze, agent_class, agent_init_opts, initial_step=None):

        self.maze = maze
        self.agent_class = agent_class
        self.agent_init_opts = agent_init_opts
        self.initial_step = initial_step
        self.sessions = []

    @ray.remote
    def run_sim_repeat(maze, agent_class, agent_init_opts, n_trials):
        # we check whether the agent class init method takes maze as an
        # argument
        # learning and omniscient agents do, other agents, e.g. heuristic, don't
        if 'maze' in inspect.getfullargspec(agent_class.__init__).args:
            simagent = agent_class(maze=maze, loc=(0,0), **agent_init_opts)
        else:
            simagent = agent_class(loc=(0,0), **agent_init_opts)

        goals = list(maze.randomgoals(n_trials))
        simsession = SimSessionGivenGoals(agent=simagent, maze=maze, goals=goals, loc_start=simagent.loc)
        simsession.run(progress_bar=False)
        
        esteps = [t.excess_steps() for t in simsession.trials]
        return esteps

    def run(self, n_trials, n_repeats):
        ray_tasks = []

        for _ in range(n_repeats):
            ray_tasks.append(self.run_sim_repeat.remote(self.maze, self.agent_class, self.agent_init_opts, n_trials))

        simsession_res = ray.get(ray_tasks)
        return simsession_res
