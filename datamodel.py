from mazex.mazes import SquareMaze
from mazex.exceptions import *
from mazex.utils import check_elements_equal

class Step:
    def __init__(self, loc, trial, previous=None):
        """Initializes the Step.
        
        Parameters:
        -----------     
        loc : node
            node indicating the current location

        trial : Trial instance
            a Trial instance to which this Step will be linked. It
            provides information about goal, maze, etc...
        
        heading : string
            this is the direction the agent is heading toward. It
            should be derived from the last agent's position by some
            external method. It can assume values '^', '>', '<',
            'v'

        previous : Step
            Previous step.
        """
        self.loc = loc
        self.trial = trial
        self.previous = previous
        # sets .subsequent for the previous step
        if previous:
            previous.subsequent = self

        self.subsequent = None
        self._heading = None

    @property
    def available_nodes(self):
        return self.trial.maze.neighbors(self.loc)

    @property
    def choice(self):
        return self.available_nodes.index(self.subsequent.loc)

    @property
    def heading(self):
        if self._heading:
            return self._heading

        elif self.previous:
            if self.previous.loc[1] < self.loc[1]:
                direction = '^'
            elif self.previous.loc[1] > self.loc[1]:
                direction = 'v'
            elif self.previous.loc[0] < self.loc[0]:
                direction = '>'
            elif self.previous.loc[0] > self.loc[0]:
                direction = '<'

        elif not self.previous:
            if self.subsequent:
                # this case this is the first step of a trial
                if self.subsequent.loc[1] > self.loc[1]:
                    direction = '^'
                elif self.subsequent.loc[1] < self.loc[1]:
                    direction = 'v'
                elif self.subsequent.loc[0] > self.loc[0]:
                    direction = '>'
                elif self.subsequent.loc[0] < self.loc[0]:
                    direction = '<'
     
            else:
                direction = None #it should only happen if it is an isolated step

        return direction

    @heading.setter
    def heading(self, direction):
        self._heading = direction

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
    def __init__(self, maze, goal, max_steps=0, goal_reached=True):
        if isinstance(maze, SquareMaze):
            self.maze = maze
        else:
            raise InvalidMaze
        self.goal = goal
        # Trial termination conditions
        self.max_steps = max_steps
        self.goal_reached = goal_reached

        self.steps = []
    
    @property
    def loc(self):
        return self.step.loc

    @property
    def step(self):
        return self.steps[-1]

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

class Session:
    def __init__(self):
        """Session initiation method.

        Parameters:
        -----------
        agent : Agent
            Agent that will solve the trials of the session

        trials : list
            List of trials to be presented to the Agent
        """
        self.trials = []
        self._current_trial_index = None     
        self._current_trial_index_manually_set = False

    def count_total_steps(self):
        """Count total number of steps in session.
        """
        total_steps = 0
        for t in self.trials:
            total_steps += len(t.steps)

        return total_steps

    @property
    def current_trial_index(self):
        if self._current_trial_index_manually_set:
            return self._current_trial_index

        else:
            return len(self.trials) - 1

    @current_trial_index.setter
    def current_trial_index(self, index):
        self._current_trial_index = index
        self._current_trial_index_manually_set = True

    @property
    def maze(self):
        """If all steps have the same maze, return that maze, else return a
        list of mazes. This might not be after some refactoring to be done in 
        a future incarnation.
        """
        mazes = [s.maze for s in self.trials]
        if check_elements_equal(mazes):
            return mazes[0]
        else:
            return mazes

    @property
    def trial(self):
        return self.trials[self.current_trial_index]
