import copy
import random
from collections import OrderedDict

import numpy as np
from scipy.stats import vonmises

import mazex as mx
from mazex.exceptions import *
from mazex.utils import (angle_between_nodes, assert_is_node,
assert_nodes_adjacent, get_von_mises_dictionary, get_von_mises_array,
euclidean_distance, manhattan_distance, scale_dict_values)

from networkx.algorithms.shortest_paths import (astar_path,
single_source_shortest_path_length)

from numba import jit

#TODO: Have Agent.evaluate() which takes care of handling penalties, etc..
#  which would then invoke {HeuristicAgent, OmniscientAgent, ...}.eval_fun()
#  which would contain the evaluation bits which are truly different from
#  agent to agent
#TODO: Equip main Agent class with reset method that resets visit_caches,
#  planned step caches, etc...
def _get_loc(self):
    return self._loc

def _set_loc(self, node):
    """Performs all the necessary operations when location is changed except
    for caching taken steps."""
    old_loc = self.loc
    new_loc = node

    if old_loc is None:
        self._loc = node

    elif new_loc is None:
        self._loc = None

    elif not old_loc is None and not node is None:
        # I am calling it a directed edge because our edges normally
        # have a specific node order specification but here we care
        # about node order in that it specifies a direction
        directed_edge = (old_loc, new_loc)

        if directed_edge in self.edge_visit_cache:
            # pop() deletes the entry AND spits an output we
            # want to silence with the ;
            # it's necessary to delete the item with pop()
            # and re-add it because only then it assumes the
            # desired order in the ordered dictionary
            self.edge_visit_cache.pop(directed_edge);

        self.edge_visit_cache[directed_edge] = self.step_count

        # set heading and update loc
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

        # remove from the step cache
        if self._step_cache and self._step_cache[0] == node:
            self._step_cache.pop(0);

    #we do the same for the node visit cache
    if node in self.node_visit_cache:
        self.node_visit_cache.pop(node)

    self.node_visit_cache[node] = self.step_count
    self.step_count += 1

def n_degree_neighborhood(G, node, n):
    """Find the n degree neighborhood of node in graph G. Returns the list of
    nodes in graph G that are at a distance of exactly n.
    """
    path_lengths = single_source_shortest_path_length(G, node)
    return [node for node, length in path_lengths.items() if length == n]

#def softmax(beta, values):
#    """A Softmax function.
#
#    Parameters:
#    -----------
#    beta : float, [0+, inf+]
#        The inverse temperature parameter of the softmax function. A
#        very high value causes greedy action, whereas a low value very
#        stochastic action selection (more exploration).
#
#    values : list or numpy.array
#        The list of values of the actions available to the agent.
#    """
#    # Reduce dimensionality of values to 1, if it is an array, else
#    # transform it into an array
#    try:
#        values = values.ravel() #will fail if it is not a numpy array
#    except AttributeError:
#        values = np.array(values)
#
#    if beta == float('inf'):
#        probs = np.zeros(len(values))
#        max_value = np.max(values)
#        index_maxs = np.flatnonzero(values == max_value)
#        n_maxs = index_maxs.size
#
#        probs[index_maxs] = 1/n_maxs
#
#    elif beta == float('-inf'):
#        probs = np.ones(len(values)) * 1/(len(values))
#
#    else:
#        #since the softmax is insensitive to subtraction of a value from
#        #all values, we subtract the mean of the values in order to avoid
#        #numerical overflows (see Daw 2009 (manuscript for Attention &
#        #Performance XXIII)
#        values = values - np.mean(values)
#
#        #this is the calculation of the numerator of the equation
#        exp_elements = np.exp(beta*values)
#
#        #this is the calculation of the denominator of the equation
#        denominator = np.sum(exp_elements)
#
#        #and hnndore we compute the action probabilities
#        probs = exp_elements/denominator
#
#    return probs

@jit(nopython=True, nogil=True)
def softmax(beta, avalues):
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

    #since the softmax is insensitive to subtraction of a value from
    #all values, we subtract the mean of the values in order to avoid
    #numerical overflows (see Daw 2009 (manuscript for Attention &
    #Performance XXIII)

    mavalues = avalues - np.mean(avalues)

    #this is the calculation of the numerator of the equation
    exp_elements = np.exp(beta*mavalues)

    #this is the calculation of the denominator of the equation
    denominator = np.sum(exp_elements)

    #and hnndore we compute the action probabilities
    probs = exp_elements/denominator

    return probs

#import numexpr as ne
#def softmax_ne(beta, avalues):
#    #avalues_arr = np.array(avalues)
#    nvals = len(avalues)
#    sumvalues = ne.evaluate("sum(avalues)")
#    mavalues = sumvalues / nvals
#
#    #mavalues = avalues - np.mean(avalues)
#
#    exp_elements = ne.evaluate("exp(beta*(mavalues))")
#    denominator = ne.evaluate("sum(exp_elements)")
#    #denominator = np.sum(exp_elements)
#
#    #probs = np.evaluate("exp_elements/denominator")
#    probs = exp_elements/denominator
#    return probs

class Agent:
    def __init__(self, beta=1, loc=None, xi=0, penalty_criterium='node',
            penalty_type='exponential', heading=None):
        """Parameters:
        -----------
        beta : float, [0+, inf+]
            The inverse temperature parameter of the softmax function. A
            very high value causes greedy action, whereas a low value very
            stochastic action selection (more exploration).

        penalty_criterium : 'node' or 'edge_unidir', 'edge_bidir'
            If 'node', the agent will penalize options that lead to nodes it has
            already visited. If 'edge', the agent will only penalize edges it
            has already visited.

        heading : string
            this is the direction the agent is heading toward. It
            should be derived from the last agent's position by some
            external method. It can assume values '^', '>', '<',
            'v'
        """
        # we are using an OrderedDict to store the last visited edges
        # because it will make it easier to make the agent forget the
        # older entries by using self.edge_visit_cache.popitem(
        # last=False)
        self.edge_visit_cache = OrderedDict()
        self.node_visit_cache = OrderedDict()
        self.step_count = 0

        self._step_cache = [] #for an agent that plans multiple steps
        # at once
        self.beta = beta
        self.xi = xi
        self.penalty_type = penalty_type
        self.penalty_criterium = penalty_criterium
        if self.penalty_criterium not in ('node', 'edge_unidir'):
            raise NotSupported

        self.heading = heading
        # we will use the property decorator to turn self.loc into
        # a smart property that agents with position memory will make
        # use of
        #
        # It's a bit klugey but necessary to first set to None or else the
        #  function call of _set_loc() will fail
        self._loc = None

        # we use _set_loc() rather than setting self._loc directly because
        #  this way we make sure that even the loc on which the Agent is
        #  initialized is recorded in the node and edge visit caches
        if loc is None:
            _set_loc(self, loc)

        elif assert_is_node(loc):
            _set_loc(self, loc)

    loc = property(_get_loc, _set_loc)

    def __reset_loc_to__(self, loc):
        """Since setting the agent's location for proximity checks for
        adjacency and complains if the nodes are not adjacent, this
        method can be called to force reset location to a specific one.

        Use with caution! Recommended usage is now first agent.loc = None,
        then agent.loc = new_node. The user should manage the node and edge
        visit caches himself when using these transitions.

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

    def choice_probs(self, values, debug=False):
        """Return choice probabilities given edge_values.
        Inputs:
        -------
        values : list or Numpy array
            List of choice values.

        debug : boolean
            If debug mode is on, when two choices have the same probability and
            are the best choice, then the first one is always picked instead
            of choosing either with 50% chance. Should only be used with
            beta = float('inf')
        """
        if debug:
            if type(values) == list:
                values = np.array([values])
            if np.count_nonzero(values == max(values)) > 1:
                if self.beta == float('inf'):
                   probs = np.zeros(len(values))
                   probs[values.index(max(values))] == 1

        else:
            probs = softmax(self.beta, values)

        return probs

    def choose_edge_index(self, values, debug=False):
        """Choose one edge using softmax decision rule based on the
        edges values

        Inputs:
        -------
        values : list
            List of choice values.

        debug : boolean
            If debug mode is on, when two choices have the same probability and
            are the best choice, then the first one is always picked instead
            of choosing either with 50% chance. Should only be used with
            beta = float('inf')

        Output:
        -------
        index : int
            Index of the chosen edge
        """
        choice_prob = self.choice_probs(values, debug=debug)

        cum_prob = np.cumsum(choice_prob)

        random_number = random.uniform(0, cum_prob[-1])

        return np.argmax(cum_prob >= random_number)

    def _constant_penalty_factor(self, nodeslist):
        """Returns the penalty factor for each node on a list.
        The rule is: for nodes which have never been visited the penalty is 0.
        All nodes which have been visited get ordered by step count at last
        visit in reverse order. The penalty is self.xi for last visited,
        self.xi*2**-1 for before last visited, self.xi*2**-2 for before before
        last visited, ...
        """
        penalties = np.zeros(len(nodeslist))

        for i,n in enumerate(nodeslist):
            if self.penalty_criterium == 'edge_unidir':
                if (self.loc, n) in self.edge_visit_cache:
                    penalties[i] = self.xi

            elif self.penalty_criterium == 'node':
                if n in self.node_visit_cache:
                    penalties[i] = self.xi

        return penalties

    def _exponential_penalty_factor(self, nodeslist):
        """Returns the penalty factor for each node on a list.
        The rule is: for nodes which have never been visited the penalty is 0.
        All nodes which have been visited get ordered by step count at last
        visit in reverse order. The penalty is self.xi for last visited,
        self.xi*2**-1 for before last visited, self.xi*2**-2 for before before
        last visited, ...
        """
        penalties = np.zeros(len(nodeslist))

        if self.penalty_criterium == 'edge_unidir':
            i_visited = []
            edgeslist = [(self.loc, n) for n in nodeslist]

            step_count_last_visit = []

            for i, e in enumerate(edgeslist):
                if e in self.edge_visit_cache:
                    step_count_last_visit.append(
                            self.edge_visit_cache[e])
                    i_visited.append(i)

        elif self.penalty_criterium == 'node':
            i_visited = []
            step_count_last_visit = []

            for i, n in enumerate(nodeslist):
                if n in self.node_visit_cache:
                    step_count_last_visit.append(
                            self.node_visit_cache[n])
                    i_visited.append(i)

        #by sorting on -step_count_last_visit we get the reverse order, i.e.,
        # the indices for sorting by decreasing value
        #TODO: instead of using this matrix voodoo this algorithm could use
        # instead the order of the elements in the dictionaries since they are
        # actually OrderedDict. e.g.: list(d.keys()).index(node)
        step_count_last_visit = np.array(step_count_last_visit)
        sorting_indices = np.argsort(-step_count_last_visit)

        # first we have to make it an array so we can later use fancy sorting
        i_visited = np.array(i_visited)

        i_visited_sorted = i_visited[sorting_indices]

        for i,x in enumerate(i_visited_sorted):
            penalties[x] = self.xi * 2**-i

        return penalties

    def evaluate_penalties_session(self, session):
        """Return a vector with value of each chosen step and a matrix with
        values for every choice at every given step such that it can be fed to
        a likelihood function for fitting purposes.

        Note: it will change agent's loc and node and edge visit_caches
        """
        self.loc = None # stupid hack just to prevent that the change of
        # location later on is not assumed if it fails some validity check
        # usually for contiguity / possible transtion
        first_loc = session.trials[0].steps[0].loc
        self.loc = first_loc

        # Find out the size of the array we need
        # Since when the agent is at the goal there is no decision to be made
        # we need n-1 rows per trial as the last one is the goal, or n in case
        # the goal was not reached
        # Session.count_total_steps already takes care of that
        n_steps = session.count_total_steps()
        value_chosen = np.zeros(n_steps)

        values = np.empty((n_steps, 4))
        values.fill(np.nan)

        step_number = 0
        for t in session.trials:
            self.loc = None # klugey but necessary because otherwise agent
            # won't allow teleportation
            self.node_visit_cache.clear()
            self.edge_visit_cache.clear()

            for s in t.steps[:-1]: # last step is the goal
                self.loc = s.loc

                step_values = self.penalty_factor(s.available_nodes)

                value_chosen[step_number] = step_values[s.choice]
                values[step_number, 0:len(step_values)] = step_values

                step_number += 1

        return (value_chosen, values)

    def evaluate_session(self, session, penalties=False):
        """Return a vector with value of each chosen step and a matrix with
        values for every choice at every given step such that it can be fed to
        a likelihood function for fitting purposes.

        Note: it will change agent's loc and node and edge visit_caches
        """
        self.loc = None # stupid hack just to prevent that the change of
        # location later on is not assumed if it fails some validity check
        # usually for contiguity / possible transtion
        first_loc = session.trials[0].steps[0].loc
        self.loc = first_loc

        # Find out the size of the array we need
        # Since when the agent is at the goal there is no decision to be made
        # we need n-1 rows per trial as the last one is the goal, or n in case
        # the goal was not reached
        # Session.count_total_steps already takes care of that
        n_steps = session.count_total_steps()
        value_chosen = np.zeros(n_steps)

        values = np.empty((n_steps, 4))
        values.fill(np.nan)

        step_number = 0
        for t in session.trials:
            self.loc = None # klugey but necessary because otherwise agent
            # won't allow teleportation
            self.node_visit_cache.clear()
            self.edge_visit_cache.clear()

            for s in t.steps[:-1]: # last step is the goal
                self.loc = s.loc

                if penalties is False:
                    step_values = self.evaluate(s.available_nodes, t.goal,
                            penalties=False)
                else:
                    step_values = self.evaluate(s.available_nodes, t.goal,
                            penalties=True)

                value_chosen[step_number] = step_values[s.choice]
                values[step_number, 0:len(step_values)] = step_values

                step_number += 1

        return (value_chosen, values)

    def penalty_factor(self, nodeslist):
        """Returns the penalty factor for each node on a list.
        The rule is: for nodes which have never been visited the penalty is 0.

        For an Agent with exponential penalty factor:
        All nodes which have been visited get ordered by step count at last
        visit in reverse order. The penalty is self.xi for last visited,
        self.xi*2**-1 for before last visited, self.xi*2**-2 for before before
        last visited, ...

        For an Agent with constant penalty factor:
        All nodes visited are penalized equally by self.xi
        """
        if self.penalty_type == 'constant':
            return self._constant_penalty_factor(nodeslist)

        elif self.penalty_type == 'exponential':
            return self._exponential_penalty_factor(nodeslist)

    def simulate_session(self, maze, goals, init_loc=(0,0)):
        trials = []
        self.node_visit_cache.clear()
        self.edge_visit_cache.clear()
        self.loc = init_loc
        for goal in goals:
            init_step = mx.SimStep(self.loc)
            self.node_visit_cache.clear()
            self.edge_visit_cache.clear()
            # we set loc to current loc such that loc gets added to edge and
            #  node_visit_cache
            self.loc = self.loc
            trial = mx.SimTrial(maze, goal, init_step, self)
            trial.solve()
            print('Solved trial %i' %len(trials))
            trials.append(trial)

        session = mx.Session()
        session.trials+=(trials)
        return session

class HeuristicAgent(Agent):
    def __init__(self, beta=1, xi=3, loc=None, penalty_criterium='node',
            penalty_type='exponential', heading=None, distribution='cos',
            kappa=None, maze_size=6):
        """Creates an agent that follows a directional heuristic.

        Parameters:
        -----------
        kappa : float
            Concentration parameter for the von Mises distribution.

        xi : float
            Cost parameter for recently performed actions. From
            the greek xanazõ, to relive ;)
        """
        super(HeuristicAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium, penalty_type=penalty_type,
                heading=heading)

        self.kappa = kappa
        if self.kappa:
            self.vm_dist = get_von_mises_array(maze_size, kappa)
            self.max_vm = vonmises.pdf(0, kappa=kappa)

        #self.angles_dic = get_angles_dictionary(maze_size)
        #self.angles_dic = angle_between_nodes

    def evaluate(self, nodeslist, goal, distribution='cos', penalties=True):
        """Returns a vector with the edge values of each edge given as
        input.

        Parameters:
        -----------
        nodeslist : list of np.array, dim=1
            Nodes for which the transition to is to be evaluated.

        goal : node
            Location of goal.

        distribution : 'cos' or 'vonmises'
            If 'cos', a cosine function is used to evaluate node choice. If
            'vonmises' a von Mises distribution is used to evaluate node
            choice.
        """

        # this little trick makes the method work even if not a list
        # but a single node is passed
        try:
            assert_is_node(nodeslist)
            nodeslist = [nodeslist]
        except InvalidNode:
            pass

        angles = [angle_between_nodes(self.loc, n, goal) for n in nodeslist]

        if distribution == 'vonmises':
            # since what is stored in self.vm_dist is a 2D array with angles
            # and values, we first determine the index of the closes angle entry
            # in the matrix and then retrieve the distribution value
            vm_indices = [np.argmin(abs(self.vm_dist[0] - angle)) for angle in angles]

            dist_val = self.vm_dist[1][vm_indices]

            dist_max = self.max_vm # deprecated

        elif distribution == 'cos':
            dist_val = np.cos(angles)

            # deprecated: formerly used for relative penalties
            dist_max = 1 # deprecated

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values = dist_val - penalties

        else:
            values = dist_val

        return values

class ManhattanAgent(Agent):
    def __init__(self, beta=1, xi=3, loc=None, penalty_criterium='node',
            penalty_type='exponential', heading=None):
        """Creates an agent that follows a Manhattan distance heuristic.

        Parameters:
        -----------
        """
        super(ManhattanAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium, penalty_type=penalty_type,
                heading=heading)

    def evaluate(self, nodeslist, goal, penalties=True):
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

        current_manhattan_distance = manhattan_distance(self.loc, goal)

        values = []
        for n in nodeslist:
            if manhattan_distance(n, goal) < current_manhattan_distance:
                values.append(1)
            else:
                values.append(-1)

        values = np.array(values)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

class CentralityAgent(Agent):
    def __init__(self, maze, centrality='edge_betweenness', loc=None, beta=1,
            xi=3, penalty_criterium='node', penalty_type='exponential',
            heading=None):
        """Creates an agent which evaluates options according to centrality
        measures.

        Parameters:
        -----------
        centrality: 'edge_betweenness', 'betweenness', 'closeness',
            'communicability_betweenness', 'current_flow_betweenness',
            'edge_current_flow_betweenness', 'degree', 'dead_end',
            'real_degree'
        """
        #TODO: change names of dead_end and real_degree to something more
        # more senseful since they aren't actual centralities
        super(CentralityAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading)

        self.maze = maze
        self._centrality = centrality

        self._scaled_centrality = None

    @property
    def centrality(self):
        return self._centrality

    @centrality.setter
    def centrality(self, centrality_string):
        self._centrality = centrality_string
        self._scaled_centrality = None

    @property
    def scaled_centrality(self):
        if not self._scaled_centrality:
            c_dict = eval('self.maze.'+self.centrality+'_centrality')
            # we scale the values such that they span the range [-1, 1]
            #  as all other agents produce values in this range
            self._scaled_centrality = scale_dict_values(c_dict, high=1, low=-1)
            #self.scaled_centrality = scaled_c_dict
        return self._scaled_centrality

    def evaluate(self, nodeslist, goal, penalties=True):

        if (self.centrality == 'edge_betweenness' or
            self.centrality == 'edge_current_flow_betweenness'):
            # It is an edge_centrality
            edgeslist = [(self.loc, n) for n in nodeslist]
            values = []
            for e in edgeslist:
                # first we check under which name the edge is stored in the
                #  maze object because each edge is only addressable by either
                #  (n1, n2) or (n2, n1)
                values.append(self.scaled_centrality[e])
        else:
            # It is a node centrality
            values = [self.scaled_centrality[n] for n in nodeslist]

        values = np.array(values)

        if penalties:
            penalties = self.penalty_factor(nodeslist)
            values -= penalties

        return values

class NStepsAgent(Agent):
    def __init__(self, maze, nsteps, loc=None, beta=1, xi=3,
            penalty_criterium='node', penalty_type='exponential',
            heading=None):
        #TODO: Allow manhattan as well as euclidian distance
        """Creates an agent that has full knowledge of the maze and
        uses Graph Theory algorithms to determine the shortest path
        to goal"""

        super(NStepsAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading)

        self.maze = maze
        self.nsteps = nsteps

    def evaluate(self, nodeslist, goal, penalties=True):
        """Gets the next node in the node cache"""
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            endpoints, midpoints = self.maze.nodes_reachable_n_steps(c,
                    self.nsteps -1)

            endpoint_distances = [
                    manhattan_distance(e, goal) for e in endpoints]

            values[i] = -min(endpoint_distances)

            # In case the goal was already transversed during those steps this
            #  option gets added the number of remaining steps.
            if goal in midpoints:
                values[i] += self.nsteps - midpoints[goal]

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

class AStarAgent(Agent):
    def __init__(self, maze, nsteps, loc=None, beta=1, xi=3,
            penalty_criterium='node', penalty_type='exponential',
            heading=None):
        #TODO: Allow manhattan as well as euclidian distance
        """Creates an agent that has full knowledge of the maze and
        uses Graph Theory algorithms to determine the shortest path
        to goal"""

        super(AStarAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading)

        self.maze = maze
        self.nsteps = nsteps

    def evaluate(self, nodeslist, goal, penalties=True):
        """Gets the next node in the node cache"""
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            reachable_nodes = []
            reachable_nodes.extend(self.maze.nodes_reachable_n_steps(c,
                    self.nsteps -1))
        reachable_nodes = list(set(reachable_nodes))

            reachable_nodes_eval = []
            for i, r in reachable_nodes:
                # The evaluation of the node is the sum of the geodesic distance
                # between current location and that node, and the euclidean/manhattan
                # distance between that node and the goal
                node_eval = serf.maze.distances_dictionary[self.loc][r] + euclidean_distance(self.loc, goal)
                reachable_nodes_eval.append(node_eval)

            value[i] = -min(reachable_nodes_eval)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

class OmniscientAgent(Agent):
    def __init__(self, maze, loc=None, beta=1, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None):
        """Creates an agent that has full knowledge of the maze and
        uses Graph Theory algorithms to determine the shortest path
        to goal"""

        super(OmniscientAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading)

        self.maze = maze

    def evaluate(self, nodeslist, goal, penalties=True):
        """Gets the next node in the node cache."""
        values = []

        current_distance_to_goal = self.maze.distances_dictionary[self.loc][
                goal]

        for n in nodeslist:
            # if a node is along the shortest path to goal it's distance to
            #  goal is less than the distance from current location
            if self.maze.distances_dictionary[n][
                    goal] < current_distance_to_goal:
                values.append(1)
            else:
                values.append(-1)

        # penalties are only relevant when there is exploration, i.e.,
        #  beta < +inf
        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        values = np.array(values)

        return values

    def plan_path(self, goal, maze):
        """Populates _step_cache with a series of nodes along the
        A* shortest path to goal and outputs that list"""
        path = astar_path(maze, self.loc, goal)

        # we discard the first node which is the start location
        self._step_cache = path[1:]

        return path

class MixtureAgent(Agent):
    #def __new__(cls, maze, loc=None, beta=1, xi=0, gamma=0, delta=0, epsilon=0,
    #        zeta=0, nsteps=0, centrality='edge_betweenness'):
    #    #obj = super(MixtureAgent, cls).__new__(cls, )
    #    #obj = super( Agent(beta, loc, xi)

    #    obj.maze = maze

    #    obj.gamma = gamma
    #    obj.delta = delta
    #    obj.epsilon = epsilon
    #    obj.zeta = zeta

    #    obj.nsteps = nsteps
    #    obj.centrality = centrality

    #    if gamma:
    #        heuristic_agent = HeuristicAgent
    #        heuristic_agent.loc = cls.loc
    #        obj.heuristic_agent = HeuristicAgent(beta, loc=loc)

    #    if delta:
    #        obj.omniscient_agent = OmniscientAgent(maze, loc=loc)
    #    if epsilon:
    #        obj.nsteps_agent = NStepsAgent(maze, nsteps, loc=loc)
    #    if zeta:
    #        obj.centrality_agent = CentralityAgent(maze, centrality=centrality,
    #                loc=loc)

    #    #super(MixtureAgent, obj).__init__(loc=loc, beta=beta, xi=xi)
    #    return obj


    def __init__(self, maze, loc=None, beta=1, xi=0, gamma=0, delta=0,
                 epsilon=0, zeta=0, nsteps=2, centrality='edge_betweenness'):
        super(MixtureAgent, self).__init__(loc=loc, beta=beta, xi=xi)

        self.maze = maze

        self._gamma = gamma #    Heuristic
        self._delta = delta #    Planning
        self._epsilon = epsilon #N Steps
        self._zeta = zeta #      Centrality

        self._nsteps = nsteps
        self._centrality = centrality #HACK: this is not being used directly
                                      # until centrality.setter

        if gamma:
            self.heuristic_agent = HeuristicAgent(loc=loc)
        else:
            self.heuristic_agent = None

        if delta:
            self.omniscient_agent = OmniscientAgent(maze, loc=loc)
        else:
            self.omniscient_agent = None

        if epsilon:
            self.nsteps_agent = NStepsAgent(maze, nsteps, loc=loc)
        else:
            self.nsteps_agent = None

        if zeta:
            self.centrality_agent = CentralityAgent(maze,
                                                    centrality=centrality,
                                                    loc=loc)
        else:
            self.centrality_agent = None

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, new_gamma):
        if not self.heuristic_agent:
            self.heuristic_agent = HeuristicAgent(loc=self.loc)
        self._gamma = new_gamma

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, new_delta):
        if not self.maze and new_delta != 0:
            Exception('MixtureAgent.maze must be set first')

        elif not self.omniscient_agent:
            self.omniscient_agent = OmniscientAgent(self.maze, loc=self.loc)
        self._delta = new_delta

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, new_epsilon):
        if not self._nsteps and new_epsilon != 0:
            raise Exception('MixtureAgent.nsteps must be set first')
        if not self.maze and new_epsilon != 0:
            Exception('MixtureAgent.maze must be set first')

        elif not self.nsteps_agent:
            self.nsteps_agent = NStepsAgent(self.maze, self.nsteps,
                                            loc=self.loc)
        self._epsilon = new_epsilon

    @property
    def zeta(self):
        return self._zeta

    @zeta.setter
    def zeta(self, new_zeta):
        if not self._centrality and new_zeta != 0:
            raise Exception('MixtureAgent.centrality must be set first')

        elif not self.centrality_agent:
            self.centrality_agent = CentralityAgent(self.maze,
                                                centrality=self._centrality,
                                                loc=self.loc)
        self._zeta = new_zeta

    @property
    def centrality(self):
        if self.centrality_agent:
            return self.centrality_agent.centrality
        else:
            return None

    @centrality.setter
    def centrality(self, new_centrality):
        if not self.maze:
            raise Exception('MixtureAgent.maze must be set first')

        self.centrality_agent = CentralityAgent(self.maze,
                                                centrality=new_centrality,
                                                loc=self.loc)
        self._centrality = new_centrality


    @property
    def nsteps(self):
        if self.nsteps_agent:
            return self._nsteps
        else:
            return None

    @nsteps.setter
    def nsteps(self, new_nsteps):
        if not self.maze:
            raise Exception('MixtureAgent.maze must be set first')

        self.nsteps_agent = NStepsAgent(self.maze, nsteps=new_nsteps,
                                        loc=self.loc)
        self._nsteps = new_nsteps

    def evaluate(self, nodeslist, goal):
        if self.gamma:
            heuristic_value = self.heuristic_agent.evaluate(nodeslist, goal,
                    penalties = False)
        else:
            heuristic_value = 0

        if self.delta:
            omniscient_value = self.omniscient_agent.evaluate(nodeslist, goal,
                    penalties = False)
        else:
            omniscient_value = 0

        if self.epsilon:
            nsteps_value = self.nsteps_agent.evaluate(nodeslist, goal,
                    penalties = False)
        else:
            nsteps_value = 0

        if self.zeta:
            centrality_value = self.centrality_agent.evaluate(nodeslist, goal,
                    penalties = False)
        else:
            centrality_value = 0

        if self.xi:
            penalties = self.penalty_factor(nodeslist)
        else:
            penalties = 0

        value = ((self.gamma*heuristic_value + self.delta*omniscient_value +
                self.epsilon*nsteps_value + self.zeta*centrality_value)
                - penalties)
        return value

    def set_loc(self, node):
        _set_loc(self, node)
        #TODO: Warning this is klugey!!! I am ignoring the individual agents'
        # _set_loc and setting _loc directly. Could lead to unexpected behavior
        # if evaluate functions depend on information generated by _set_loc,
        # other than for penalty factor which is defined at the superclass
        # Agent level
        if self._gamma:
            self.heuristic_agent._loc = node
        if self._delta:
            self.omniscient_agent._loc = node
        if self._epsilon:
            self.nsteps_agent._loc = node
        if self._zeta:
            self.centrality_agent._loc = node

    loc = property(_get_loc, set_loc)

class SwitchAgent(HeuristicAgent):
    def __init__(self, beta, maze, xi, loc=None, penalty_criterium='node',
            penalty_type='exponential', distribution='cos', kappa=None,
            heading=None, maze_size=6, switch_distance=0):
        super(SwitchAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium, penalty_type=penalty_type,
                heading=heading, distribution=distribution, kappa=kappa,
                maze_size=maze_size)
        self.maze = maze
        self.switch_distance = switch_distance
        #self.switched = False

    def evaluate(self, nodeslist, goal, penalty_criterium='node'):

        if self.maze.distances_dictionary[self.loc][goal] <= self.switch_distance:
            #FIXME: this omniscient agent does not have penalties into account!
            omniscient_agent = OmniscientAgent(self.loc, self.maze)
            values = omniscient_agent.evaluate(nodeslist, goal)
            #self.switched = True
            #print('omniscient!')
        else:
            values = super(MixtureAgent, self).evaluate(nodeslist, goal,
                    penalty_criterium = penalty_criterium)
            #print('heuristic!')

        return values

class BrownianAgent(Agent):
    def __init__(self, beta, loc=None, xi=0, penalty_criterium='node',
            penalty_type='exponential', heading=None):
        """Creates an agent that follows semi-Brownian motion or truly
        Brownian for xi=0.
        """
        super(BrownianAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading)

    def evaluate(self, nodeslist, goal, penalties=True):
        values = np.zeros(len(nodeslist))

        if penalties:
            penalties = self.penalty_factor(nodeslist)
            values = values - penalties

        return values
