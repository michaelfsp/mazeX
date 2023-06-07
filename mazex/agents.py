import copy
import random
from collections import OrderedDict

import numpy as np
from scipy.stats import vonmises

import mazex as mx
from mazex.exceptions import *
from mazex.utils import (angle_between_nodes, assert_is_node,
assert_nodes_adjacent, get_von_mises_dictionary, get_von_mises_array,
euclidian_distance, manhattan_distance, scale_dict_values)

from networkx.algorithms.shortest_paths import (astar_path,
single_source_shortest_path_length)

from numba import jit
import math

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

#---------------Stuff for the most recent ERL & co implementation-------------
def reward_function(s, s_prime, s_T, step_reward_fun, goal_reward):
    '''Returns the reward associated with the current transition.
    '''
    if s_prime == s_T:
        R = step_reward_fun(s, s_prime, s_T) + goal_reward
    else:
        R = step_reward_fun(s, s_prime, s_T)
    return R
    
    
# These are the best case scenario costs and rewards, i.e., what could be achievable
# in a fully connected environment
#optimistic_disc_step_costs = lambda d, gamma, step_cost: sum([gamma**i *step_cost for i in range(d)])
optimistic_disc_step_costs = lambda d, gamma, step_cost: sum([gamma**(i-1) *step_cost for i in range(1, d+1, 1)]) if d!=0 else 0
optimistic_disc_goal_reward = lambda d, gamma, goal_reward: gamma**(d-1) * goal_reward if d !=0 else 0

def cost_to_go_goal_reward_init(s, s_T, gamma, step_cost, goal_reward):
    manhattan_distance_0 = abs(s_T[0] - s[0]) + abs(s_T[1] - s[1])
    
    return optimistic_disc_step_costs(manhattan_distance_0, gamma, step_cost) + optimistic_disc_goal_reward(
    manhattan_distance_0, gamma, goal_reward)

def spatial_drive(s, s_T, m, n):
    s_xdist = abs(s_T[0] - s[0])
    s_ydist = abs(s_T[1] - s[1])
    sum_dists = s_xdist**n + s_ydist**n
    D = sum_dists**(1/m)
    return D

def spatial_reward(s, s_prime, s_T, m, n):
    """Here we calculate the spatial reward based on a specific transition from s to s_prime with a goal set in s_T
    $ r(s, s_prime, s_T) = D(s, s_T) - D(s, s_T)$
    """
    D_s = spatial_drive(s, s_T, m, n)
    D_s_prime = spatial_drive(s_prime, s_T, m, n)
    R = D_s - D_s_prime
    return R

def spatial_drive_dist(d_s, m, n):
    xdist = abs(d_s[0])
    ydist = abs(d_s[1])
    D_d_s = (xdist**n + ydist**n)**(1/m)
    return D_d_s

def spatial_reward_K(d_s, K, m, n):
    """Here we calculate the spatial reward based on initial state and a displacement vector K
    $ r(s, K) = D(d_s) - D(d_s+K)$
    where $d_s$ is the corrdinates in homeostatic space taking $s_T$ as the origin
    K is the displacement vector from s to s_prime
    """
    D_d_s = spatial_drive_dist(d_s, m, n)
    d_s_plus_K = (d_s[0] + K[0], d_s[1] + K[1])
    D_d_s_plus_K = spatial_drive_dist(d_s_plus_K, m, n)
    R = D_d_s - D_d_s_plus_K
    return R

def spatial_reward_init(s, s_T, gamma, m, n):
    """This is the best case scenario, which corresponds to a fully connected maze."""
    discounted_rewards = 0
    # compute initial displacement along x and y axes
    x_dist = abs(s_T[0] - s[0])
    y_dist = abs(s_T[1] - s[1])
    d_s = (x_dist, y_dist)
    manhattan_dist_0 = d_s[0] + d_s[1]
    
    # these are fixed tuples representing a step along x or y in the direction of the goal
    K_x = (-1, 0)
    K_y = (0, -1)
    
    discounted_rewards = 0
    # we iterate over possible steps
    # the agent should always prefer larger rewards first
    for i in range(0, manhattan_dist_0, 1):
        if spatial_reward_K(d_s, K_x, m, n) > spatial_reward_K(d_s, K_y, m, n):
            # then the agent does a step in x
            K_t = K_x
        elif spatial_reward_K(d_s, K_x, m, n) < spatial_reward_K(d_s, K_y, m, n):
            # then the agent does a step in y
            K_t = K_y
        else:
            # then a step in x or y are equivalent so we decide by convention for x
            # technically the agent should choose either K_x or K_y with 0.5 prob each
            # but it shouldn't make a difference to assume this convention instead
            K_t = K_x
        R = spatial_reward_K(d_s, K_t, m, n)
        discounted_rewards += gamma**i * R
        d_s = (d_s[0] + K_t[0], d_s[1] + K_t[1])
    # sanity check
    if d_s != (0,0):
        raise Exception('Final distance is not zero')
    return discounted_rewards 

def spatial_rewards(d, gamma, m, n):
    """Not being used"""
    discounted_rewards = 0
    for i, k in enumerate(range(d, 0, -1)):
        D_s = (k**n)**(1/m)
        D_s_prime = ((k-1)**n)**(1/m)
        r = D_s - D_s_prime
        discounted_rewards += gamma**i * r
    return discounted_rewards

#-----------------------------------------------------------------------------



def euclidean_drive(s, s_T, n_over_m):
    """
    s: previous node
    s_T: goal node
    m_over_n: ration between m and n parameters of drive function
    """
    return euclidian_distance(s, s_T)**n_over_m

def euclidean_reward(s, s_prime, s_T, n_over_m):
    s_euclidean_drive = euclidean_drive(s, s_T, n_over_m)
    s_prime_euclidean_drive = euclidean_drive(s_prime, s_T, n_over_m)
    return s_euclidean_drive - s_prime_euclidean_drive

def n_degree_neighborhood(G, node, n):
    """Find the n degree neighborhood of node in graph G. Returns the list of
    nodes in graph G that are at a distance of exactly n.
    """
    path_lengths = single_source_shortest_path_length(G, node)
    return [node for node, length in path_lengths.items() if length == n]

# TODO: implement `def ERL_R` which just retrieves the tensor with the Rewards
# and use that in init_Q

# TODO: d_E vector -> R_e matrix

def R_e(maze, n_over_m):
    """Creates tensor with euclidean rewards.

    Parameters:
    -----------
    maze : maze
        Maze for which rewards are created. It only matters to find out which
        actions are available at any state.
    n_over_m : float
        Power to elevate euclidean distance to in the calculation of the
        Euclidean rewards

    Returns:
    --------
    out : array(nxnxn)
        It returns an array of nxnxn where n is the number of nodes in the maze
        with R(s, s', s_T).
        If s and s' are not connected: R(s,s',:) = NaN
        If s == s_T: R(s,s,s) = 0 and R(s,:,s) = NaN, regardles of connectivity
        , i.e., the only action available at the terminal state is
        self-transition and its reward is defined to be 0.
    """
    R_size = len(maze.nodes())
    R = np.full((R_size, R_size, R_size), np.nan)

    for i, s in enumerate(maze.nodes()):
        for k, s_T in enumerate(maze.nodes()):
            if s == s_T:
                # here we define value of transitions to iself if at s_T
                R[i, i, i] =  0
            else:
                for j, s_prime in enumerate(maze.nodes()):
                    if maze.has_edge(s, s_prime):
                        R[i, j, k] = euclidean_reward(s, s_prime, s_T, n_over_m)
    return R

def init_Q(maze, initval=0, n_over_m=None, Q_val_s_equal_s_prime=np.nan,
           Q_vals_s_T=None, Q_val_s_equal_s_prime_equal_s_T=0):
    """Initializes Q(s, s', s_T) tensor. It is meant to contain value of taking action s' in state s when goal is s_T.

    Parameters
    ----------
    maze : maze
        Maze to which this Q table is initialized. It is used to lookup disconnected nodes to insert the corresponding Qs as NaN
    initval : float, "euclidean"
        Value to insert for existing connections. If "euclidean", then the calculated euclidean reward is used, in which case n_over_m becomes mandatory parameter.
    n_over_m : float, optonal
        Ratio n/m  (i.e. exponent of euclidean distance), following homeostatic RL equation for homeostatic drive
    Q_val_s_equal_s_prime : np.nan, int, float
        Value to be given to all "transitions to self".
    Q_vals_s_T : None, int, float
        Value to be given to transitions out of terminal state (goal) onto adjacent nodes. If
        None, then the initval is applied, otherwise this value is used. Can be
        helpful to implement appropriate update rules for Q(s,s_T,s_T), i.e.,
        the Q of the last step. By definition v(s_T) = 0 for terminal states,
        so Q(s_T, :, s_T) should also be zero
    Q_val_s_equal_s_prime_equal_s_T : np.nan, int or float
        Value to be given to the transition to itself when at the terminal node
        This is important for several things, among other things, for applying
        the Bellman operator, potentially to get the Q backups right for the
        terminal state, etc.
    """
    Q_size = len(maze.nodes())
    Q = np.full((Q_size, Q_size, Q_size), np.nan)
    for i, v in enumerate(maze.nodes()):
        # here we define the value of transitions to itself
        # regardless of destination
        if Q_val_s_equal_s_prime is not np.nan:
            Q[i, i, :] = Q_val_s_equal_s_prime

        # here we define value of transitions to iself if at s_T
        Q[i, i, i] = Q_val_s_equal_s_prime_equal_s_T

        for j, w in enumerate(maze.nodes()):
            if maze.has_edge(v, w):
                for k, g in enumerate(maze.nodes()):
                    if (v == g) and (Q_vals_s_T is not None):
                        # here we define the value of transitions from s_T to
                        # other locations this is important for Q updates: we want 
                        # $ max_a{Q(s_prime, a)} = 0 $ by definition for the
                        # terminal node
                        Q[i, j, k] = Q_vals_s_T

                    elif initval == "euclidean":
                        Q[i, j, k] = euclidean_reward(v, w, g, n_over_m)

                    elif type(initval) in (float, int):
                        Q[i, j, k] = initval

                    else:
                        raise Exception('initval must be float, int or "euclidean", not {}'.format(type(initval)))
    return Q


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
    if math.isinf(beta):
        probs = np.zeros(len(avalues))
        probs[avalues == max(avalues)] = 1

    else:
        #since the softmax is insensitive to subtraction of a value from
        #all values, we subtract the mean of the values in order to avoid
        #numerical overflows (see Daw 2009 (manuscript for Attention &
        #Performance XXIII)

        # I am now subtracting the max instead of the average as it might be
        # more numerically stable
        mavalues = avalues - np.max(avalues)

        #this is the calculation of the numerator of the equation
        exp_elements = np.exp(beta*mavalues)

        #this is the calculation of the denominator of the equation
        denominator = np.sum(exp_elements)

        #and hnndore we compute the action probabilities
        probs = exp_elements/denominator

    return probs


def normsoftmax(gamma, avalues):
    """From https://openreview.net/pdf?id=4g7nCbpjNwd
    Normsoftmax: Normalize the input and softmax to accelerate and stabilize training
    """
    numerator = avalues / gamma
    denominator = np.min((np.std(avalues/gamma), 1))
    softmax_argument = numerator / denominator
    return softmax(1, softmax_argument)


@jit(nopython=True, nogil=True)
def unitvarsoftmax(beta, avalues):
    if len(avalues) == 1:
        # normalization by std only works for 2+ elements
        return np.array([1.])
    elif math.isinf(beta):
        # If beta is +np.inf the the rescaling doesn't matter and is a waste of time
        return softmax(beta, avalues)
    else:
        avalues_rescaled = avalues / np.std(avalues)
        return softmax(beta, avalues_rescaled)


#import jax
#import jax.numpy as jnp
#@jax.jit
#def softmax_jax(beta, avalues):
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
#
#    #since the softmax is insensitive to subtraction of a value from
#    #all values, we subtract the mean of the values in order to avoid
#    #numerical overflows (see Daw 2009 (manuscript for Attention &
#    #Performance XXIII)
#
#    mavalues = avalues - jnp.mean(avalues)
#
#    #this is the calculation of the numerator of the equation
#    exp_elements = jnp.exp(beta*mavalues)
#
#    #this is the calculation of the denominator of the equation
#    denominator = jnp.sum(exp_elements)
#
#    #and hnndore we compute the action probabilities
#    probs = exp_elements/denominator
#
#    return probs
#
#
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
            penalty_type='exponential', heading=None, policy='softmax'):
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

        if policy == 'softmax':
            self.pi_fun = softmax
        elif policy == 'unitvarsoftmax':
            self.pi_fun = unitvarsoftmax
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
            probs = self.pi_fun(self.beta, values)

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
            # also prevents triggering of value update for the RL agents
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

            # once the iteration through all the locations but the last one
            # has happened we move the agent to the last location so that it 
            # can trigger a value update for the RL agents
            self.loc = t.steps[-1].loc

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
        #TODO: make this more efficient by only calling these functions if
        # self.xi != 0
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


class ERLAgent(Agent):
    """This Agent is legacy. Use OneStepTD instead!
    """
    def __init__(self, maze, gamma, n_over_m, alpha, initval='euclidean', beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None):

        super(ERLAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading)

        self.maze = maze
        self.gamma = gamma
        self.n_over_m = n_over_m
        self.alpha = alpha
        # Reminder: order in Q is (loc, step, goal)
        self.Q = init_Q(maze, initval, n_over_m)
        self.goal = None

    def _set_loc_ERL(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            old_loc = self.loc
            new_loc = node
            if old_loc != new_loc:
                self.update_Q(old_loc, new_loc, self.goal)
        _set_loc(self, node)

    loc = property(_get_loc, _set_loc_ERL)

    def get_Q_a(self, s, s_prime, s_T):
        """Helper function to retrieve Q(s,s',s_T) by looking up and using correct indices.
        """
        s_i = list(self.maze.nodes()).index(s)
        s_prime_i = list(self.maze.nodes()).index(s_prime)
        s_T_i = list(self.maze.nodes()).index(s_T)

        return self.Q[s_i, s_prime_i, s_T_i]

    def set_Q_a(self, s, s_prime, s_T, new_Q_a):
        s_i = list(self.maze.nodes()).index(s)
        s_prime_i = list(self.maze.nodes()).index(s_prime)
        s_T_i = list(self.maze.nodes()).index(s_T)
        self.Q[s_i, s_prime_i, s_T_i] = new_Q_a

    def get_all_Q_a(self, s, s_T):
        s_i = list(self.maze.nodes()).index(s)
        s_T_i = list(self.maze.nodes()).index(s_T)
        all_Q_s_s_T = self.Q[s_i, :, s_T_i]
        nan_i = np.isnan(all_Q_s_s_T)
        Q_as = all_Q_s_s_T[~nan_i]

        return Q_as

    def update_Q(self, s, s_prime, s_T):
        Q_s_a = self.get_Q_a(s, s_prime, s_T)
        R = euclidean_reward(s, s_prime, s_T, self.n_over_m)
        if s_prime == s_T:
            # by definition the value of a terminal state is zero, i.e.,
            # v(s_T) = 0
            max_Q_s_prime = 0
        else:
            max_Q_s_prime = np.max(self.get_all_Q_a(s_prime, s_T))
        RPE = self.alpha * (R + self.gamma * max_Q_s_prime - Q_s_a)
        new_Q_s_a = Q_s_a + RPE
        self.set_Q_a(s, s_prime, s_T, new_Q_s_a)

    def evaluate(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            values[i] = self.get_Q_a(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_Q=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_Q() via _set_loc_ERL(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores Q after session has been evaluated.
        """
        if restore_Q:
            old_Q = copy.deepcopy(self.Q)

        res = super(ERLAgent, self).evaluate_session(session, penalties=penalties)

        if restore_Q:
            self.Q = old_Q

        return res


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

        #angles = [angle_between_nodes(self.loc, n, goal) for n in nodeslist]
        angles = []
        for n in nodeslist:
            if self.loc == goal:
                # if the agent is at the goal already we define all angles as 1rad as it is the equivalent of moving in the opposite direction
                angles.append(np.pi)
            else:
                angles.append(angle_between_nodes(self.loc, n, goal))

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


class OneStepTD(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(OneStepTD, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)
        V_s_prime = self.get_V(s_prime, s_T)
        
        R = self.reward_function(s, s_prime, s_T)

        RPE = R + self.gamma * V_s_prime - V_s_old
        V_s_new = V_s_old + self.alpha * RPE
        
        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


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
                node_eval = self.maze.distances_dictionary[self.loc][r] + euclidian_distance(self.loc, goal)
                reachable_nodes_eval.append(node_eval)

            values[i] = -min(reachable_nodes_eval)

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


class Q0(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6):
        super(Q0, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.Q = self._init_Q(self.initfun)
        else: 
            self.Q = np.full((self.maze_size**2, self.maze_size**2, 4), self.initfun, dtype='float')

        for s_i, s in enumerate(maze.nodes()):
            for s_T_i, s_T in enumerate(maze.nodes()):
                if s_i == s_T_i:
                    self.Q[s_i, s_T_i, :] = 0

        self.goal = None
        self.reward_function = reward_function
    
    def _init_Q(self, initfun):
        Q = np.full((self.maze_size**2, self.maze_size**2, 4), np.nan)
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                for k, s_prime in enumerate(self.maze.adjacent_neighbors(s)):
                    Q[i, j, k] = initfun(s_prime, s_T)
        return Q
        
    def _set_loc_Q0(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            if node:
                old_loc = self.loc
                new_loc = node
                if old_loc != new_loc:
                    self.update_Q(old_loc, new_loc, self.goal)
        _set_loc(self, node)
        
    loc = property(_get_loc, _set_loc_Q0)
    
    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def _get_action_index(self, s, s_T, s_prime):
        """Helper function to retrieve indices of actions. It is used to retrieve values from the Q(s, s_T, s_prime) tensor.
        
        Returns a tuple with the "coordinates" of the action in the tensor
        """
        s_i = self._get_node_index(s)
        s_T_i = self._get_node_index(s_T)
        s_prime_i = self.maze.adjacent_neighbors(s).index(s_prime)
        return (s_i, s_T_i, s_prime_i)
        
    def get_Q(self, s, s_T, s_prime):
        """Helper function to retrieve Q(s,s_T,s_prime) by looking up and using correct indices.
        """
        Q_s_s_T_s_prime_idxs = self._get_action_index(s, s_T, s_prime)
        return self.Q[Q_s_s_T_s_prime_idxs]

    def update_Q(self, s, s_prime, s_T):
        # indices in the order that mazeX orders nodes in the maze
        s_prime_i, s_T_i = self._get_node_index([s_prime, s_T])
        
        # the index of the action in the Q tensor depends on the order of node listing in mazeX _and_ the order
        # of maze.adjacent_neighbors() for s_prime_i
        
        Q_s_s_T_s_prime_idxs = self._get_action_index(s, s_T, s_prime)

        Q_old = self.get_Q(s, s_T, s_prime)
        max_a_Q_s_prime = np.nanmax(self.Q[s_prime_i, s_T_i, :])
        
        R = self.reward_function(s, s_prime, s_T)

        RPE = R + self.gamma * max_a_Q_s_prime - Q_old
        #if RPE != 0:
        #    import ipdb; ipdb.set_trace()
        Q_new = Q_old + self.alpha * RPE
        
        self.Q[Q_s_s_T_s_prime_idxs] = Q_new

    def evaluate(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            values[i] = self.get_Q(self.loc, self.goal, c)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_Q=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_Q() via _set_loc_Q0(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores Q after session has been evaluated.
        """
        if restore_Q:
            old_Q = copy.deepcopy(self.Q)

        res = super(Q0, self).evaluate_session(session, penalties=penalties)

        if restore_Q:
            self.Q = old_Q

        return res


class Q01R(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initval, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6):
        super(Q01R, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        self.initval = initval
        self.Q = self._init_Q1R(self.initval)

        self.goal = None
        self.reward_function = reward_function
    
    def _init_Q1R(self, initfun):
        Q = np.full((self.maze_size**2, 4), np.nan)
        for i, s in enumerate(self.maze.nodes()):
            for k, _ in enumerate(self.maze.adjacent_neighbors(s)):
                    Q[i, k] = self.initval
        return Q
        
    def _set_loc_Q01R(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            if node:
                old_loc = self.loc
                new_loc = node
                if old_loc != new_loc:
                    self.update_Q01R(old_loc, new_loc, self.goal)
        _set_loc(self, node)
        
    loc = property(_get_loc, _set_loc_Q01R)
    
    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def _get_action_index(self, s, s_prime):
        """Helper function to retrieve indices of actions. It is used to retrieve values from the Q(s, s_T, s_prime) tensor.
        
        Returns a tuple with the "coordinates" of the action in the tensor
        """
        s_i = self._get_node_index(s)
        s_prime_i = self.maze.adjacent_neighbors(s).index(s_prime)
        return (s_i, s_prime_i)
        
    def get_Q(self, s, s_prime):
        """Helper function to retrieve Q(s,s_T,s_prime) by looking up and using correct indices.
        """
        Q_s_s_prime_idxs = self._get_action_index(s, s_prime)
        return self.Q[Q_s_s_prime_idxs]

    def update_Q01R(self, s, s_prime, s_T):
        # Needs to have s_T as argument so the reward can be calculated but the agent has only
        # a single value representation and doesn't use the goal information for its update
        
        # indices in the order that mazeX orders nodes in the maze
        s_prime_i = self._get_node_index(s_prime)
        
        # the index of the action in the Q tensor depends on the order of node listing in mazeX _and_ the order
        # of maze.adjacent_neighbors() for s_prime_i
        

        Q_old = self.get_Q(s, s_prime)
        max_a_Q_s_prime = np.nanmax(self.Q[s_prime_i, :])
        
        R = self.reward_function(s, s_prime, s_T)

        RPE = R + self.gamma * max_a_Q_s_prime - Q_old
        #if RPE != 0:
        #    import ipdb; ipdb.set_trace()
        Q_new = Q_old + self.alpha * RPE
        
        Q_s_s_prime_idxs = self._get_action_index(s, s_prime)
        self.Q[Q_s_s_prime_idxs] = Q_new

    def evaluate(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            values[i] = self.get_Q(self.loc, c)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_Q=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_Q() via _set_loc_Q0(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores Q after session has been evaluated.
        """
        if restore_Q:
            old_Q = copy.deepcopy(self.Q)

        res = super(Q01R, self).evaluate_session(session, penalties=penalties)

        if restore_Q:
            self.Q = old_Q

        return res


class Zlearning(Agent):
    def __init__(self, maze, alpha, gamma=1, rho=10, initval=1, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(Zlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initval = initval
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        
        self.goal = None
        self.Z = self._init_Z(initval)

#        if self.gamma:
#            self.evaluate = self.evaluate_w_gamma
#        else:
#            self.evaluate = self.evaluate_no_gamma

        self.evaluate = self.evaluate_no_gamma

    def _init_Z(self, initval):
        Z = float(initval) * np.ones((self.maze_size**2, self.maze_size**2), dtype=np.float128)
        # We need to make sure the sure that Z[i,i] = 1 because the cost of the
        # terminal state has to be 0, which equates to a z-value of 1
        np.fill_diagonal(Z, 1)
        return Z
        
    def _set_loc_Zlearning(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_Z(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_Zlearning)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_Z(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.Z[s_i, s_T_i]

    def update_Z(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        Z_s_old = self.get_Z(s, s_T)
        Z_s_prime = self.get_Z(s_prime, s_T)
        
        #R = self.reward_function(s, s_prime, s_T)

        ZPE = np.exp(-self.rho) * Z_s_prime - Z_s_old
        Z_s_new = Z_s_old + self.alpha * ZPE
        
        self.Z[s_i, s_T_i] = Z_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            #values[i] = self.reward_function(self.loc, c, goal)
            values[i] = -mx.utils.euclidian_distance(c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class Vmaxlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(Vmaxlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        nodeslist = list(self.maze.neighbors(s))
        Q_values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(s, c, s_T)
            Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        
        VRPE = Q_values.max() - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class Vmaxprelearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(Vmaxprelearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        #s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        #V_s_old = self.get_V(s, s_T)

        #nodeslist = list(self.maze.neighbors(s))
        #Q_values = np.zeros(len(nodeslist))

        #for i, c in enumerate(nodeslist):
        #    R = self.reward_function(s, c, s_T)
        #    Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        #
        #VRPE = Q_values.max() - V_s_old

        #V_s_new = V_s_old + self.alpha * VRPE

        #self.V[s_i, s_T_i] = V_s_new
        pass

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        s_i, s_T_i = self._get_node_index([self.loc, goal])
        V_s_old = self.get_V(self.loc, goal)

        values = np.zeros(len(nodeslist))
        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        VRPE = values.max() - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        s_i, s_T_i = self._get_node_index([self.loc, goal])
        V_s_old = self.get_V(self.loc, goal)

        values = np.zeros(len(nodeslist))
        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        VRPE = values.max() - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class MaxVmaxlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(MaxVmaxlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        nodeslist = list(self.maze.neighbors(s))
        Q_values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(s, c, s_T)
            if c == s_prime:
                if s_prime != s_T:
                    s_prime_neighbors = list(self.maze.neighbors(s_prime))
                    Q_s_prime_neighbors = np.zeros(len(s_prime_neighbors))
                    for j, s_prime_neighbor in enumerate(s_prime_neighbors):
                        R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                        Q_s_prime_neighbors[j] = R_s_prime_neighbor + self.gamma * self.get_V(s_prime_neighbor, s_T)
                    V_s_prime = Q_s_prime_neighbors.max()
                else:
                    V_s_prime = self.get_V(s_prime, s_T)
                Q_values[i] = (R + self.gamma * V_s_prime)

            else:
                Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        
        VRPE = Q_values.max() - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class MaxVlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(MaxVlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)
        R = self.reward_function(s, s_prime, s_T)       

        if s_prime != s_T:
            s_prime_neighbors = list(self.maze.neighbors(s_prime)) 
            Q_s_prime = np.zeros(len(s_prime_neighbors))
            for i, s_prime_prime in enumerate(s_prime_neighbors):
                Q_s_prime[i] = self.reward_function(s_prime, s_prime_prime, s_T) + self.gamma * self.get_V(s_prime_prime, s_T)

            V_s_prime = Q_s_prime.max()
        else:
            V_s_prime = self.get_V(s_prime, s_T)
        
        RPE = R + (self.gamma * V_s_prime) - V_s_old
        V_s_new = V_s_old + self.alpha * RPE
        
        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class Vmaxsprimelearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(Vmaxsprimelearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        _, s_prime_i, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_prime_old = self.get_V(s_prime, s_T)

        s_prime_nodeslist = list(self.maze.neighbors(s_prime))
        Q_values_s_prime = np.zeros(len(s_prime_nodeslist))

        for i, c in enumerate(s_prime_nodeslist):
            R = self.reward_function(s_prime, c, s_T)
            Q_values_s_prime[i] = (R + self.gamma * self.get_V(c, s_T))
        
        VRPE = Q_values_s_prime.max() - V_s_prime_old

        V_s_prime_new = V_s_prime_old + self.alpha * VRPE

        self.V[s_prime_i, s_T_i] = V_s_prime_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class VmaxVlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(VmaxVlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, s_prime_i, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        nodeslist_s_prime = list(self.maze.neighbors(s_prime))
        Q_values_s_prime = np.zeros(len(nodeslist_s_prime))

        for i, s_prime_prime in enumerate(nodeslist_s_prime):
            r_s_prime = self.reward_function(s_prime, s_prime_prime, s_T)
            Q_values_s_prime[i] = (r_s_prime + self.gamma * self.get_V(s_prime_prime, s_T))

        V_s_prime = Q_values_s_prime.max()
        
        R = self.reward_function(s, s_prime, s_T)
        VRPE = R + self.gamma * V_s_prime - V_s_old 
        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class ExpVlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(ExpVlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)
        R = self.reward_function(s, s_prime, s_T)
        
        if s_prime != s_T:
            nodeslist_s_prime = list(self.maze.neighbors(s_prime))
            Q_values_s_prime = np.zeros(len(nodeslist_s_prime))

            for i, s_prime_prime in enumerate(nodeslist_s_prime):
                r_s_prime_prime = self.reward_function(s_prime, s_prime_prime, s_T)
                Q_values_s_prime[i] = (r_s_prime_prime + self.gamma * self.get_V(s_prime_prime, s_T))

            V_s_prime = np.sum(self.choice_probs(Q_values_s_prime) * Q_values_s_prime)
        else:
            V_s_prime = self.get_V(s_prime, s_T)
        
        VRPE = R + self.gamma * V_s_prime - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res
