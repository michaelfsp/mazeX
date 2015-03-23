import numpy as np
import random
from collections import OrderedDict

from mazex.exceptions import *
from mazex.utils import (angle_between_nodes, assert_is_node,
assert_nodes_adjacent, get_von_mises_dictionary, get_von_mises_array,
get_angles_dictionary)

from networkx.algorithms.shortest_paths import astar_path

from scipy.stats import vonmises

def _get_loc(self):
    return self._loc

def _set_loc(self, node):
    """Performs all the necessary operations when location is changed except
    for caching taken steps."""
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

    if self._step_cache and self._step_cache[0] == node:
        self._step_cache.pop(0);
    
def softmax(beta, values):
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
    try:
        values = values.ravel() #will fail if it is not a numpy array
    except AttributeError:
        values = np.array(values)

    if beta == float('inf'):
        probs = np.zeros(len(values))
        max_value = np.max(values)
        index_maxs = np.flatnonzero(values == max_value)
        n_maxs = index_maxs.size

        probs[index_maxs] = 1/n_maxs

    elif beta == float('-inf'):
        probs = np.ones(len(values)) * 1/(len(values))
    
    else:
        #since the softmax is insensitive to subtraction of a value from
        #all values, we subtract the mean of the values in order to avoid
        #numerical overflows (see Daw 2009 (manuscript for Attention &
        #Performance XXIII)
        values-=np.mean(values)

        #this is the calculation of the numerator of the equation
        exp_elements = np.exp(beta*values)

        #this is the calculation of the denominator of the equation
        denominator = np.sum(exp_elements)  

        #and hnndore we compute the action probabilities
        probs = exp_elements/denominator    

    return probs


class Agent:
    def __init__(self, loc, beta, xi=0, penalty_criterium='node',
            relative_penalty=False, heading=None):
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

        relative_penalty : boolean
            If True a percentage of the value is subtracted instead of an
            absolute value.
            Note that if False, xi is still scaled by the maximum von Mises
            value for the chosen kappa.

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
        self.penalty_criterium = penalty_criterium
        self.relative_penalty = relative_penalty
        if self.penalty_criterium not in ('node', 'edge_unidir'):
            raise NotSupported

        self.heading = heading
        # we will use the property decorator to turn self.loc into
        # a smart property that agents with position memory will make
        # use of
        if assert_is_node(loc):
            self._loc = loc 

    def set_loc(self, node):
        old_loc = self.loc
        new_loc = node

        _set_loc(self, new_loc)

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
        
        #we do the same for the node visit cache
        if node in self.node_visit_cache:
            self.node_visit_cache.pop(node)

        self.node_visit_cache[node] = self.step_count

        self.step_count += 1
   
    loc = property(_get_loc, set_loc)

    def __reset_loc_to__(self, loc):
        """Since setting the agent's location for proximity checks for
        adjacency and complains if the nodes are not adjacent, this
        method can be called to force reset location to a specific one.

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

    def choice_probs(self, values):
        """Return choice probabilities given edge_values.
        """
        return softmax(self.beta, values)

    def choose_edge_index(self, values):
        """Choose one edge using softmax decision rule based on the
        edges values

        Output:
        -------
        index : int
            Index of the chosen edge
        """
        choice_prob = softmax(self.beta, values)

        cum_prob = np.cumsum(choice_prob)
        
        random_number = random.uniform(0, cum_prob[-1])

        return np.argmax(cum_prob >= random_number)

    def penalty_factor(self, nodeslist):
        """Returns the penalty factor for each node on a list.
        The rule is: for nodes which have never been visited the penalty is 0.
        All nodes which have been visited get ordered by step count at last
        visit in reverse order. The penalty is self.xi for last visited,
        self.xi*2**-1 for before last visited, self.xi*2**-2 for before before
        last visited, ...
        """
        
        penalties = np.zeros(len(nodeslist))

        if self.penalty_criterium == 'edge_unidir':
            edgeslist = [(self.loc, n) for n in nodeslist]
            
            step_count_last_visit = []

            for e in edgeslist:
                if e in self.edge_visit_cache:
                    step_count_last_visit.append(
                            self.edge_visit_cache[e])
                else:
                    # nodes that have never been visited we will
                    # assume they have been visited in "step -1"
                    # the value -1 doesn't play any actual role
                    # as these steps will have no penalization
                    step_count_last_visit.append(-1)

        elif self.penalty_criterium == 'node':
            step_count_last_visit = []

            for n in nodeslist:
                if n in self.node_visit_cache:
                    step_count_last_visit.append(
                            self.node_visit_cache[n])
                else:
                    # nodes that have never been visited we will
                    # assume they have been visited in "step -1"
                    # the value -1 doesn't play any actual role
                    # as these steps will have no penalization
                    step_count_last_visit.append(-1)

        step_count_last_visit = np.array(step_count_last_visit)

        #by sorting on -step_count_last_visit we get the reverse order, i.e.,
        # the indices for sorting by decreasing value
        sorting_indices = np.argsort(-step_count_last_visit)
        sorted_step_count = step_count_last_visit[sorting_indices]
        helper_index_matrix = np.vstack((sorting_indices, sorted_step_count))

        helper_matrix_only_visited_ones = helper_index_matrix[
                :,helper_index_matrix[1,:]>0]

        for i,x in enumerate(helper_matrix_only_visited_ones[0,:]):
            penalties[x] = self.xi * 2**-i

        return penalties
        
class HeuristicAgent(Agent):
    def __init__(self, loc, beta, xi, penalty_criterium='node',
            relative_penalty=False, heading=None, distribution='cos',
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
                penalty_criterium=penalty_criterium,
                relative_penalty=relative_penalty, heading=heading)

        self.kappa = kappa
        if self.kappa:
            self.vm_dist = get_von_mises_array(maze_size, kappa)
            self.max_vm = vonmises.pdf(0, kappa=kappa) 
        
        self.angles_dic = get_angles_dictionary(maze_size)
   
    def evaluate(self, nodeslist, goal, distribution='cos'):
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
        
        angles = [self.angles_dic[self.loc][n][goal] for n in nodeslist]
        
        if distribution == 'vonmises':
            # since what is stored in self.vm_dist is a 2D array with angles
            # and values, we first determine the index of the closes angle entry
            # in the matrix and then retrieve the distribution value
            vm_indices = [np.argmin(abs(self.vm_dist[0] - angle)) for angle in angles]

            dist_val = self.vm_dist[1][vm_indices]

            dist_max = self.max_vm

        elif distribution == 'cos':
            dist_val = np.cos(angles)

            dist_max = 1

        penalties = self.penalty_factor(nodeslist)

        values = dist_val - penalties
 
        return values

class NStepsAgent(Agent):
    def __init__(self, loc, maze, nsteps, beta = float('inf'), xi=0,
            penalty_criterium='node', relative_penalty=False, heading=None):
        """Creates an agent that has full knowledge of the maze and
        uses Graph Theory algorithms to determine the shortest path
        to goal"""
        
        super(NStepsAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                relative_penalty=relative_penalty, heading=heading) 

        self.maze = maze
        self.nsteps = nsteps

    def evaluate(self, nodeslist, goal):
        """Gets the next node in the node cache"""
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
           
            endpoints = [c]
            cycles = 0
            #while nsteps -1 because we are already starting from the choice
            # nodes and not from the start node
            while cycles < self.nsteps - 1:
                middlepoints = copy.deepcopy(endpoints)
                endpoints = []
                for e in middlepoints:
                    endpoints.extend(maze.neighbors(e))
                cycles += 1

            #the idea is to collect all points that can be reached in n steps
            # from the starting point c. this includes enpoints that are
            # reached by backtracking, i.e., the agent reverts steps

            endpoint_distances = [
                    mx.utils.manhattan_distance(e,g) for e in endpoints]
            values[i] = min(endpoint_distances)

        penalties = self.penalty_factor(nodeslist)

        values -= penalties

        return values

class OmniscientAgent(Agent):
    def __init__(self, loc, maze, beta = float('inf'), xi=0,
            penalty_criterium='node', relative_penalty=False, heading=None):
        """Creates an agent that has full knowledge of the maze and
        uses Graph Theory algorithms to determine the shortest path
        to goal"""
        
        super(OmniscientAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                relative_penalty=relative_penalty, heading=heading) 

        self.maze = maze

    def evaluate(self, nodeslist, goal):
        """Gets the next node in the node cache"""
        if not self._step_cache:
            self.plan_path(goal, self.maze);

        values = []

        for n in nodeslist:
            if n == self._step_cache[0]:
                values.append(1)
            else:
                values.append(-1)

        # penalties are only relevant for exploration
        penalties = self.penalty_factor(nodeslist)

        values -= penalties

        return values

    def plan_path(self, goal, maze):
        """Populates _step_cache with a series of nodes along the
        A* shortest path to goal and outputs that list"""
        path = astar_path(maze, self.loc, goal)

        # we discard the first node which is the start location
        self._step_cache = path[1:]

        return path
        
class MixtureAgent(HeuristicAgent):
    def __init__(self, loc, beta, maze, xi, penalty_criterium='node',
            relative_penalty=False, distribution='cos', kappa=None,
            heading=None, maze_size=6, switch_distance=0):
        super(MixtureAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                relative_penalty=relative_penalty, heading=heading,
                distribution=distribution, kappa=kappa, maze_size=maze_size)
        self.maze = maze
        self.switch_distance = switch_distance
        #self.switched = False
    
    def evaluate(self, nodeslist, goal, penalty_criterium='node', relative_penalty=False):

        if self.maze.distances_dictionary[self.loc][goal] <= self.switch_distance:
            #FIXME: this omniscient agent does not have penalties into account!
            omniscient_agent = OmniscientAgent(self.loc, self.maze)
            values = omniscient_agent.evaluate(nodeslist, goal)
            #self.switched = True
            #print('omniscient!')
        else:
            values = super(MixtureAgent, self).evaluate(nodeslist, goal,
                    penalty_criterium = penalty_criterium, relative_penalty = relative_penalty)
            #print('heuristic!')

        return values

class DirectedRandomAgent(Agent):
    def __init__(self, loc, beta, conc, xi=0, penalty_criterium='node',
            relative_penalty=False, uheading=None, direction=0):
        """Creates an agent that follows semi-random motion.
        The choice probabilities of this agent follow
        the circular von Mises distribution.
        
        Parameters:
        -----------
        conc : float
            Concentration parameter for the von Mises distribution.

        direction : float , optional, [0, 2*pi]
            Direction (location parameter for the von Mises
            distribution in radians off the center.
        """
        super(DirectedRandomAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                relative_penalty=relative_penalty, heading=heading) 

        self.direction = direction
        self.conc = conc

    def edge_value(self, edges):
        """Calculates the value of each edge available to the agent.

        Parameters:
        -----------

        edges : list
            List of edges to evaluate
        """
        pass

    def next_path_seg(self, loc, goal, mazemodel, memory=None):
        """Calculates the position the agent wants to visit next.
        """
        pass

