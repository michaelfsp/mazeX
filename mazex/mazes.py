import functools
import numpy as np
import networkx as nx
import random
import copy
from functools import lru_cache, partial
import re

from mazex.exceptions import *
from mazex.utils import copydoc, to_peano_coords, sdd_path, euclidian_distance
import mazex.analysis

#TODO: save centrality measures as attributes of edges and nodes

def add_reverse_direction_edge_names(edge_centrality_dic):
    """Since different edge centrality calculation algorithms name edges
    differently and only in one way, we add the symmetrical label definition.
    """
    new_dic = copy.deepcopy(edge_centrality_dic)
    for n1, n2 in edge_centrality_dic.keys():
        new_dic[(n2, n1)] = edge_centrality_dic[((n1, n2))]
    return new_dic

@lru_cache(maxsize=128)
def cached_betweenness_centrality(maze):
    return nx.algorithms.betweenness_centrality(maze)

@lru_cache(maxsize=128)
def cached_edge_betweenness_centrality(maze):
    ec = nx.algorithms.edge_betweenness_centrality(maze)
    edge_centralities_dic_complete = add_reverse_direction_edge_names(ec)
    return edge_centralities_dic_complete

@lru_cache(maxsize=128)
def cached_closeness_centrality(maze):
    return nx.algorithms.closeness_centrality(maze)

@lru_cache(maxsize=128)
def communicability_betweenness_centrality(maze):
    return nx.algorithms.communicability_betweenness_centrality(maze)

@lru_cache(maxsize=128)
def cached_current_flow_betweenness_centrality(maze):
    #NOTE: HACK - we have to create a Graph from maze or else networkx
    # complains about not getting argument 'size'
    return nx.centrality.current_flow_betweenness_centrality(nx.Graph(maze))

@lru_cache(maxsize=128)
def cached_edge_current_flow_betweenness_centrality(maze):
    #NOTE: HACK - we have to create a Graph from maze or else networkx
    # complains about not getting argument 'size'
    ec = nx.centrality.edge_current_flow_betweenness_centrality(nx.Graph(
    maze))
    edge_centralities_dic_complete = add_reverse_direction_edge_names(ec)
    return edge_centralities_dic_complete

@lru_cache(maxsize=128)
def cached_degree_centrality(maze):
    return nx.algorithms.degree_centrality(maze)

def maze_from_ASCII(maze_str):
    """Generate a maze configuration from text.

    Example:

    maze_str = '''
    ._. ._. . .
    | | . ._. .
    . . ._. . .
    | | . | . |
    ._._._. ._.
    | | ._. |_.
    '''

    maze_from_ASCII(maze_str)
    """
    string_lines = maze_str.splitlines()
    string_lines.reverse()
    line_lengths = [len(line) for line in string_lines]
    set_lengths = set(line_lengths)
    set_lengths.remove(0) # remove empty lines

    if  len(set_lengths) > 1:
        raise InvalidMaze

    maze_size = int((set_lengths.pop()+1)/2)

    h_edges = []
    v_edges = []
    for i, line_index in enumerate(np.nonzero(line_lengths)[0]):
        line = string_lines[line_index]

        # horizontal edges
        # this is the relationship between index in the string and maze coordinates
        h_edges_str_i = [a.start() for a in re.finditer('_', line)]
        h_edges_i = (np.array(h_edges_str_i)/2) - 0.5
        h_edges_i = h_edges_i
        for j in h_edges_i:
            j = int(j)
            h_edges.append(((j,i), (j+1,i)))

        # vertical edges
        v_edges_str_i = [a.start() for a in re.finditer('\|', line)]
        v_edges_i = np.array(v_edges_str_i)/2
        v_edges_i = v_edges_i
        for j in v_edges_i:
            j = int(j)
            v_edges.append(((j,i), (j,i+1)))

    maze = SquareMaze(maze_size)
    maze.add_edges_from(h_edges)
    maze.add_edges_from(v_edges)

    return maze

@lru_cache(maxsize=128)
def nodes_reachable_n_steps(maze, node, n):
    """Find all nodes reachable in exactly n steps starting from node. This
    includes nodes reachable by backtracking.

    Output:
    -------
    endpoints : list
        Points reached in exactly n steps
    midpoints : dictionary
        Points traveled through and their respective distances from node.
        Note: Includes node.
    """
    # in practice this is obtained by getting the n, n-2, n-4, ...
    # neighborhood of the node
    path_lengths = nx.single_source_shortest_path_length(maze, node, cutoff=n)
    valid_lengths = range(n, -1, -2)

    endpoints = [node for node, length in path_lengths.items() if length in
            valid_lengths]
    # now we prepare a dictionary with the midpoints crossed and their
    #  respective distances from starting point. for that we just need to
    #  remove the entries in path_lengths that correspond to the endpoints
    # we have to iterate over a deepcopy of the original dictionary because
    #  modifying the dictionary during the iteration results in problems
    for node, distance in copy.deepcopy(path_lengths).items():
        if distance == n:
            path_lengths.pop(node); # semi colon suppresses output of item

    midpoints = path_lengths

    return endpoints, midpoints

@functools.lru_cache(1024)
def least_sdd_path_steps_matrix(maze, gamma, m_over_n, peano_order=False):
    "Returns matrix with least SDD for all start, goal pairs for a given maze."
    n_nodes = len(maze.nodes())
    erl_steps_matrix = np.full((n_nodes, n_nodes), np.nan, dtype='float')

    nodes = list(maze.nodes())
    if peano_order:
        peano_order = [to_peano_coords(node) for node in nodes]
        ordered_nodes = [nodes[i] for i in peano_order]
    else:
        ordered_nodes = nodes

    for iv, v in enumerate(ordered_nodes):
        for iw, w in enumerate(ordered_nodes):
            if v == w:
                erl_steps_matrix[iv, iw] = 0
            else:
                all_simple_paths = list(nx.algorithms.all_simple_paths(maze, v, w))
                all_simple_paths_sdd = np.array([sdd_path(p, m_over_n, gamma) for p in all_simple_paths])
                # we find all the paths where SDD is minimal, in case there is
                # more than one where that is the case
                least_sdd_paths_i = np.where(all_simple_paths_sdd == all_simple_paths_sdd.min())[0]
                least_sdd_paths = [all_simple_paths[i] for i in least_sdd_paths_i]
                # we need to subract 1 because the paths include the start and
                # end node and we are interested in the number of nodes
                # traversed
                least_sdd_paths_lengths = [len(p)-1 for p in least_sdd_paths]
                erl_steps_matrix[iv, iw] = np.average(least_sdd_paths_lengths)
    return erl_steps_matrix


@functools.lru_cache(1024)
def Pdiff(maze):
    return np.linalg.inv(maze.degree_matrix) @ nx.adjacency_matrix(maze)

@functools.lru_cache(1024)
def DiffR(maze, gamma):
    return np.linalg.inv(np.identity(36) - gamma * Pdiff(maze))

class SquareMaze(nx.classes.graph.Graph):
    """Generic class of for square mazes.
    """
    def __init__(self, maze_size=None, name=None):
        """Generates a square maze of given maze_size and no connections.
        Total number of nodes will maze_size**2.
        """
        super(SquareMaze, self).__init__()

        if maze_size:
            self.maze_size = maze_size

        self.name = name

        # We will calculate the distances matrix once and then only
        # recalculate if either the nodes or eges of the maze have changed
        # We save the snapshot of both at time of calculation in these objects
        self._has_changed_since_matrix = True
        self._has_changed_since_euclidean_matrix = True
        self._has_changed_since_matrix_peano = True
        self._has_changed_since_dictionary = True
        self._has_changed_since_to_coord_map = True
        self._has_changed_since_least_sdd = True
        self._has_changed_since_optimal_choice_dic = True

        self._distances_matrix = None
        self._distances_matrix_peano = None
        self._distances_dictionary = None
        self._distances_to_coord_map = None
        self._least_sdd_matrix = None
        self._optimal_choice_dic = None
        self._euclidean_distances_matrix = None
        self._Pdiff = None
        self._DiffR = None

        self.nodes_reachable_n_steps = partial(nodes_reachable_n_steps,
                                               self)

    @property
    def betweenness_centrality(self):
        return cached_betweenness_centrality(self)

    @property
    def edge_betweenness_centrality(self):
        return cached_edge_betweenness_centrality(self)

    @property
    def closeness_centrality(self):
        return cached_closeness_centrality(self)

    @property
    def communicability_betweenness_centrality(self):
        return communicability_betweenness_centrality(self)

    @property
    def current_flow_betweenness_centrality(self):
        return cached_current_flow_betweenness_centrality(self)

    @property
    def edge_current_flow_betweenness_centrality(self):
        return cached_edge_current_flow_betweenness_centrality(self)

    @property
    def degree_centrality(self):
        return cached_degree_centrality(self)

    # TODO: change names of dead_end and real_degree to something more
    # more senseful since they aren't actual centralities

    @property
    @lru_cache(maxsize=128)
    def dead_end_centrality(self):
        '''It is not a real centrality.
        '''
        dead_end_centrality_dic = {}
        for node, degree in self.degree():
            if degree == 1:
                dead_end_centrality_dic[node] = -1
            else:
                dead_end_centrality_dic[node] = 1
        return dead_end_centrality_dic

    @property
    def d_minus_half(self):
        return np.sqrt(np.linalg.inv(self.degree_matrix))

    @property
    def d_half(self):
        return np.sqrt(self.degree_matrix)

    @property
    @lru_cache(maxsize=128)
    def fiedler_vector_centrality(self):
        '''It is not a real centrality either. It simply is the value of the
        eigenvector corresponding to the second smallest eigenvalue of the
        normalized Graph Laplacian.
        '''
        """
        # this return a matrix. need to conver it to array
        normed_laplacian = nx.normalized_laplacian_matrix(self)
        normed_laplacian_array = normed_laplacian.toarray()

        # The Laplacian matrix is symmetric so we can use eigh which is for
        # Hermitian/Symmetric matrices
        eigvals, eigvecs = np.linalg.eigh(normed_laplacian_array)

        # The eigenvalues returned by eigh are already sorted so we can
        # confidently take the eigenvector of index 1 which would be the
        # second smallest eigenvector which is the Fiedler vector
        fiedler_vector = eigvecs[1]
        fiedler_vector = nx.fiedler_vector(self)

        scaled_fiedler_vector = np.dot(self.d_minus_half, fiedler_vector)

        scaled_fiedler_vector = np.where(fiedler_vector<0, 0, 1)

        fiedler_vector_centrality_dic = {}

        # I believe the order of the nodes on the adjacency matrix and
        # therefore that of the entries on the fiedler vector are the same as
        # in maze.nodes()
        for n, v in zip(self.nodes(), scaled_fiedler_vector):
            fiedler_vector_centrality_dic[n] = v
        """

        # this bit is copied from nx.spectral_ordering because only in this
        # way can we be certain of the order of nodes in the fiedler vector
        G = nx.algebraicconnectivity._preprocess_graph(self, weight=None)

        find_fiedler = nx.algebraicconnectivity._get_fiedler_func('tracemin')

        fiedler_vector_centrality_dic = {}

        for component in nx.connected_components(G):
            size = len(component)
            if size > 2:
                L = nx.laplacian_matrix(G, component)
                fiedler = find_fiedler(L, None, False, 1e-8)[1]
                for x, c, u in sorted(zip(fiedler, range(size), component)):
                    fiedler_vector_centrality_dic[u] = x

        return fiedler_vector_centrality_dic

    @property
    @lru_cache(maxsize=128)
    def fiedler_vector_partition(self):
        partition = {}
        for k, v in self.fiedler_vector_centrality.items():
            if v < 0.0:
                partition[k] = 0
            else:
                partition[k] = 1

        return partition

    @property
    @lru_cache(maxsize=128)
    def normalized_fiedler_vector_centrality(self):
        '''It is not a real centrality either. It simply is the value of the
        eigenvector corresponding to the second smallest eigenvalue of the
        normalized Graph Laplacian.
        '''
        """
        # this return a matrix. need to conver it to array
        normed_laplacian = nx.normalized_laplacian_matrix(self)
        normed_laplacian_array = normed_laplacian.toarray()

        # The Laplacian matrix is symmetric so we can use eigh which is for
        # Hermitian/Symmetric matrices
        eigvals, eigvecs = np.linalg.eigh(normed_laplacian_array)

        # The eigenvalues returned by eigh are already sorted so we can
        # confidently take the eigenvector of index 1 which would be the
        # second smallest eigenvector which is the Fiedler vector
        fiedler_vector = eigvecs[1]
        fiedler_vector = nx.fiedler_vector(self)

        scaled_fiedler_vector = np.dot(self.d_minus_half, fiedler_vector)

        scaled_fiedler_vector = np.where(fiedler_vector<0, 0, 1)

        fiedler_vector_centrality_dic = {}

        # I believe the order of the nodes on the adjacency matrix and
        # therefore that of the entries on the fiedler vector are the same as
        # in maze.nodes()
        for n, v in zip(self.nodes(), scaled_fiedler_vector):
            fiedler_vector_centrality_dic[n] = v
        """

        # this bit is copied from nx.spectral_ordering because only in this
        # way can we be certain of the order of nodes in the fiedler vector
        G = nx.algebraicconnectivity._preprocess_graph(self, weight=None)

        find_fiedler = nx.algebraicconnectivity._get_fiedler_func('tracemin')

        fiedler_vector_centrality_dic = {}

        for component in nx.connected_components(G):
            size = len(component)
            if size > 2:
                L = nx.laplacian_matrix(G, component)
                fiedler = find_fiedler(L, None, True, 1e-8)[1]
                for x, c, u in sorted(zip(fiedler, range(size), component)):
                    fiedler_vector_centrality_dic[u] = x/(np.sqrt(
                        self.degree[u]))

        return fiedler_vector_centrality_dic

    @property
    @lru_cache(maxsize=128)
    def normalized_fiedler_vector_partition(self):
        partition = {}
        for k, v in self.normalized_fiedler_vector_centrality.items():
            if v < 0:
                partition[k] = 0
            else:
                partition[k] = 1

        return partition

    @property
    @lru_cache(maxsize=128)
    def spectral_ordering_centrality(self):
        specord = nx.spectral_ordering(self)

        spectral_ordering_centrality_dic = {}
        for n in self.nodes():
            spectral_ordering_centrality_dic[n] = specord.index(n)

        return spectral_ordering_centrality_dic

    @property
    @lru_cache(maxsize=128)
    def real_degree_centrality(self):
        '''It is not a real centrality.
        '''
        return self.degree()

    #TODO: move this to networkx
    @property
    def degree_matrix(self):
        degree_matrix = np.zeros((self.maze_size**2, self.maze_size**2))

        for i, (nd) in enumerate(self.degree()):
            degree_matrix[i,i] = nd[1]

        return degree_matrix

    @property
    def euclidean_distances_matrix(self):
        # If the graph has changed since last calculation of the matrix:
        if self._has_changed_since_euclidean_matrix:
            # for some reason the NetworkX algorithm produces a numpy matrix
            # instead of an array, so we np.array it
            dm = np.zeros((len(list(self.nodes())), len(list(self.nodes()))))
            for i, n1 in enumerate(list(self.nodes())):
                for j, n2 in enumerate(list(self.nodes())):
                    if i != j:
                        dm[i,j] = euclidian_distance(n1, n2)

            self._euclidean_distances_matrix = dm
            self._has_changed_since_euclidean_matrix = False

        return self._euclidean_distances_matrix

    @property
    def distances_matrix(self):
        # If the graph has changed since last calculation of the matrix:
        if self._has_changed_since_matrix:
            # for some reason the NetworkX algorithm produces a numpy matrix
            # instead of an array, so we np.array it
            self._distances_matrix = np.asarray(
                    nx.algorithms.floyd_warshall_numpy(self))
            self._has_changed_since_matrix = False

        return self._distances_matrix

    def least_sdd_matrix(self):
        #FIXME: this should take different argument values for m_over_n and gamma
        # If the graph has changed since last calculation of the matrix:
        if self._has_changed_since_least_sdd:
            # for some reason the NetworkX algorithm produces a numpy matrix
            # instead of an array, so we np.array it
            self._least_sdd_matrix = least_sdd_path_steps_matrix(self, gamma=1, m_over_n=1, peano_order=False)
            self._has_changed_since_least_sdd = False

        return self._least_sdd_matrix
    
    @property
    def Pdiff(self):
        return Pdiff(self)

    def DiffR(self, gamma):
        return DiffR(self, gamma)

    @property
    def optimal_choice_dic(self):
        if self._has_changed_since_optimal_choice_dic:

            # If the graph has changed since last calculation of the dictionary:
            optimal_choice_dic = {}
            for loc in self.nodes():
                optimal_choice_dic[loc] = {}
                for goal in self.nodes():
                    optimal_choice = []
                    for s_prime in self.neighbors(loc):
                        if self.distances_dictionary[s_prime][goal] < self.distances_dictionary[loc][goal]:
                            optimal_choice.append(s_prime)
                    optimal_choice_dic[loc][goal] = optimal_choice

            self._optimal_choice_dic = optimal_choice_dic
            self._has_changed_since_optimal_choice_dic = False

        return self._optimal_choice_dic

    @property
    def distances_matrix_peano(self):
        # If the graph has changed since last calculation of the matrix:
        if self._has_changed_since_matrix_peano:
            # for some reason the NetworkX algorithm produces a numpy matrix
            # instead of an array, so we np.array it
            idx_nodes_peano = [mazex.utils.to_peano_coords(n) for n in self.nodes()]
            # TODO: generalize this for mazes of different sizes
            # this is a list of all nodes sorted along shortest path distance
            # on a Peano maze from position (0,0)
            nodes_peano_sorted = [None] * 36
            for i, d in enumerate(range(36)):
                nodes_peano_sorted[i] = list(self.nodes())[idx_nodes_peano.index(d)]

            self._distances_matrix_peano = np.asarray(
                    nx.algorithms.floyd_warshall_numpy(self, nodelist=nodes_peano_sorted))
            self._has_changed_since_matrix_peano = False

        return self._distances_matrix_peano


    @property
    def distances_dictionary(self):
        if not self._has_changed_since_dictionary:
            return self._distances_dictionary
        else:
            self._distances_dictionary = nx.algorithms.floyd_warshall(self)
            self._has_changed_since_dictionary = False
            return self._distances_dictionary

    @property
    def distances_to_coord(self):
        if not self._has_changed_since_to_coord_map:
            return self._distances_to_coord_map
        else:
            maze_dists = np.unique(self.distances_matrix)
            mapping_dict = {}

            for goal in self.node:
                for n in self.node:
                    dist_coord = mazex.utils.to_dist_coord(n, self, goal)

                    try:
                        mapping_dict[dist_coord][goal].append(n)

                    except KeyError:
                        try:
                            mapping_dict[dist_coord][goal] = [n]

                        except KeyError:
                            mapping_dict[dist_coord] = {}
                            mapping_dict[dist_coord][goal] = [n]

            self._distances_to_coord_map = mapping_dict
            self._has_changed_since_to_coord_map = False
            return self._distances_to_coord_map

    @property
    def maze_size(self):
        return self._maze_size

    @maze_size.setter
    def maze_size(self, maze_size):
        super(SquareMaze, self).__init__()

        for i in np.ndindex((maze_size, maze_size)):
            super(SquareMaze, self).add_node(i, pos=i)

        self._maze_size = maze_size

    @property
    def pos(self):
        return nx.get_node_attributes(self, name='pos')

    @property
    def possible_edges(self):
        """List all possible connections for given maze.
        """
        possible_edges = []

        for n in self.nodes():
            for nbr in self.adjacent_neighbors(n):
                if nbr[0]>n[0] or nbr[1]>n[1]:
                    possible_edges.append((n, nbr))
        # This is optional but makes the output pretier
        possible_edges.sort()

        return possible_edges

    @property
    def absent_edges(self):
        """List edges that could exist but do not in the maze's
        current state.
        """
        absent_edges = self.possible_edges

        for e in self.edges():
            try:
                absent_edges.remove(e)
            except ValueError:
                absent_edges.remove((e[1],e[0]))

        return absent_edges

    @property
    def n_possible_edges(self):
        return 2*((self.maze_size**2)-self.maze_size)

    def adjacent_neighbors(self, n):
        """Returns a list of nodes located in the immediate physical
        vicinity of node n. It does not include "diagonal" neighbours.

        Parameters
        ----------
        n : node
            A node in the graph

        Returns
        -------
        nlist : list
            A list of nodes physically adjacent to n
        """
        adj_list = ((n[0]-1, n[1]), (n[0], n[1]-1), (n[0]+1, n[1]),
                (n[0], n[1]+1))
        adj_neighbors = []  

        for n in adj_list:
            if n in self.nodes():
                adj_neighbors.append(n)

        return adj_neighbors

    def randomgoals(self, n):
        """Retrieve a generator of a node sequence where nodes have been
        randomly sampled ensuring no repetition in every cycle of maze_size**2
        elements or between any two subsequent elements (relevant for the
        transition between cycles.

        Parameters:
        -----------
        n   : int
            Length of sequence to be generated.
        """
        # very first iteration
        n_retrieved = 0
        last_retrieved = None

        while n_retrieved < n:
            if n_retrieved == 0 or n_retrieved % self.maze_size**2 == 0:
                goals = list(self.nodes())
                random.shuffle(goals)

            n_retrieved += 1
            goal = goals.pop()

            if goal != last_retrieved:
                last_retrieved = goal
                yield goal

            else:
                goals.append(goal)
                goal = goals.pop(0)
                last_retrieved = goal
                yield goal

    def randomgoals_alltrials(self, n):
        #FIXME: Doesn't work!
        """Retrieve a generator of a node sequence where nodes have been
        randomly sampled ensuring no repetition of trial types (start, goal)
        in every cycle of (maze_size**2)*(maze_size**2 - 1) elements.

        Parameters:
        -----------
        n   : int
            Length of sequence to be generated.
        """
        # very first iteration
        n_retrieved = 0
        last_retrieved = None

        cycle_length = (self.maze_size**2)*((self.maze_size**2)-1)

        trial_types = {}
        for node in self.nodes():
            other_nodes = list(self.nodes())
            other_nodes.remove(node)
            trial_types[node] = other_nodes

        while n_retrieved < n:
            if n_retrieved % cycle_length == 0:
                # beginning of a new cycle
                trial_type_buffer = copy.deepcopy(trial_types)

                available_first_in_cycle_goals = list(self.nodes())
                if last_retrieved:
                    available_first_in_cycle_goals.remove(last_retrieved)

                goal = random.choice(available_first_in_cycle_goals)
                last_retrieved = goal

            elif not (n_retrieved+1) % cycle_length == 0:
                # number of goals left for last retrieved
                n_left_last = len(trial_type_buffer[last_retrieved])

                goal = trial_type_buffer[last_retrieved].pop(random.randint(
                    0, n_left_last -1))

                if len(trial_type_buffer[goal]) == 0:
                    # it's a 'dead end' and should be saved for the very end
                    trial_type_buffer[last_retrieved].append(goal)
                    continue

            else:
                # it's the last goal in the cycle
                goal = trial_type_buffer[last_retrieved].pop(0)

            last_retrieved = goal
            n_retrieved += 1
            yield goal

    @copydoc(nx.classes.graph.Graph)
    def add_edge(self, u, v, attr_dict=None, **attr):
        super(SquareMaze, self).add_edge(u, v, attr_dict=attr_dict, **attr)
        self._has_changed = True

    @copydoc(nx.classes.graph.Graph)
    def add_edges_from(self, ebunch, attr_dict=None, **attr):
        super(SquareMaze, self).add_edges_from(ebunch, attr_dict=None, **attr)
        self._has_changed = True

    @copydoc(nx.classes.graph.Graph)
    def remove_edge(self, u, v):
        super(SquareMaze, self).remove_edge(u, v)
        self._has_changed = True

    @copydoc(nx.classes.graph.Graph)
    def remove_edges_from(self, ebunch):
        super(SquareMaze, self).remove_edges_from(ebunch)
        self._has_changed = True

    @copydoc(nx.classes.graph.Graph)
    def clear(self):
        super(SquareMaze, self).clear()
        self._has_changed = True

    @copydoc(nx.classes.graph.Graph)
    def add_path(self, nodes, **attr):
        super(SquareMaze, self).add_path(nodes, **attr)
        self._has_changed = True

    # -------- Below are NetworkX methods which are not yet supported --------
    @staticmethod
    def add_node(self):
        raise NodeModification

    @staticmethod
    def add_nodes_from(self):
        raise NodeModification
    #TODO:
    """
    @classmethod
    def remove_node(self):
        raise NodeModification

    @classmethod
    def remove_nodes_from(self):
        raise NodeModification

    @classmethod
    def add_weighted_edges_from(self):
        raise EdgeWeight

    @classmethod
    def add_cycle(self):
        raise NotSupported

    @classmethod
    def add_star(self):
        raise NotSupported
    """

class RandomSquareMaze(SquareMaze):
    """Generator of SuqareMaze with random connections.
    """
    def __new__(self, maze_size, n_connections):
        """Generates a SquareMaze with random connectivity of
        given connectedness.

        Parameters:
        -----------
        maze_size : int
            Maze size. Number of nodes will be maze_size*maze_size.

        n_connections : int
            Number of edges desired in the maze.
        """
        maze = SquareMaze(maze_size)

        edges_to_add = random.sample(maze.possible_edges, n_connections) 

        maze.add_edges_from(edges_to_add)

        return maze

class RandomButConnectedSquareMaze(SquareMaze):
    """Generator for a random maze where all nodes are connected.

    Parameters:
    -----------
    maze_size : int
        Maze size. Number of nodes will be maze_size*maze_size.

    connectedness : float [0,1]
        Fraction of all possible connections that actually exist.
    """
    def __new__(self, maze_size, n_connections):
        """Generates a SquareMaze with random connectivity of
        given connectedness.
        
        Note: This method is not very smart and could be easily
        optimized.

        Parameters:
        -----------
        maze_size : int
            Maze size. Number of nodes will be maze_size*maze_size.

        n_connections : int [(maze_size**2)-1, 2*((maze_size**2)-maze_size)]
            Number of connections
        """
        if n_connections < (maze_size**2)-1 or n_connections > 2*(
                (maze_size**2)-maze_size):
            raise NumberEdgesException

        maze = SquareMaze(maze_size)

        maze.add_edges_from(maze.possible_edges)
        
        edges_to_remove = random.sample(list(maze.edges()),
                maze.n_possible_edges-n_connections)

        maze.remove_edges_from(edges_to_remove)

        while not nx.is_connected(maze):
            connection_to_remove = random.choice(list(
                    maze.edges()))
            connection_to_add = random.choice(
                    maze.absent_edges)
            maze.remove_edge(*connection_to_remove)
            maze.add_edge(*connection_to_add)

        return maze

