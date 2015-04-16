import numpy as np
import networkx as nx
import random
import copy
from functools import lru_cache

from mazex.exceptions import *
from mazex.utils import copydoc
import mazex.analysis

#TODO: save centrality measures as attributes of edges and nodes
@lru_cache(maxsize=10)
def cached_betweenness_centrality(maze):
    return nx.algorithms.betweenness_centrality(maze)

@lru_cache(maxsize=10)
def cached_edge_betweenness_centrality(maze):
    return nx.algorithms.edge_betweenness_centrality(maze)

@lru_cache(maxsize=10)
def cached_closeness_centrality(maze):
    return nx.algorithms.closeness_centrality(maze)

class SquareMaze(nx.classes.graph.Graph):
    """Generic class of for square mazes.
    """
    def __init__(self, size):
        """Generates a square maze of given size and no connections.
        Total number of nodes will size**2.
        """
        super(SquareMaze, self).__init__()

        for i in np.ndindex((size, size)):
            super(SquareMaze, self).add_node(i, attr_dict={'pos':i})

        self.size = size

        # We will calculate the distances matrix once and then only
        # recalculate if either the nodes or eges of the maze have changed
        # We save the snapshot of both at time of calculation in these objects
        self._has_changed_since_matrix = True
        self._has_changed_since_dictionary = True
        self._has_changed_since_to_coord_map = True

        self._distances_matrix = None
        self._distances_dictionary = None
        self._distances_to_coord_map = None

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
    def distances_matrix(self):
        # If the graph has not changed since last calculation of the matrix:
        if not self._has_changed_since_matrix:
            return self._distances_matrix
        else:
            # for some reason the NetworkX algorithm produces a numpy matrix
            # instead of an array, so we np.array it
            self._distances_matrix = np.array(
                    nx.algorithms.floyd_warshall_numpy(self).astype(int))
            self._has_changed_since_matrix = False
            return self._distances_matrix

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
    def pos(self):
        return nx.get_node_attributes(self, 'pos')

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
        return 2*((self.size**2)-self.size)

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
        randomly sampled ensuring no repetition in every cycle of size**2
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
            if n_retrieved == 0 or n_retrieved % self.size**2 == 0:
                goals = self.nodes()
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
    def __new__(self, size, connectedness):
        """Generates a SquareMaze with random connectivity of
        given connectedness.

        Parameters:
        -----------
        size : int
            Maze size. Number of nodes will be size*size.

        connectedness : float [0,1]
            Fraction of all possible connections that actually exist.
        """
        maze = SquareMaze(size)

        edges_to_add = random.sample(maze.possible_edges, round(
            maze.n_possible_edges*connectedness)) 

        maze.add_edges_from(edges_to_add)

        return maze

class RandomButConnectedSquareMaze(SquareMaze):
    """Generator for a random maze where all nodes are connected.

    Parameters:
    -----------
    size : int
        Maze size. Number of nodes will be size*size.

    connectedness : float [0,1]
        Fraction of all possible connections that actually exist.
    """
    def __new__(self, size, n_connections):
        """Generates a SquareMaze with random connectivity of
        given connectedness.
        
        Note: This method is not very smart and could be easily
        optimized.

        Parameters:
        -----------
        size : int
            Maze size. Number of nodes will be size*size.

        n_connections : int [(size**2)-1, 2*((size**2)-size)]
            Number of connections
        """
        if n_connections < (size**2)-1 or n_connections > 2*(
                (size**2)-size):
            raise NumberEdgesException

        maze = SquareMaze(size)

        maze.add_edges_from(maze.possible_edges)
        
        edges_to_remove = random.sample(maze.edges(),
                maze.n_possible_edges-n_connections)

        maze.remove_edges_from(edges_to_remove)

        while not nx.is_connected(maze):
            connection_to_remove = random.choice(
                    maze.edges())
            connection_to_add = random.choice(
                    maze.absent_edges)
            maze.remove_edge(*connection_to_remove)
            maze.add_edge(*connection_to_add)

        return maze

