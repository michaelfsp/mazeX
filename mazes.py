import numpy as np
import networkx as nx
import random

from mazex.exceptions import *

class SquareMaze(nx.classes.graph.Graph):
    """Generic class of for square mazes.
    """
    def __init__(self, size):
        """Generates a square maze of given size and no connections.
        Total number of nodes will size**2.
        """

        # TODO: Check whether the graph instance is a possible maze
        # configuration
        G = nx.Graph()
        
        for i in np.ndindex((size, size)):
            G.add_node(i, attr_dict={'pos':i})

        super(SquareMaze, self).__init__(G)
        self.size = size

        # We will calculate the distances matrix once and then only
        # recalculate if either the nodes or eges of the maze have changed
        # We save the snapshot of both at time of calculation in these objects
        self._distances_matrix_nodes = None
        self._distances_matrix_edges = None
        self._distances_matrix = None

    @property
    def distances_matrix(self):
        # If the graph has not changed since last calculation of the matrix:
        if (self._distances_matrix_nodes == self.nodes() and
                self._distances_matrix_edges == self.edges()):
            return self._distances_matrix
        else:
            self._distances_matrix = nx.algorithms.floyd_warshall_numpy(self).astype(int)
            self._distances_matrix_nodes = self.nodes()
            self._distances_matrix_edges = self.edges()

            return self._distances_matrix
    
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

