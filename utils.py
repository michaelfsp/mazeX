"""
Assorted utils.
"""
import numpy as np
from mazex.exceptions import *

def angle_between_nodes(loc, node1, node2):
    """A common function that determines the angle between a given edge and
    the goal.

    Parameters:
    -----------
    node : node or tuple
        Node on the maze we want to evaluate the angle of
    
    loc : node or tuple
        The current node
    
    goal : node or turple
        The node of the ,y coordinates of the goal location.
    """
    node1_vec = np.array(node1) - np.array(loc)
    node2_vec = np.array(node2) - np.array(loc)
    
    return np.arccos(np.dot(node1_vec, node2_vec) / (np.linalg.norm(
        node1_vec)*np.linalg.norm(node2_vec)))

def assert_is_node(node):
    # test the node follows our node naming convention
    if not len(node) == 2:
        raise InvalidNode
    elif not type(node[0]) is int or not type(node[1]) is int:
        raise InvalidNode
    elif node[0] < 0 or node[1] < 0:
        raise InvalidNode
    else:
        return True

def assert_nodes_adjacent(node1, node2):
    """Determine whether node1 and node2 are in the immediate vicinity,
    simply using Euclidian distance. Note that it does not check whether
    this node exists on the maze nor whether the path between node1 and
    node2 is free.
    Note that transition to same node (staying in place) is not allowed.
    """
    assert_is_node(node1)
    assert_is_node(node2)
    if np.sqrt((node2[0] - node1[0])**2 + (node2[1] - node1[1])**2) == 1:
        return True
    else:
        raise IlegalStep

