"""
Assorted utils.
"""
import numpy as np
import networkx as nx
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

def cos_between_nodes(loc, node1, node2):
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
    
    return np.dot(node1_vec, node2_vec) / (np.linalg.norm(
        node1_vec)*np.linalg.norm(node2_vec))


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

def check_elements_equal(lst):
    """Check that all elements of list are identical.

    Solution from # http://stackoverflow.com/q/3844948/
    pointed out at (together with more solutions): 
    http://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list
    -are-identical

    This implementation is the most efficient of the ones there listed for the
    uses we are giving it.
    """
    return not lst or lst.count(lst[0]) == len(lst)

def copydoc(cls):
    """Copies docstring from a parent class. Code from
    http://article.gmane.org/gmane.comp.python.general/630549"""
    def _fn(fn):
        if fn.__name__ in cls.__dict__:
            fn.__doc__ = cls.__dict__[fn.__name__].__doc__
        return fn
    return _fn

def check_list_is_empty(seq):
    """Check if nested list is empty.
    Solution from (http://stackoverflow.com/questions/1593564/
    python-how-to-check-if-a-nested-list-is-essentially-empty)
    """
    try:
        return all(map(check_list_is_empty, seq))
    except TypeError:
        return False

def generate_angles_dictionary(size):
    from mazex.mazes import RandomSquareMaze

    maze = RandomSquareMaze(size, 1)

    angles = {}
    for l in maze.nodes():
        angles[l] = {}
        n_list = maze.nodes()
        n_list.remove(l)
        for n in n_list:
            angles[l][n] = {}
            for g in n_list:
                angles[l][n][g] = angle_between_nodes(l, n, g)
    return angles

def generate_von_mises_dictionary(size, kappa):
    import pickle
    from scipy.stats import vonmises
    from mazex.mazes import RandomSquareMaze

    maze = RandomSquareMaze(size, 1)

    try:
        filename_vm = str(size)+'_'+str(kappa)+'.pickle'
        f = open(filename_vm, 'rb')
        vm_prob = pickle.load(f)

    except FileNotFoundError:
        try:
            filename = str(size)+'.pickle'
            f = open(filename, 'rb')
            angles = pickle.load(f)

        except FileNotFoundError:
            angles = generate_angles_dictionary(size)
            filename = str(size)+'.pickle'
            with open(filename, 'wb') as f:
                pickle.dump(angles, f)

        vm_prob = {}
        for l in maze.nodes():
            vm_prob[l] = {}
            n_list = maze.nodes()
            n_list.remove(l)
            for n in n_list:
                vm_prob[l][n] = {}
                for g in n_list:
                    prob = vonmises.pdf(angles[l][n][g], kappa=kappa)
                    vm_prob[l][n][g] = prob

        with open(filename_vm, 'wb') as f:
            pickle.dump(vm_prob, f)

    return vm_prob

def invert_dictionary(dic):
    """Return a dictionary with the inverse of the mapping of non-unique
    dictionary dict.
    Adapted to Python 3 from solution found in
    http://stackoverflow.com/questions/483666/python-reverse-inverse-a-mapping
    """
    inv_map = {}
    for k, v in dic.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map

def max_distances(maze, goals=None):
    # the output of this function is the maximum of an array called
    # max_distances_all
    # we first need to create that array based on the three possible scenarios
    # no goal, single goals, multiple goals
    # for each goal calculate the maximum Maze and Manhattan distances, store
    # that in a list, turn list into array and calculate the maximum over the
    # first axis
    if not goals:
        max_maze_distance = nx.algorithms.distance_measures.diameter(maze)
        max_manhattan_distance = 2 * (maze.size-1)
        return (max_maze_distance, max_manhattan_distance)

    # In case only one goal was passed as argument
    # TODO: This is very hacky. improve...
    try:
        assert_is_node(goals)
        goals = [goals]
    except InvalidNode:
        pass

    max_distances_all = []
    
    for goal in goals:
        max_maze_distance = nx.algorithms.distance_measures.eccentricity(maze,
                goal)
        # find largest x distance to this goal
        max_x_dist = max((goal[0], maze.size-1 - goal[0]))
        # same for y
        max_y_dist = max((goal[1], maze.size-1 - goal[1]))
        max_manhattan_distance = max_x_dist + max_y_dist

        max_distances_all.append([max_maze_distance, max_manhattan_distance])

    max_distances_all = np.array(max_distances_all)
    return max_distances_all.max(0)

def to_dist_direction(n, loc, maze, goal):
    """Calculate the direction in the distances space of a given transition
    from loc to n.
    
    I am not sure if it is possible to have something other than
    the 4 possible transitions I have identified so I will make
    conditions very explicit in order to detect eventual
    exceptions
    """
    loc = to_dist_coord(loc, maze, goal)
    n = to_dist_coord(n, maze, goal)

    if n[0]>loc[0] and n[1]>loc[1]:
        "NE transition in our convention for the axis on this diagram"
        return 0
    elif n[0]>loc[0] and n[1]<loc[1]:
        "SE transition in our convention for the axis on this diagram"
        return 1
    elif n[0]<loc[0] and n[1]<loc[1]:
        "SW transition in our convention for the axis on this diagram"
        return 2
    elif n[0]<loc[0] and n[1]>loc[1]:
        "NW transition in our convention for the axis on this diagram"
        return 3
    else:
        raise WeirdDirection

def to_dist_coord(n, maze, goal):
    """Returns the coordinates of node n in distance coordinates in a tuple
    of the shape (maze_distance, manhattan_distance).
    """
    return (int(maze.distances_dictionary[n][goal]),
            abs(goal[0] - n[0]) + abs(goal[1] - n[1]))

