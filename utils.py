"""
Assorted utils.
"""
import numpy as np
import networkx as nx
from mazex.exceptions import *

# These imports are only used by the generate_von_Mises_dictionary function
#  and could be moved to inside the function, if one really wants to cut on
#  the imports
import shelve
from scipy.stats import vonmises

# this import was problematic when left here so I moved it to inside the
# function where it was actually used
#from mazex.mazes import RandomSquareMaze

import os


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
        pass#raise IlegalStep

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

def get_angles_dictionary(size):
    cache_dir = '..mazex'+os.sep+'cache'
    cache_file = cache_dir + os.sep + 'cached_angles'
    cached_object = str(size)

    try:
        dic = shelve.open(cache_file)
        angles = dic[cached_object]
        
    # this is bad practice because it will catch any exception while we only
    #  want to catch it here if it is a 'file not found' exception
    #  however, the 'file not found' exception spat by shelve is not specific
    except:
        print('angles_dictionary_not_found')
        #check that directory exists
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)

        dic = shelve.open(cache_file)

        # and now do the calculations proper
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

        # and store them for later use
        dic[cached_object] = angles
    
    # we close the dictionary so it doesn't stay in memory

    finally:
        dic.close()
        return angles

def get_von_mises_array(size, kappa, decimals=2):
    '''
    Return a 2D array with angles and von Mises distribution values for all
    the angles that could be encountered in a maze with size = size.

    Decimals determines the precision of the array - TO BE IMPLEMENTED!
    '''
    #TODO: implement option decimals!

    angles_dic = get_angles_dictionary(size)

    angles = []

    list_nested_dic_leaves(angles_dic, angles)

    angles = np.sort(angles)

    #remove nans
    angles = angles[~np.isnan(angles)]

    #remove redundant entries
    angles = np.unique(angles)

    vm_array = np.array([angles, vonmises.pdf(angles, kappa=kappa)])

    return vm_array

def get_von_mises_dictionary(size, kappa):
    """Return a dictionary containing the von Mises probabilites for the value
    corresponding to the chosen kappa and angle between any two contiguous
    nodes in the maze
    """
    cache_dir = '.mazex'+os.sep+'cache'
    cache_file = cache_dir + os.sep + 'cached_vm'

    try:
        d = shelve.open(cache_file)

    # this is bad practice because it will catch any exception while we only
    #  want to catch it here if it is a 'file not found' exception
    #  however, the 'file not found' exception spat by shelve is not specific
    except:
        #check that directory exists
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)

        d = shelve.open(cache_file)


    try:
        # check if already calculated for this size-kappa
        size_kappa_dic = d['vm'][size][kappa]

    except KeyError:
        # then we calculate denovo

        # if d['angles'] has not already been instantiated into a dictionary
        #  then do so

        if 'vm' in d.keys():
            vm_dic = d['vm']

            if not size in vm_dic.keys():
                vm_dic[size] = {}

        else:
            vm_dic = {}
            d[size] = {}

        from mazex.mazes import RandomSquareMaze

        angles = get_angles_dictionary(size)

        maze = RandomSquareMaze(size, 1)

        # here we generate the von Mises probabilites dictionary
        # for the value corresponding to the chosen kappa and angle between
        # any two contiguous nodes in the maze
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

        size_kappa_dic = vm_prob

    vm_dic[size][kappa] = size_kappa_dic
    d['vm'] = vm_dic
    d.close()

    return size_kappa_dic

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

def nested_dictionary_depth(dic):
    """Checks the depth of a nested dictionary structure, e.g., dic = {
    'a':{'z':1, 'y':2}, 'b':{'x':3, 'w': 4}}.    

    It currently assumes that the dictionary depth is homogeneous across keys"""

    level = 0

    key_list = list(dic.keys())
    lower_dic = dic[key_list[0]]
    is_dic = True

    while is_dic:
        try:
            key_list_lower_dic = list(lower_dic.keys())

            lower_dic = lower_dic[key_list_lower_dic[0]]
            level += 1

        except AttributeError:
            is_dic = False

    return level

"""
def list_angles_in_dictionary(dic):
    '''Lists all angles in a nested dictionary containing angles between nodes
    .'''
    values = list(dic.values())
    
    levels_to_go = nested_dictionary_depth(dic)

    angles = []

    while levels_to_go:
        for d in
"""

def list_nested_dic_leaves(d, l):
    for k, v in d.items():
        if isinstance(v, dict):
            list_nested_dic_leaves(v, l)
        else:
            l.append(v)

def manhattan_distance(node1, node2):
    """Returns Manhattan distance between node1 and node2."""
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

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

