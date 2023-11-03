"""
Assorted utils.
"""
import functools
import numpy as np
import math
import networkx as nx
from mazex.exceptions import *

import copy
import mazex as mx

# These imports are only used by the generate_von_Mises_dictionary function
#  and could be moved to inside the function, if one really wants to cut on
#  the imports
import shelve
from scipy.stats import vonmises
from scipy.spatial.distance import euclidean

# this import was problematic when left here so I moved it to inside the
# function where it was actually used
# from mazex.mazes import RandomSquareMaze

import os
from functools import lru_cache


@lru_cache(maxsize=65536)
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

    return np.arccos(
        np.dot(node1_vec, node2_vec)
        / (np.linalg.norm(node1_vec) * np.linalg.norm(node2_vec))
    )


@lru_cache(maxsize=65536)
def angle_between_node_and_heading(loc, node1, heading):
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
    if heading == ">":
        heading_node = (loc[0] + 1, loc[1])
    elif heading == "v":
        heading_node = (loc[0], loc[1] - 1)
    elif heading == "<":
        heading_node = (loc[0] - 1, loc[1])
    elif heading == "^":
        heading_node = (loc[0], loc[1] + 1)
    else:
        raise InvalidHeading

    return angle_between_nodes(loc, node1, heading_node)


@lru_cache(maxsize=128)
def angles_dictionary(size):
    from mazex.mazes import SquareMaze

    maze = SquareMaze(size)

    angles = {}
    for l in maze.nodes():
        angles[l] = {}
        n_list = list(maze.nodes())
        n_list.remove(l)
        for n in n_list:
            angles[l][n] = {}
            g_list = list(maze.nodes())
            g_list.remove(l)
            g_list.remove(n)
            for g in g_list:
                angles[l][n][g] = angle_between_nodes(l, n, g)
    return angles


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

    return np.dot(node1_vec, node2_vec) / (
        np.linalg.norm(node1_vec) * np.linalg.norm(node2_vec)
    )


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
    if np.sqrt((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2) == 1:
        return True
    else:
        # TODO: Fix this ugly hack!
        pass  # raise IlegalStep


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


# 2520 is the number of possible combinations for a maze of 6x6
# speed gains of caching this are in the order of only 5%
# @lru_cache(maxsize=2520)
# FIXME: refactor code to fix typo in name of this function...
def euclidian_distance(node1, node2):
    """Returns Manhattan distance between node1 and node2."""
    return np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


def get_angles_dictionary(size):
    """This method is deprecated. I am now using lru_cache."""
    cache_dir = "..mazex" + os.sep + "cache"
    cache_file = cache_dir + os.sep + "cached_angles"
    cached_object = str(size)

    try:
        dic = shelve.open(cache_file)
        angles = dic[cached_object]

    # this is bad practice because it will catch any exception while we only
    #  want to catch it here if it is a 'file not found' exception
    #  however, the 'file not found' exception spat by shelve is not specific
    except:
        print("angles_dictionary_not_found")
        # check that directory exists
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)

        dic = shelve.open(cache_file)

        angles = angles_dictionary(size)

        # and store them for later use
        dic[cached_object] = angles

    # we close the dictionary so it doesn't stay in memory

    finally:
        dic.close()
        return angles


def get_von_mises_array(size, kappa, decimals=2):
    """
    Return a 2D array with angles and von Mises distribution values for all
    the angles that could be encountered in a maze with size = size.

    Decimals determines the precision of the array - TO BE IMPLEMENTED!
    """
    # TODO: implement option decimals!

    angles_dic = get_angles_dictionary(size)

    angles = []

    list_nested_dic_leaves(angles_dic, angles)

    angles = np.sort(angles)

    # remove nans
    angles = angles[~np.isnan(angles)]

    # remove redundant entries
    angles = np.unique(angles)

    vm_array = np.array([angles, vonmises.pdf(angles, kappa=kappa)])

    return vm_array


def get_von_mises_dictionary(size, kappa):
    """Return a dictionary containing the von Mises probabilites for the value
    corresponding to the chosen kappa and angle between any two contiguous
    nodes in the maze
    """
    cache_dir = ".mazex" + os.sep + "cache"
    cache_file = cache_dir + os.sep + "cached_vm"

    try:
        d = shelve.open(cache_file)

    # this is bad practice because it will catch any exception while we only
    #  want to catch it here if it is a 'file not found' exception
    #  however, the 'file not found' exception spat by shelve is not specific
    except:
        # check that directory exists
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)

        d = shelve.open(cache_file)

    try:
        # check if already calculated for this size-kappa
        size_kappa_dic = d["vm"][size][kappa]

    except KeyError:
        # then we calculate denovo

        # if d['angles'] has not already been instantiated into a dictionary
        #  then do so

        if "vm" in d.keys():
            vm_dic = d["vm"]

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
            n_list = list(maze.nodes())
            n_list.remove(l)
            for n in n_list:
                vm_prob[l][n] = {}
                for g in n_list:
                    prob = vonmises.pdf(angles[l][n][g], kappa=kappa)
                    vm_prob[l][n][g] = prob

        size_kappa_dic = vm_prob

    vm_dic[size][kappa] = size_kappa_dic
    d["vm"] = vm_dic
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


def number_possible_mazes(size, n_edges):
    """Function to calculate the maximum possible number of mazes of size =
    size and n_edges.
    This will not discount for mazes which are equivalente based on symmetry
    or restrict only to fully connected mazes.
    """
    n_possible_edge_locations = 2 * (size * (size - 1))
    return int(
        math.factorial(n_possible_edge_locations)
        / (
            math.factorial(n_possible_edge_locations - n_edges)
            * math.factorial(n_edges)
        )
    )


def number_connected_non_isomorphic_mazes(size, n_edges, n_samples=10000):
    """This function tries to estimate the number of existing connected mazes
    of a given size that are non-isomorphic, i.e., excluding mazes which are
    equivalent by some internal symmetry through a Monte Carlo sampling
    approach.

    The algorithm is the following:
        Sample n_edges from the list of possible square mazes of size = size
        Append that to mazes list
        Check if it is a repetition
            if: break
            else: check the following:
                if it is disconnected
                if it is isomorphic to any maze on mazes list
        Back to step one

        Count number of disconnected and isomorphisms and accepted mazes (what
        is left after those checks)

        Calculate the fraction of accepted mazes and multiply that by the total
        number of possible mazes."""
    n_possible_mazes = number_possible_mazes(size, n_edges)
    # we initiate the counters that will allow us to estimate the fractions
    # we only need one of these but will keep both for a sanity check
    n_excluded_not_connected = 0
    n_excluded_isomorphic = 0
    n_excluded_is_same = 0
    n_accepted = 0

    mazes = []
    # Generate first maze
    maze = mx.RandomSquareMaze(size, n_edges)
    mazes.append(maze)
    # now we check to which category we count the first maze
    # first we check if it is disconnected:
    if not nx.is_connected(maze):
        n_excluded_not_connected += 1
    else:
        n_accepted += 1

    sample_n = 1
    while sample_n < n_samples:
        maze = mx.RandomSquareMaze(size, n_edges)

        # we check if this is not a repetition
        maze_edges = list(maze.edges())
        maze_edges.sort()
        for m in mazes:
            m_edges = list(m.edges())
            m_edges.sort()

            if m_edges == maze_edges:
                # it is a repetition, we count it and go on to the next maze
                n_excluded_is_same += 1
                break  # breaks search for same maze
        else:
            # this is a novel maze
            # NOTE: this else belongs to for!!
            # this block of code is only executed if the above for is exausted
            # and it hasn't broken by maze repetition

            # it is not a repetition and we proceed
            # we now check it is connected
            if not nx.is_connected(maze):
                n_excluded_not_connected += 1
                sample_n += 1
                mazes.append(maze)
                continue  # go on with the next maze

            for m in mazes:
                if nx.is_isomorphic(m, maze):
                    n_excluded_isomorphic += 1
                    sample_n += 1
                    mazes.append(maze)
                    break
            else:
                # this maze is a survivor!
                # neither a repetition, nor disconnected, nor isomorphic
                n_accepted += 1
                sample_n += 1
                mazes.append(maze)

    fraction_accepted = n_accepted / (
        n_accepted + n_excluded_not_connected + n_excluded_isomorphic
    )

    print(
        "N_accepted: %i \nN_same: %i \nN_not_connected: %i \nN_isomorphic: %i"
        % (
            n_accepted,
            n_excluded_is_same,
            n_excluded_not_connected,
            n_excluded_isomorphic,
        )
    )

    print("\nDifferent mazes searched: %i" % len(mazes))
    print(
        "\nOrder of magnitude of mazes: %f"
        % np.log10(fraction_accepted * n_possible_mazes)
    )

    return fraction_accepted * n_possible_mazes


def number_connected_non_isomorphic_mazes_with_rep(size, n_edges, n_samples=10000):
    """This function tries to estimate the number of existing connected mazes
    of a given size that are non-isomorphic, i.e., excluding mazes which are
    equivalent by some internal symmetry through a Monte Carlo sampling
    approach.

    The algorithm is the following:
        Sample n_edges from the list of possible square mazes of size = size
        Append that to mazes list
        Check if it is a repetition
            if: break
            else: check the following:
                if it is disconnected
                if it is isomorphic to any maze on mazes list
        Back to step one

        Count number of disconnected and isomorphisms and accepted mazes (what
        is left after those checks)

        Calculate the fraction of accepted mazes and multiply that by the total
        number of possible mazes."""
    n_possible_mazes = number_possible_mazes(size, n_edges)
    # we initiate the counters that will allow us to estimate the fractions
    # we only need one of these but will keep both for a sanity check
    n_excluded_not_connected = 0
    n_excluded_isomorphic = 0
    n_excluded_is_same = 0
    n_accepted = 0

    mazes = []
    # Generate first maze
    maze = mx.RandomSquareMaze(size, n_edges)
    mazes.append(maze)
    # now we check to which category we count the first maze
    # first we check if it is disconnected:
    if not nx.is_connected(maze):
        n_excluded_not_connected += 1
    else:
        n_accepted += 1

    sample_n = 1
    while sample_n < n_samples:
        maze = mx.RandomSquareMaze(size, n_edges)

        if not nx.is_connected(maze):
            n_excluded_not_connected += 1
            sample_n += 1
            mazes.append(maze)
            continue  # go on with the next maze

        for m in mazes:
            if nx.is_isomorphic(m, maze):
                # we check that the isomorphic maze is not itself
                maze_edges = list(maze.edges())
                maze_edges.sort()

                m_edges = list(m.edges())
                m_edges.sort()

                if m_edges == maze_edges:
                    # it is a repetition, we count it and try to see if there
                    #  are nevertheless isomorphisms with other mazes already
                    #  in the list
                    n_excluded_is_same += 1
                    continue  # carries on finding isomorphisms

                else:
                    # it is a true isomorphism
                    n_excluded_isomorphic += 1
                    sample_n += 1
                    mazes.append(maze)
                    break
        else:
            # this maze is a survivor!
            # neither disconnected, nor isomorphic but it might be a repetition
            n_accepted += 1
            sample_n += 1
            mazes.append(maze)

    fraction_accepted = n_accepted / (
        n_accepted + n_excluded_not_connected + n_excluded_isomorphic
    )

    print(
        "N_accepted: %i \nN_same: %i \nN_not_connected: %i \nN_isomorphic: %i"
        % (
            n_accepted,
            n_excluded_is_same,
            n_excluded_not_connected,
            n_excluded_isomorphic,
        )
    )

    print("\nDifferent mazes searched: %i" % len(mazes))
    print(
        "\nOrder of magnitude of mazes: %f"
        % np.log10(fraction_accepted * n_possible_mazes)
    )

    return fraction_accepted * n_possible_mazes


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
        max_manhattan_distance = 2 * (maze.maze_size - 1)
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
        max_maze_distance = nx.algorithms.distance_measures.eccentricity(maze, goal)
        # find largest x distance to this goal
        max_x_dist = max((goal[0], maze.maze_size - 1 - goal[0]))
        # same for y
        max_y_dist = max((goal[1], maze.maze_size - 1 - goal[1]))
        max_manhattan_distance = max_x_dist + max_y_dist

        max_distances_all.append([max_maze_distance, max_manhattan_distance])

    max_distances_all = np.array(max_distances_all)
    return max_distances_all.max(0)


def find_pareto_front(point_cloud):
    """Find pareto front by sorting point_cloud in decending x coordinate,
    adding the first point to the front, then going down the points and adding
    to the front each point whose y coordinate is larger than that of the
    last point on the front.

    Written by Thomas Akam
    """
    sorted_point_cloud = point_cloud[(-point_cloud[:, 0]).argsort()]
    pareto_front = [sorted_point_cloud[0]]
    for point in sorted_point_cloud[1:, :]:
        if point[1] > pareto_front[-1][1]:
            pareto_front.append(point)
    return np.array(pareto_front)


def find_pareto_fronts(point_cloud, n_fronts):
    """Find a set of fronts

    Written by Thomas Akam
    """
    sorted_point_cloud = point_cloud[(-point_cloud[:, 0]).argsort()]
    pareto_fronts = [[(None, -np.inf)] for _ in range(n_fronts)]
    for point in sorted_point_cloud:
        for front in pareto_fronts:
            if point[1] > front[-1][1]:
                front.append(point)
                break
    pareto_fronts = [np.array(front[1:]) for front in pareto_fronts]
    return pareto_fronts


def find_pareto_front_by_thirds(cost_values, n=15):
    """Return the indices of elements on the Pareto front, split by thirds.

    Array should be 2xN
    """
    # center the values
    centered_cost_values = cost_values - np.transpose([cost_values.mean(axis=1)])
    # normalize the values to max
    maxima = np.max(centered_cost_values, axis=1)
    centered_cost_values /= np.transpose([maxima])

    # NOTE: notice this renaming
    cost_values = centered_cost_values

    # we get the indices of elements on the positive quadrant
    positive_indices = np.nonzero(np.all(centered_cost_values > 0, axis=0))[0]

    # here we haven't yet constrained it to the positive values!
    upper_third_i = np.nonzero(
        cost_values[1, :]
        > (np.sin(2 * np.pi / 6) / np.cos(2 * np.pi / 6)) * cost_values[0, :]
    )
    lower_third_i = np.nonzero(
        cost_values[1, :]
        < (np.sin(1 * np.pi / 6) / np.cos(1 * np.pi / 6)) * cost_values[0, :]
    )
    middle_third_i = np.nonzero(
        (
            cost_values[1, :]
            > (np.sin(1 * np.pi / 6) / np.cos(1 * np.pi / 6)) * cost_values[0, :]
        )
        & (
            cost_values[1, :]
            < (np.sin(2 * np.pi / 6) / np.cos(2 * np.pi / 6)) * cost_values[0, :]
        )
    )

    # now we get indices that are on each third of the positive quadrant
    pos_upper_third_i = np.intersect1d(positive_indices, upper_third_i)
    pos_lower_third_i = np.intersect1d(positive_indices, lower_third_i)
    pos_middle_third_i = np.intersect1d(positive_indices, middle_third_i)

    # we get the array of sorting indices in reverse order so we can then
    # iterate over them
    sorting_i = np.flipud(np.argsort(np.sum(centered_cost_values**2, axis=0)))

    pareto_front_upper_i = []
    pareto_front_lower_i = []
    pareto_front_middle_i = []

    # TODO: fix bug that causes the length of the different outputs to be
    # different
    j = 0
    while (
        len(pareto_front_lower_i) < n
        or len(pareto_front_middle_i) < n
        or len(pareto_front_upper_i) < n
    ):
        if sorting_i[j] in pos_upper_third_i:
            pareto_front_upper_i.append(sorting_i[j])
        elif sorting_i[j] in pos_lower_third_i:
            pareto_front_lower_i.append(sorting_i[j])
        elif sorting_i[j] in pos_middle_third_i:
            pareto_front_middle_i.append(sorting_i[j])
        j += 1

    return pareto_front_upper_i, pareto_front_lower_i, pareto_front_middle_i


def rmse(predictions, targets):
    """Calculates Root Mean Squared Error."""
    # the following might be faster than the current implementation
    # n = len(predictions)
    # rmse = np.linalg.norm(predictions - targets) / np.sqrt(n)
    # return rmse
    if type(predictions) is list:
        predictions = np.array(predictions)

    if type(targets) is list:
        targets = np.array(targets)

    return np.sqrt(((predictions - targets) ** 2).mean())


def distance_along_line(array):
    """Given a set of points, return a tuple with sections along the line.

    Assumes that array is sorted.

    Returns indices only
    """
    array = copy.deepcopy(array)
    # First we need to normalize the array
    # find the largest dimension axis
    long_axis = np.argmax(array.shape)
    min_vals = np.min(array, axis=long_axis)
    array -= min_vals
    max_vals = np.max(array, axis=long_axis)
    array /= max_vals

    distance_along_line = np.zeros(max(array.shape))
    for i, point in enumerate(array[:-1]):
        # Calculate distance to previous point
        d_to_next = euclidean(array[i], array[i + 1])

        # sum that distance to the cumulative distance (running integral)
        total_d = d_to_next + distance_along_line[i]
        distance_along_line[i + 1] = total_d

    return distance_along_line


def sectioned_pareto_front(array, n_sections):
    """Given a set of points, return a tuple with sections along the line.

    Assumes that array is sorted.

    Returns indices only
    """
    array = copy.deepcopy(array)
    # First we need to normalize the array
    # find the largest dimension axis
    long_axis = np.argmax(array.shape)
    min_vals = np.min(array, axis=long_axis)
    array -= min_vals
    max_vals = np.max(array, axis=long_axis)
    array /= max_vals

    distance_along_line = np.zeros(max(array.shape))
    for i, point in enumerate(array[:-1]):
        # Calculate distance to previous point
        d_to_next = euclidean(array[i], array[i + 1])

        # sum that distance to the cumulative distance (running integral)
        total_d = d_to_next + distance_along_line[i]
        distance_along_line[i + 1] = total_d

    line_length = distance_along_line[-1]

    section_length = line_length / n_sections

    section_limits = [i * section_length for i in range(n_sections + 1)]

    section_indices = np.array(np.zeros(max(array.shape)))
    current_index = 0
    for i, d in enumerate(distance_along_line):
        if d > section_limits[current_index + 1]:
            current_index += 1
        section_indices[i] = current_index

    return section_indices


def scale_dict_values(dic, high=1.0, low=0.0):
    """Scale the values in a dictionary to a [low, high] range while
    preserving the key->value pairing.
    """
    dic_val_array = np.array(list(dic.values()))
    dic_scaled_val_array = scale_vector(dic_val_array, high=high, low=low)
    scaled_dic = dict((k, v) for k, v in zip(dic.keys(), dic_scaled_val_array))
    return scaled_dic


def scale_vector(rawvector, high=1.0, low=0.0):
    minval = min(rawvector)
    maxval = max(rawvector)
    rng = maxval - minval
    return high - (((high - low) * (maxval - rawvector)) / rng)


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

    if n[0] > loc[0] and n[1] > loc[1]:
        # FIXME: might be wrong?" NE transition in our convention for the axis on this diagram"
        return 0
    elif n[0] > loc[0] and n[1] < loc[1]:
        # FIXME: might be wrong?" SE transition in our convention for the axis on this diagram"
        return 1
    elif n[0] < loc[0] and n[1] < loc[1]:
        # FIXME: might be wrong?" SW transition in our convention for the axis on this diagram"
        return 2
    elif n[0] < loc[0] and n[1] > loc[1]:
        # FIXME: might be wrong? "NW transition in our convention for the axis on this diagram"
        return 3
    else:
        raise WeirdDirection


def to_dist_direction_euclidian(n, loc, maze, goal):
    """Calculate the direction in the distances space of a given transition
    from loc to n.

    I am not sure if it is possible to have something other than
    the 4 possible transitions I have identified so I will make
    conditions very explicit in order to detect eventual
    exceptions
    """
    loc_G = to_dist_coord_euclidian(loc, maze, goal)
    n = to_dist_coord_euclidian(n, maze, goal)

    dists_available = np.array(
        [to_dist_coord_euclidian(a, maze, goal) for a in maze.neighbors(loc)]
    )

    best_euclidian = np.min(dists_available[:, 1])

    if n[0] > loc_G[0] and n[1] != best_euclidian:
        "NE transition in our convention for the axis on this diagram"
        return 0
    elif n[0] > loc_G[0] and n[1] == best_euclidian:
        "SE transition in our convention for the axis on this diagram"
        return 1
    elif n[0] < loc_G[0] and n[1] == best_euclidian:
        "SW transition in our convention for the axis on this diagram"
        return 2
    elif n[0] < loc_G[0] and n[1] != best_euclidian:
        "NW transition in our convention for the axis on this diagram"
        return 3
    else:
        raise WeirdDirection


def to_dist_coord(n, maze, goal):
    """Returns the coordinates of node n in distance coordinates in a tuple
    of the shape (maze_distance, manhattan_distance).
    """
    return (
        int(maze.distances_dictionary[n][goal]),
        abs(goal[0] - n[0]) + abs(goal[1] - n[1]),
    )


def to_dist_coord_euclidian(n, maze, goal):
    """Returns the coordinates of node n in distance coordinates in a tuple
    of the shape (maze_distance, manhattan_distance).
    """
    return (
        int(maze.distances_dictionary[n][goal]),
        np.sqrt((goal[0] - n[0]) ** 2 + (goal[1] - n[1]) ** 2),
    )


@lru_cache(maxsize=36)
def to_peano_coords(node):
    """Returns the coordinate of a node on a Peano space-filling curve."""
    peano_maze = mx.examplemazes.peano_maze
    coord = peano_maze.distances_dictionary[(0, 0)][node]
    return int(coord)


def euclidean_drive(s, s_T, m_over_n=1):
    """
    s: previous node
    s_T: goal node
    m_over_n: ration between m and n parameters of drive function
    """
    return euclidian_distance(s, s_T) ** m_over_n


def sdd_path(path, m_over_n=1, gamma=1, s_T=None):
    """Given a path over a trial, returns the sum discounted distance.
    Parameters:
         path: list of locations

         m_over_n: ratio between m and n parameters from the HRL paper equations. 1 is equal to just taking euclidean distance

        gamma: temporal discount factor
    """

    if not s_T:
        s_T = path[-1]

    sdd = 0
    for i, _ in enumerate(path[:-1]):
        s_prime = path[i + 1]
        s_prime_euclidean_drive = euclidean_drive(s_prime, s_T, m_over_n)
        sdd += gamma**i * s_prime_euclidean_drive
    return sdd


def pad_with_nan(lst, max_len=None):
    """Pad list of lists with nans so that that every row has same length.

    Parameters:
        max_len: length of the longest row. if not specified it will be computed. pre-specifying will make it faster
    """
    if not max_len:
        # Find the maximum length
        max_len = max(len(sublist) for sublist in lst)

    # Pad each sublist with NaN to make them all of the same length
    padded_list = [sublist + [np.nan] * (max_len - len(sublist)) for sublist in lst]

    # Convert to numpy array
    return np.array(padded_list)
