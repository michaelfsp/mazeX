import sys
# so we can import mazex.
sys.path.append('../../')
import mazex as mx
from mazex.utils import check_list_is_empty

import copy

def test_distances_to_coord():
    maze = mx.RandomButConnectedSquareMaze(6, 40)
    inverse_dic = copy.deepcopy(maze.distances_to_coord)
    distances_dic = copy.deepcopy(maze.distances_dictionary)

    for n1 in maze.nodes():
        for n2 in maze.nodes():
            distance = mx.analysis.to_dist_coord(n1, maze, n2)
            assert n2 in inverse_dic[distance][n1]
            inverse_dic[distance][n1].remove(n2)

    all_nodes_left = [list(inverse_dic[k].values()) for k in inverse_dic.keys(
        )]

    assert check_list_is_empty(all_nodes_left)
