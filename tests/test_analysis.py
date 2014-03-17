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

def test_avg_dist_dir_prob():
    import numpy as np
    import mazex as mx
    maze = mx.SquareMaze(4)
    agent = mx.Agent((0,0), 1)
    edgeslist = [((0,0), (0,1)), ((0,1), (1,1)), ((1,1), (1,0)),
            ((1,0), (2,0)), ((2,0), (2,1)), ((2,1), (3,1)), ((3,1), (3,0)),
            ((3,1), (3,2)), ((3,2), (3,3)), ((1,1), (1,2)), ((1,2), (2,2)),
            ((2,2), (2,3)), ((2,3), (1,3)), ((1,3), (0,3)), ((0,3), (0,2))]
    maze.add_edges_from(edgeslist)
    path1 = [(0,0), (0,1), (1,1), (1,2), (2,2)]
    path2 = [(2,2), (1,2), (1,1), (1,0), (2,0), (2,1), (3,1), (3,2), (3,3)]
    path3 = [(3,3), (3,2), (3,1), (2,1), (2,0), (1,0), (1,1), (1,2), (2,2),
            (2,3)]
    path4 = [(2,3), (1,3), (0,3), (0,2)]
    paths = [path1, path2, path3, path4]
    trials = []
    for path in paths:
        steps = [mx.Step(maze, x, path[-1]) for x in path]
        trial = mx.Trial(steps, agent)
        #HACK: add a previous to each step
        for p, s in zip(range(0, len(trial.steps)), trial.steps[1:]):
            s.previous = trial.steps[p]
        trials.append(trial)
    session = mx.Session(agent, trials)
    # total moves per dist dir
    moves = np.array([0, 0, 17, 7])
    # total number of times available per dist dir
    availability = np.array([16, 7, 17, 7])
    # sum of n-1 for every occurrence
    n_minus_one = np.array([17, 8, 17, 3])

    simple_prob = moves/availability

    assert np.all(simple_prob == mx.analysis.avg_dist_dir_prob(session,
        plot=False))


