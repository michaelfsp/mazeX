
import mazex as mx
import mazex.utils as utils
"""
angles = []
maze = mx.RandomSquareMaze(6, 1)
for l in maze.nodes():
    n1_list = maze.nodes()
    n1_list.remove(l)
    for n in n1_list:
        g_list = maze.nodes()
        g_list.remove(l)
        g_list.remove(n)
        for g in g_list:
            angles.append(utils.angle_between_nodes(l, n, g))
"""

"""
dg_trajectory = [8, 7, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
sigma_trajectory = []
while len(sigma_trajectory) != len(dg_trajectory)-1:
    i = len(sigma_trajectory)

    if dg_trajectory[i] > dg_trajectory[i+1]:
        # if the distance drecreased, the agent acted optimally
        sigma_trajectory.append(0)
    elif dg_trajectory[i] < dg_trajectory[i+1]:
        # otherwise we need to find how big the detour was
        # so we search the distances list starting from the point the
        # distance increased until we find the same distance again. The
        # index of that occurrence is the number of extra steps the agent
        # will have to do because of its stupidity here, as compared to an
        # optimal agent
        extra_steps = dg_trajectory[i+1:].index(dg_trajectory[i]) + 1

        # we define the deviation sigma
        sigma_trajectory.append(int(extra_steps/2))

        del dg_trajectory[i+1:i+extra_steps+1]
"""
"""
connections = [[(0,0), (1,0)], [(1,0), (2,0)], [(2,0), (3,0)], [(3,0), (4,0)],
        [(0,0), (0,1)], [(4,0), (4,1)], [(0,1), (1,1)], [(1,1), (2,1)],
        [(3,1), (4,1)], [(0,1), (0,2)], [(1,1), (1,2)], [(2,1), (2,2)],
        [(4,1), (4,2)], [(0,2), (1,2)], [(1,2), (2,2)], [(2,2), (3,2)],
        [(3,2), (4,2)], [(1,2), (1,3)], [(2,2), (2,3)], [(3,2), (3,3)],
        [(0,3), (1,3)], [(2,3), (3,3)], [(3,3), (4,3)], [(0,3), (0,4)],
        [(2,3), (2,4)], [(3,3), (3,4)], [(4,3), (4,4)], [(0,4), (1,4)],
        [(1,4), (2,4)], [(2,4), (3,4)]]
"""
"""
def extract(dict_in, dict_out):
    for key, value in dict_in.items():
        if isinstance(value, dict): # If value itself is dictionary
            extract(value, dict_out)
        elif isinstance(value, unicode):
            # Write to dict_out
            dict_out[key] = value
    return dict_out

def traverse(o, tree_types=(list, tuple)):
    if isinstance(o, tree_types):
        for value in o:
            for subvalue in traverse(value):
                yield subvalue
    else:
        yield o
"""

def myprint(d, l):
    for k, v in d.items():
        if isinstance(v, dict):
            myprint(v, l)
        else:
            l.append(v)
