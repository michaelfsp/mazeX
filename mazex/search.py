import numpy as np
import mazex as mx


def _euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def strict_topology_optimization_euclidean_reduction(
    maze, min_avail_nodes=3, as_fraction=True
):
    """Like strict_topology_optimization but the vector navigation agent
    selects the neighbor that maximally reduces Euclidean distance to goal:
        value(n) = d_euclidean(location, goal) - d_euclidean(n, goal)
    """
    magent = mx.agents.OmniscientAgent(maze)
    n_discordant_points = 0
    goals = maze.nodes()
    for g in goals:
        trial = mx.Trial(maze, g)
        eval_points = list(maze.nodes())
        eval_points.remove(g)
        for p in eval_points:
            step = mx.Step(p, trial)
            if len(step.available_nodes) >= min_avail_nodes:
                # Vector navigation: maximize reduction in Euclidean distance
                d_pg = _euclidean_distance(p, g)
                h_vals = np.array(
                    [d_pg - _euclidean_distance(n, g) for n in step.available_nodes]
                )
                h_chosen_indices = np.nonzero(h_vals == h_vals.max())[0]
                h_chosen = set(step.available_nodes[i] for i in h_chosen_indices)

                # Planning agent: shortest graph path
                magent.loc = None
                magent.loc = p
                m_vals = magent.evaluate(step.available_nodes, g, penalties=False)
                m_chosen_indices = np.nonzero(m_vals == m_vals.max())[0]
                m_chosen = set(step.available_nodes[i] for i in m_chosen_indices)

                if len(h_chosen.intersection(m_chosen)) == 0:
                    n_discordant_points += 1

    if as_fraction:
        total_trials = len(maze.nodes()) * (len(maze.nodes()) - 1)
        n_discordant_points /= total_trials

    return n_discordant_points


def strict_topology_optimization(
    maze, distance="euclidian", min_avail_nodes=3, as_fraction=True
):
    if distance == "euclidian":
        hagent = mx.agents.HeuristicAgent(beta=float("inf"), xi=0)
    elif distance == "manhattan":
        hagent = mx.agents.ManhattanAgent(beta=float("inf"), xi=0)

    magent = mx.agents.OmniscientAgent(maze)
    n_discordant_points = 0
    goals = maze.nodes()
    for g in goals:
        trial = mx.Trial(maze, g)
        eval_points = list(maze.nodes())
        eval_points.remove(g)
        for p in eval_points:
            step = mx.Step(p, trial)
            if len(step.available_nodes) >= min_avail_nodes:
                hagent.loc = None
                hagent.loc = p
                h_vals = hagent.evaluate(step.available_nodes, g, penalties=False)
                h_chosen_indices = np.nonzero(h_vals == h_vals.max())[0]
                h_chosen = []
                for i in h_chosen_indices:
                    h_chosen.append(step.available_nodes[i])

                magent.loc = None
                magent.loc = p
                m_vals = magent.evaluate(step.available_nodes, g, penalties=False)
                m_chosen_indices = np.nonzero(m_vals == m_vals.max())[0]
                m_chosen = []
                for i in m_chosen_indices:
                    m_chosen.append(step.available_nodes[i])

                set_h_chosen = set(h_chosen)
                set_m_chosen = set(m_chosen)
                if len(set_h_chosen.intersection(set_m_chosen)) == 0:
                    n_discordant_points += 1

    if as_fraction:
        total_trials = len(maze.nodes()) * (len(maze.nodes()) - 1)
        n_discordant_points /= total_trials

    return n_discordant_points
