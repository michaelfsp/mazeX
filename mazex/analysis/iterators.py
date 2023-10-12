import numpy as np
import mazex as mx


def exclude_goal(steps):
    for s in steps:
        if s.loc == s.trial.goal:
            pass
        else:
            yield s


def first_exit(session):
    """Return iterator over the steps of session such that only the first time
    they are visited is picked and subsequent visits are ignored.
    """
    i_trial = 0
    i_last_trial = len(session.trials) - 1
    visited_nodes = []

    while i_trial <= i_last_trial:
        i_step = 0
        i_last_step = len(session.trials[i_trial].steps) - 1

        while i_step < i_last_step:
            step = session.trials[i_trial].steps[i_step]
            if step.loc not in visited_nodes:
                visited_nodes.append(step.loc)
                yield step
            i_step += 1

        # we get here in the last step of every trial
        i_trial += 1
        visited_nodes.clear()


def first_exit_over_steps(steps):
    """Return iterator over the steps of session such that only the first time
    they are visited is picked and subsequent visits are ignored.

    This iterator is the same as first_exit except that it takes as input an
    iterable of steps, whereas first_exit expects a session as input.
    """
    for s in steps:
        if s.path.count(s.loc) == 1 and s.loc != s.trial.goal:
            yield s
        else:
            pass


def first_edge_crossing(session):
    """Return iterator over the steps of session such that only the first time
    an edge is crossed is the origin step returned and subsequent edge
    crossings are ignored.
    """
    i_trial = 0
    i_last_trial = len(session.trials) - 1
    crossed_edges = []

    while i_trial <= i_last_trial:
        i_step = 0
        i_last_step = len(session.trials[i_trial].steps) - 1

        while i_step < i_last_step:
            step = session.trials[i_trial].steps[i_step]
            edge = (step.loc, step.subsequent.loc)
            if edge not in crossed_edges and (edge[1], edge[0]) not in crossed_edges:
                crossed_edges.append(edge)
                yield step
            i_step += 1

        # we get here in the last step of every trial
        i_trial += 1
        crossed_edges.clear()


def orthogonal_planning_vectornav(
    steps,
    maze,
    distance="euclidian",
    only_forward=True,
    discard_backtracked=True,
    either_plan_or_nav=True,
):
    if distance == "euclidian":
        hagent = mx.agents.HeuristicAgent(beta=float("inf"), xi=0)
    elif distance == "manhattan":
        hagent = mx.agents.ManhattanAgent(beta=float("inf"), xi=0)
    magent = mx.agents.OmniscientAgent(maze, xi=0)

    for step in steps:
        if step.previous:
            if discard_backtracked and step.previous.loc == step.subsequent.loc:
                # by continuing to the next iteration of the for loop we skip
                # over steps where the agent went to where it just came from
                continue
            else:
                if only_forward:
                    available_nodes = [
                        a for a in step.available_nodes if a != step.previous.loc
                    ]
                else:
                    available_nodes = [a for a in step.available_nodes]
        else:
            available_nodes = [a for a in step.available_nodes]

        hagent.loc = None
        hagent.loc = step.loc
        h_vals = hagent.evaluate(available_nodes, step.trial.goal, penalties=False)
        h_chosen_indices = np.nonzero(h_vals == h_vals.max())[0]
        h_chosen = []
        for i in h_chosen_indices:
            h_chosen.append(available_nodes[i])

        magent.loc = None
        magent.loc = step.loc
        m_vals = magent.evaluate(available_nodes, step.trial.goal, penalties=False)
        m_chosen_indices = np.nonzero(m_vals == m_vals.max())[0]
        m_chosen = []
        for i in m_chosen_indices:
            m_chosen.append(available_nodes[i])

        set_h_chosen = set(h_chosen)
        set_m_chosen = set(m_chosen)
        if len(set_h_chosen.intersection(set_m_chosen)) == 0:
            # import ipdb; ipdb.set_trace()
            if either_plan_or_nav:
                if (
                    step.subsequent.loc in set_h_chosen
                    or step.subsequent.loc in set_m_chosen
                ):
                    yield step
            else:
                yield step
        else:
            pass
