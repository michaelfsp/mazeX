from collections import Iterator

def first_exit(session):
    """Return iterator over the steps of session such that only the first time
    they are visited is picked and subsequent visits are ignored.
    """
    i_trial = 0
    i_last_trial = len(session.trials) -1
    visited_nodes = []

    while i_trial <= i_last_trial:
        i_step = 0
        i_last_step = len(session.trials[i_trial].steps) -1
        
        while i_step < i_last_step:
            step = session.trials[i_trial].steps[i_step]
            if step.loc not in visited_nodes:
                visited_nodes.append(step.loc)
                yield step
            i_step += 1
        
        #we get here in the last step of every trial
        i_trial += 1
        visited_nodes.clear()

def first_edge_crossing(session):
    """Return iterator over the steps of session such that only the first time
    an edge is crossed is the origin step returned and subsequent edge
    crossings are ignored.
    """
    i_trial = 0
    i_last_trial = len(session.trials) -1
    crossed_edges = []

    while i_trial <= i_last_trial:
        i_step = 0
        i_last_step = len(session.trials[i_trial].steps) -1
        
        while i_step < i_last_step:
            step = session.trials[i_trial].steps[i_step]
            edge = (step.loc, step.subsequent.loc)
            if edge not in crossed_edges and (
                    edge[1],edge[0]) not in crossed_edges:
                crossed_edges.append(edge)
                yield step
            i_step += 1
        
        #we get here in the last step of every trial
        i_trial += 1
        crossed_edges.clear()

