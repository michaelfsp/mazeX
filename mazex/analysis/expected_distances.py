import copy

import mazex as mx
import numpy as np
import scipy

#-----------------------Random Walk-------------------------------------------
def T_random_walk(maze):
    T = np.zeros(2*[len(maze.nodes())])

    for i, s in enumerate(maze.nodes()):
        s_degree = maze.degree(s)
        neighbors = list(maze.neighbors(s))
        neighbors_idx = [list(maze.nodes()).index(n) for n in neighbors]
        T[i, neighbors_idx] = 1/s_degree
    return T

    
def T_random_walk_absorbing_s_T(maze, s_T):
    T = T_random_walk(maze)

    i_s_T = list(maze.nodes()).index(s_T)
    T[i_s_T, :] = 0
    T[i_s_T, i_s_T] = 1

    return T

def Q_random_walk_absorbing_s_T(maze, s_T):
    """Compute the submatrix Q from the block decomposition of matrix T"""
    T = T_random_walk_absorbing_s_T(maze, s_T)

    i_s_T = list(maze.nodes()).index(s_T)
    Q = np.delete(np.delete(T, i_s_T, axis=0), i_s_T, axis=1)

    return Q

def expected_distance_random_walk(maze, s, s_T):

    if s == s_T:
        return 0

    else:
        Q = Q_random_walk_absorbing_s_T(maze, s_T)
        
        # Because the index of s in the Q matrix is not the same is in the T matrix,
        # which is the same as maze.nodes(), we need to find out the index of in
        # Q
        list_nodes_without_s_T = list(maze.nodes())
        list_nodes_without_s_T.remove(s_T)
        # index of s in Q
        i_s_in_Q = list_nodes_without_s_T.index(s)

        # the index in N is the same as in Q
        N = np.linalg.inv(np.identity(len(Q)) - Q)

        return np.sum(N[i_s_in_Q, :])

def random_walk_exp_distances_matrix(maze):
    dists = np.zeros(2*[len(maze.nodes())])

    for i, s in enumerate(maze.nodes()):
        for j, s_T in enumerate(maze.nodes()):
            dists[i, j] = expected_distance_random_walk(maze, s, s_T)

    return dists


#---------------------------Given T---------------------------------
def expected_distance_given_T(maze, T, s_T):
    j = list(maze.nodes()).index(s_T)
    Q = np.delete(np.delete(T, j, axis=0), j, axis=1)

    # the index in N is the same as in Q
    N = np.linalg.inv(np.identity(len(Q)) - Q)

    return np.sum(N, axis=1)

#---------------------------For a given Agent---------------------------------
def T_agent_absorbing_s_T(maze, agent, s_T):
    T = np.zeros(2*[len(maze.nodes())])

    for i, s in enumerate(maze.nodes()):
        if s != s_T:
            agent.loc = None
            agent.loc = s
            neighbors = list(maze.neighbors(s))
            neighbors_idx = [list(maze.nodes()).index(n) for n in neighbors]
            choice_probs = agent.choice_probs(agent.evaluate(nodeslist=neighbors, goal=s_T, penalties=False))
            T[i, neighbors_idx] = choice_probs
        else:
            T[i, :] = 0
            T[i, i] = 1
    return T

def Q_agent_absorbing_s_T(maze, agent, s_T):
    """Compute the submatrix Q from the block decomposition of matrix T"""
    T = T_agent_absorbing_s_T(maze, agent, s_T)

    i_s_T = list(maze.nodes()).index(s_T)
    Q = np.delete(np.delete(T, i_s_T, axis=0), i_s_T, axis=1)

    return Q

def expected_distance_agent(maze, agent, s, s_T):

    if s == s_T:
        return 0

    else:
        Q = Q_agent_absorbing_s_T(maze, agent, s_T)

        # Because the index of s in the Q matrix is not the same is in the T matrix,
        # which is the same as maze.nodes(), we need to find out the index of in
        # Q
        list_nodes_without_s_T = list(maze.nodes())
        list_nodes_without_s_T.remove(s_T)
        # index of s in Q
        i_s_in_Q = list_nodes_without_s_T.index(s)

        # the index in N is the same as in Q
        N = np.linalg.inv(np.identity(len(Q)) - Q)

        return np.sum(N[i_s_in_Q, :])


def agent_exp_distances_matrix(maze, agent):
    dists = np.zeros(2*[len(maze.nodes())])

    for i, s in enumerate(maze.nodes()):
        for j, s_T in enumerate(maze.nodes()):
            dists[i, j] = expected_distance_agent(maze, agent, s, s_T)

    return dists


#------------------------------Legacy-----------------------------------------
def T_softmax_s_T(s_T, beta, v, maze):
    """Computes the transition probabilities matrix T_\pi for s_T as goal using a softmax decision policy.
    """
    T = np.zeros(2*[len(maze.nodes())])
    
    s_T_i = list(maze.nodes()).index(s_T)
    for i, s in enumerate(maze.nodes()):
        if s == s_T:
            # we define s_T to be an absorbing state so transitions to itself occur with probability 1
            T[i,i] = 1
        else:
            neighbors = list(maze.neighbors(s))
            neighbor_idx = [list(maze.nodes()).index(n) for n in neighbors]
            neighbor_vals = v[neighbor_idx, s_T_i]
            s_prime_probs = mx.agents.softmax(beta, neighbor_vals)
            T[i, neighbor_idx] = s_prime_probs
    return T

def T_softmax_s_T_one_step(s_T, beta, v, gamma, reward_function, maze):
    """Computes the transition probabilities matrix T_\pi for s_T as goal using a softmax decision policy.
    """
    T = np.zeros(2*[len(maze.nodes())])
    
    s_T_i = list(maze.nodes()).index(s_T)
    for i, s in enumerate(maze.nodes()):
        if s == s_T:
            # we define s_T to be an absorbing state so transitions to itself occur with probability 1
            T[i,i] = 1
        else:
            neighbors = list(maze.neighbors(s))
            # one step rewards
            neighbor_r = np.array([reward_function(s, s_prime, s_T) for s_prime in neighbors])
            neighbor_idx = [list(maze.nodes()).index(n) for n in neighbors]
            neighbor_vals = v[neighbor_idx, s_T_i]
            transition_vals = neighbor_r + gamma * neighbor_vals
            s_prime_probs = mx.agents.softmax(beta, transition_vals)
            if np.any(np.isnan(s_prime_probs)):
                raise Exception
            else:
                T[i, neighbor_idx] = s_prime_probs
    return T

def expected_T_distance_one_step(beta, v, gamma, reward_function, maze):
    """Computes the expected distance of a policy using v as v_pi and a softmax_decision rule with beta as beta.
    It uses the method of transition matrix block decomposition and matrix inversion.
    
    T = |I | 0 0 0 ...|
        |__|__________|
        |S | Q        |
    
    (I - Q)^-1 gives number of expected visits to all other states until reaching s_T. Summing over those we get the expected path length
        
    see the book by Snell "Finite Markov Chains" (1983) page 59 or for example
    https://www.cantorsparadise.com/magic-tricks-with-markov-chains-96a4465c20a
    """
    dists = np.zeros(2*[len(maze.nodes())])
    
    for j, s_T in enumerate(maze.nodes()):
        T = T_softmax_s_T_one_step(s_T, beta, v, gamma, reward_function, maze)
        Q = np.delete(np.delete(T, j, axis=0), j, axis=1)
        exp_visits = np.linalg.inv(np.identity(len(maze.nodes())-1) - Q)
        # we insert back a column and a row of zeros just so we get the indices right again
        exp_visits_with_s_T = np.insert(np.insert(exp_visits, j, 0, axis=0),j , 0, axis=1)
        for i, s in enumerate(maze.nodes()):
            dists[i, j] = np.sum(exp_visits_with_s_T[i, :])
    return dists

def expected_T_distance(beta, v, maze):
    """Computes the expected distance of a policy using v as v_pi and a softmax_decision rule with beta as beta.
    It uses the method of transition matrix block decomposition and matrix inversion.
    
    T = |I | 0 0 0 ...|
        |__|__________|
        |S | Q        |
    
    (I - Q)^-1 gives number of expected visits to all other states until reaching s_T. Summing over those we get the expected path length
        
    see the book by Snell "Finite Markov Chains" (1983) page 59 or for example
    https://www.cantorsparadise.com/magic-tricks-with-markov-chains-96a4465c20a
    """
    dists = np.zeros(2*[len(maze.nodes())])
    
    for j, s_T in enumerate(maze.nodes()):
        T = T_softmax_s_T(s_T, beta, v, maze)
        Q = np.delete(np.delete(T, j, axis=0), j, axis=1)
        exp_visits = np.linalg.inv(np.identity(len(maze.nodes())-1) - Q)
        # we insert back a column and a row of zeros just so we get the indices right again
        exp_visits_with_s_T = np.insert(np.insert(exp_visits, j, 0, axis=0),j , 0, axis=1)
        for i, s in enumerate(maze.nodes()):
            dists[i, j] = np.sum(exp_visits_with_s_T[i, :])
    return dists

"""This function is deprecated!!"""
def expected_softmax_distance(beta, v, maze, method='iteration'):
    '''We will solve this by finding the fixed point of the equation
    E_distance(s,s_T) = 
     - 0 if s==s_T
     - 1 + \sum_{s_prime} p(s_prime|s) * E_distance(s_prime,s_T), otherwise
    '''
    distances_x0 = np.zeros(2*[len(maze.nodes())])
    
    def loop(dists):
        dists_old = copy.deepcopy(dists)
        dists_new = np.zeros_like(dists_old)
        for i, s in enumerate(maze.nodes()):
            for j, s_T in enumerate(maze.nodes()):
                if s == s_T:
                    # distance between goal and itself should be zero
                    pass
                else:
                    neighbor_idx = [list(maze.nodes()).index(n) for n in maze.neighbors(s)]
                    neighbor_vals = v[neighbor_idx, j]
                    s_prime_probs = mx.agents.softmax(beta, neighbor_vals)
                    s_prime_dists = dists_old[neighbor_idx, j]
                    dists_new[i, j] = 1 + np.sum(s_prime_probs * s_prime_dists)
        return dists_new
    
    final_distances = scipy.optimize.fixed_point(loop, distances_x0, method=method)
    return final_distances

#------------------Different legacy-------------------------------------------
def B_star_v(v, gamma, reward_function, maze):
    """Bellman optimality operator for state values
    Parameters:
    -----------
    v : 2D array with state values. shape should be SxS
    """
    v_old = copy.deepcopy(v)
    v_new = np.zeros_like(v_old)
    
    for i, s in enumerate(maze.nodes()):
        for j, s_T in enumerate(maze.nodes()):
            if s == s_T:
                # we define the value of being at the goal as NaN
                # v_new[i, j] = np.nan
                # we define the value of being at goal as 0
                pass
            else:
                neighbors = list(maze.neighbors(s))
                v_s_s_prime = []
                for s_prime in neighbors:
                    k = list(maze.nodes()).index(s_prime)
                    v_s_s_prime.append(reward_function(s, s_prime, s_T) + gamma * v_old[k,j])
                max_v_s_s_prime = max(v_s_s_prime)
                v_new[i, j] = max_v_s_s_prime
    return v_new

def greedy_policy_distance(v, maze):
    distances = np.zeros(2*[len(maze.nodes())])
    
    for i, s0 in enumerate(maze.nodes()):
        for j, s_T in enumerate(maze.nodes()):
            if s0 == s_T:
                step_count = 0
            else:
                s = s0
                step_count = 0
                while s != s_T:
                    neighbors = list(maze.neighbors(s))
                    neighbor_idx = [list(maze.nodes()).index(n) for n in neighbors]
                    s_prime_i = np.argmax(v[neighbor_idx, j])
                    s = neighbors[s_prime_i]
                    step_count += 1
            distances[i,j] = step_count
    return distances
