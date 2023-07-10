
class ERLAgent(Agent):
    """This Agent is legacy. Use OneStepTD instead!
    """
    def __init__(self, maze, gamma, n_over_m, alpha, initval='euclidean', beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None):

        super(ERLAgent, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading)

        self.maze = maze
        self.gamma = gamma
        self.n_over_m = n_over_m
        self.alpha = alpha
        # Reminder: order in Q is (loc, step, goal)
        self.Q = init_Q(maze, initval, n_over_m)
        self.goal = None

    def _set_loc_ERL(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            old_loc = self.loc
            new_loc = node
            if old_loc != new_loc:
                self.update_Q(old_loc, new_loc, self.goal)
        _set_loc(self, node)

    loc = property(_get_loc, _set_loc_ERL)

    def get_Q_a(self, s, s_prime, s_T):
        """Helper function to retrieve Q(s,s',s_T) by looking up and using correct indices.
        """
        s_i = list(self.maze.nodes()).index(s)
        s_prime_i = list(self.maze.nodes()).index(s_prime)
        s_T_i = list(self.maze.nodes()).index(s_T)

        return self.Q[s_i, s_prime_i, s_T_i]

    def set_Q_a(self, s, s_prime, s_T, new_Q_a):
        s_i = list(self.maze.nodes()).index(s)
        s_prime_i = list(self.maze.nodes()).index(s_prime)
        s_T_i = list(self.maze.nodes()).index(s_T)
        self.Q[s_i, s_prime_i, s_T_i] = new_Q_a

    def get_all_Q_a(self, s, s_T):
        s_i = list(self.maze.nodes()).index(s)
        s_T_i = list(self.maze.nodes()).index(s_T)
        all_Q_s_s_T = self.Q[s_i, :, s_T_i]
        nan_i = np.isnan(all_Q_s_s_T)
        Q_as = all_Q_s_s_T[~nan_i]

        return Q_as

    def update_Q(self, s, s_prime, s_T):
        Q_s_a = self.get_Q_a(s, s_prime, s_T)
        R = euclidean_reward(s, s_prime, s_T, self.n_over_m)
        if s_prime == s_T:
            # by definition the value of a terminal state is zero, i.e.,
            # v(s_T) = 0
            max_Q_s_prime = 0
        else:
            max_Q_s_prime = np.max(self.get_all_Q_a(s_prime, s_T))
        RPE = self.alpha * (R + self.gamma * max_Q_s_prime - Q_s_a)
        new_Q_s_a = Q_s_a + RPE
        self.set_Q_a(s, s_prime, s_T, new_Q_s_a)

    def evaluate(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        # Detect if we are in the first step of an episode
        if self.goal != goal:
            # conduct a V(S) local update
            s_neighbors = list(self.maze.neighbors(self.loc))
            V_s_neighbors = np.array([self.get_V(s_prime,goal) for s_prime in s_neighbors])
            choice_probs = softmax(self.beta, V_s_neighbors)
            V_s = np.sum(choice_probs * V_s_neighbors)
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            values[i] = self.get_Q_a(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_Q=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_Q() via _set_loc_ERL(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores Q after session has been evaluated.
        """
        if restore_Q:
            old_Q = copy.deepcopy(self.Q)

        res = super(ERLAgent, self).evaluate_session(session, penalties=penalties)

        if restore_Q:
            self.Q = old_Q

        return res


class OneStepTD(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(OneStepTD, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)
        V_s_prime = self.get_V(s_prime, s_T)
        
        R = self.reward_function(s, s_prime, s_T)

        RPE = R + self.gamma * V_s_prime - V_s_old
        V_s_new = V_s_old + self.alpha * RPE
        
        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class VVlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(VVlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)

        # This is a bit of a hack
        self.real_beta = beta
        self.beta = np.inf
        self.S_prime = None
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)

            if node == self.goal:
                if self.S_prime:
                    self.S_prime = None
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)
        if s_prime == s_T:
            V_s_prime = self.get_V(s_prime, s_T)
        else:
            nodeslist = list(self.maze.neighbors(s_prime))
            values = np.zeros(len(nodeslist))

            for i, c in enumerate(nodeslist):
                R = self.reward_function(s_prime, c, s_T)
                values[i] = (R + self.gamma * self.get_V(c, s_T))
            S_prime_prime_i = self.choose_edge_index(values)
            S_prime_prime = nodeslist[S_prime_prime_i]
            self.S_prime = S_prime_prime
            V_s_prime = self.get_V(S_prime_prime, s_T)

        
        R = self.reward_function(s, s_prime, s_T)

        RPE = R + self.gamma * V_s_prime - V_s_old
        V_s_new = V_s_old + self.alpha * RPE
        
        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        self.goal = goal
        # In the first step of a trial
        if not self.S_prime:
            # TODO: make it such that the goal is not necessarily set by this
            # function so that self._set_loc can then update the self.Q
            # It is a bit hacky...
            values = np.zeros(len(nodeslist))

            for i, c in enumerate(nodeslist):
                R = self.reward_function(self.loc, c, goal)
                values[i] = (R + self.gamma * self.get_V(c, goal))

        # In the other steps
        else:
            # because self.beta is np.inf we only need to make one of values
            # higher so it will be chosen for sure
            S_prime_i = nodeslist.index(self.S_prime)
            values = np.zeros(len(nodeslist))
            values[S_prime_i] = 1

        # TODO: figure out what to do about the penalties here
        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

#    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
#        """Gets the next node in the node cache"""
#        # TODO: make it such that the goal is not necessarily set by this
#        # function so that self._set_loc can then update the self.Q
#        # It is a bit hacky...
#        self.goal = goal
#        values = np.zeros(len(nodeslist))
#
#        # save some computation time for agents that don't do learning
#        for i, c in enumerate(nodeslist):
#            values[i] = self.reward_function(self.loc, c, goal)
#
#        if penalties:
#            penalties = self.penalty_factor(nodeslist)
#
#            values -= penalties
#
#        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class Vmaxlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(Vmaxlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        nodeslist = list(self.maze.neighbors(s))
        Q_values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(s, c, s_T)
            Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        
        VRPE = Q_values.max() - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class Vmaxprelearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(Vmaxprelearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        #s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        #V_s_old = self.get_V(s, s_T)

        #nodeslist = list(self.maze.neighbors(s))
        #Q_values = np.zeros(len(nodeslist))

        #for i, c in enumerate(nodeslist):
        #    R = self.reward_function(s, c, s_T)
        #    Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        #
        #VRPE = Q_values.max() - V_s_old

        #V_s_new = V_s_old + self.alpha * VRPE

        #self.V[s_i, s_T_i] = V_s_new
        pass

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        s_i, s_T_i = self._get_node_index([self.loc, goal])
        V_s_old = self.get_V(self.loc, goal)

        values = np.zeros(len(nodeslist))
        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        VRPE = values.max() - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        s_i, s_T_i = self._get_node_index([self.loc, goal])
        V_s_old = self.get_V(self.loc, goal)

        values = np.zeros(len(nodeslist))
        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        VRPE = values.max() - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class MaxVmaxlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(MaxVmaxlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        nodeslist = list(self.maze.neighbors(s))
        Q_values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(s, c, s_T)
            if c == s_prime:
                if s_prime != s_T:
                    s_prime_neighbors = list(self.maze.neighbors(s_prime))
                    Q_s_prime_neighbors = np.zeros(len(s_prime_neighbors))
                    for j, s_prime_neighbor in enumerate(s_prime_neighbors):
                        R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                        Q_s_prime_neighbors[j] = R_s_prime_neighbor + self.gamma * self.get_V(s_prime_neighbor, s_T)
                    V_s_prime = Q_s_prime_neighbors.max()
                else:
                    V_s_prime = self.get_V(s_prime, s_T)
                Q_values[i] = (R + self.gamma * V_s_prime)

            else:
                Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        
        VRPE = Q_values.max() - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class MaxVlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(MaxVlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)
        R = self.reward_function(s, s_prime, s_T)       

        if s_prime != s_T:
            s_prime_neighbors = list(self.maze.neighbors(s_prime)) 
            Q_s_prime = np.zeros(len(s_prime_neighbors))
            for i, s_prime_prime in enumerate(s_prime_neighbors):
                Q_s_prime[i] = self.reward_function(s_prime, s_prime_prime, s_T) + self.gamma * self.get_V(s_prime_prime, s_T)

            V_s_prime = Q_s_prime.max()
        else:
            V_s_prime = self.get_V(s_prime, s_T)
        
        RPE = R + (self.gamma * V_s_prime) - V_s_old
        V_s_new = V_s_old + self.alpha * RPE
        
        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class Vmaxsprimelearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(Vmaxsprimelearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        _, s_prime_i, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_prime_old = self.get_V(s_prime, s_T)

        s_prime_nodeslist = list(self.maze.neighbors(s_prime))
        Q_values_s_prime = np.zeros(len(s_prime_nodeslist))

        for i, c in enumerate(s_prime_nodeslist):
            R = self.reward_function(s_prime, c, s_T)
            Q_values_s_prime[i] = (R + self.gamma * self.get_V(c, s_T))
        
        VRPE = Q_values_s_prime.max() - V_s_prime_old

        V_s_prime_new = V_s_prime_old + self.alpha * VRPE

        self.V[s_prime_i, s_T_i] = V_s_prime_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class VmaxVlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(VmaxVlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, s_prime_i, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        nodeslist_s_prime = list(self.maze.neighbors(s_prime))
        Q_values_s_prime = np.zeros(len(nodeslist_s_prime))

        for i, s_prime_prime in enumerate(nodeslist_s_prime):
            r_s_prime = self.reward_function(s_prime, s_prime_prime, s_T)
            Q_values_s_prime[i] = (r_s_prime + self.gamma * self.get_V(s_prime_prime, s_T))

        V_s_prime = Q_values_s_prime.max()
        
        R = self.reward_function(s, s_prime, s_T)
        VRPE = R + self.gamma * V_s_prime - V_s_old 
        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class ExpVlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(ExpVlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)
        R = self.reward_function(s, s_prime, s_T)
        
        if s_prime != s_T:
            nodeslist_s_prime = list(self.maze.neighbors(s_prime))
            Q_values_s_prime = np.zeros(len(nodeslist_s_prime))

            for i, s_prime_prime in enumerate(nodeslist_s_prime):
                r_s_prime_prime = self.reward_function(s_prime, s_prime_prime, s_T)
                Q_values_s_prime[i] = (r_s_prime_prime + self.gamma * self.get_V(s_prime_prime, s_T))

            V_s_prime = np.sum(self.choice_probs(Q_values_s_prime) * Q_values_s_prime)
        else:
            V_s_prime = self.get_V(s_prime, s_T)
        
        VRPE = R + self.gamma * V_s_prime - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class ExpVExpVlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(ExpVExpVlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        nodeslist = list(self.maze.neighbors(s))
        Q_values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(s, c, s_T)
            if c == s_prime:
                if s_prime != s_T:
                    s_prime_neighbors = list(self.maze.neighbors(s_prime))
                    Q_s_prime_neighbors = np.zeros(len(s_prime_neighbors))
                    for j, s_prime_neighbor in enumerate(s_prime_neighbors):
                        R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                        Q_s_prime_neighbors[j] = R_s_prime_neighbor + self.gamma * self.get_V(s_prime_neighbor, s_T)
                    V_s_prime = np.sum(self.choice_probs(Q_s_prime_neighbors) * Q_s_prime_neighbors)
                else:
                    V_s_prime = self.get_V(s_prime, s_T)
                Q_values[i] = (R + self.gamma * V_s_prime)

            else:
                Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        
        V_s_new_target = np.sum(self.choice_probs(Q_values) * Q_values)

        VRPE = V_s_new_target - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class MaxVmaxSymlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(MaxVmaxSymlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        nodeslist = list(self.maze.neighbors(s))
        Q_values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(s, c, s_T)
            if c == s_prime:
                if s_prime != s_T:
                    s_prime_neighbors = list(self.maze.neighbors(s_prime))
                    Q_s_prime_neighbors = np.zeros(len(s_prime_neighbors))
                    for j, s_prime_neighbor in enumerate(s_prime_neighbors):
                        R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                        Q_s_prime_neighbors[j] = R_s_prime_neighbor + self.gamma * self.get_V(s_prime_neighbor, s_T)
                    V_s_prime = Q_s_prime_neighbors.max()
                else:
                    V_s_prime = self.get_V(s_prime, s_T)
                Q_values[i] = (R + self.gamma * V_s_prime)

            else:
                Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        
        VRPE = Q_values.max() - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new
        self.V[s_T_i, s_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class ExpVMaxVlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(ExpVMaxVlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        nodeslist = list(self.maze.neighbors(s))
        Q_values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(s, c, s_T)
            if c == s_prime:
                if s_prime != s_T:
                    s_prime_neighbors = list(self.maze.neighbors(s_prime))
                    Q_s_prime_neighbors = np.zeros(len(s_prime_neighbors))
                    for j, s_prime_neighbor in enumerate(s_prime_neighbors):
                        R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                        Q_s_prime_neighbors[j] = R_s_prime_neighbor + self.gamma * self.get_V(s_prime_neighbor, s_T)
                    #V_s_prime = np.sum(self.choice_probs(Q_s_prime_neighbors) * Q_s_prime_neighbors)
                    V_s_prime = Q_s_prime_neighbors.max()
                else:
                    V_s_prime = self.get_V(s_prime, s_T)
                Q_values[i] = (R + self.gamma * V_s_prime)

            else:
                Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        
        V_s_new_target = np.sum(self.choice_probs(Q_values) * Q_values)

        VRPE = V_s_new_target - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class MaxVExpVlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(MaxVExpVlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        nodeslist = list(self.maze.neighbors(s))
        Q_values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(s, c, s_T)
            if c == s_prime:
                if s_prime != s_T:
                    s_prime_neighbors = list(self.maze.neighbors(s_prime))
                    Q_s_prime_neighbors = np.zeros(len(s_prime_neighbors))
                    for j, s_prime_neighbor in enumerate(s_prime_neighbors):
                        R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                        Q_s_prime_neighbors[j] = R_s_prime_neighbor + self.gamma * self.get_V(s_prime_neighbor, s_T)
                    V_s_prime = np.sum(self.choice_probs(Q_s_prime_neighbors) * Q_s_prime_neighbors)
                else:
                    V_s_prime = self.get_V(s_prime, s_T)
                Q_values[i] = (R + self.gamma * V_s_prime)

            else:
                Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        
        #V_s_new_target = np.sum(self.choice_probs(Q_values) * Q_values)
        V_s_new_target = Q_values.max()


        VRPE = V_s_new_target - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class ExpVExpVlearning21(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(ExpVExpVlearning21, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, s_prime_i, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        nodeslist = list(self.maze.neighbors(s))
        Q_values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(s, c, s_T)
            if c == s_prime:
                if s_prime != s_T:
                    s_prime_neighbors = list(self.maze.neighbors(s_prime))
                    Q_s_prime_neighbors = np.zeros(len(s_prime_neighbors))
                    for j, s_prime_neighbor in enumerate(s_prime_neighbors):
                        R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                        Q_s_prime_neighbors[j] = R_s_prime_neighbor + self.gamma * self.get_V(s_prime_neighbor, s_T)
                    V_s_prime = np.sum(self.choice_probs(Q_s_prime_neighbors) * Q_s_prime_neighbors)
                else:
                    V_s_prime = self.get_V(s_prime, s_T)
                Q_values[i] = (R + self.gamma * V_s_prime)

            else:
                Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        
        V_s_new_target = np.sum(self.choice_probs(Q_values) * Q_values)

        VRPE = V_s_new_target - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new
        self.V[s_prime_i, s_T_i] = V_s_prime

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class VEVlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(VEVlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        nodeslist = list(self.maze.neighbors(s))
        Q_values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(s, c, s_T)
            Q_values[i] = (R + self.gamma * self.get_V(c, s_T))

        V_s_update_1_target = np.sum(self.choice_probs(Q_values) * Q_values)
        VRPE1 = V_s_update_1_target - V_s_old
        V_s_new_1 = V_s_old + self.alpha * VRPE1

        self.V[s_i, s_T_i] = V_s_new_1
        
        # Now round 2
        V_s_old = self.get_V(s, s_T)
        Q_values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(s, c, s_T)
            if c == s_prime:
                if s_prime != s_T:
                    s_prime_neighbors = list(self.maze.neighbors(s_prime))
                    Q_s_prime_neighbors = np.zeros(len(s_prime_neighbors))
                    for j, s_prime_neighbor in enumerate(s_prime_neighbors):
                        R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                        Q_s_prime_neighbors[j] = R_s_prime_neighbor + self.gamma * self.get_V(s_prime_neighbor, s_T)
                    V_s_prime = np.sum(self.choice_probs(Q_s_prime_neighbors) * Q_s_prime_neighbors)
                else:
                    V_s_prime = self.get_V(s_prime, s_T)
                Q_values[i] = (R + self.gamma * V_s_prime)

            else:
                Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        
        V_s_new_target = np.sum(self.choice_probs(Q_values) * Q_values)

        VRPE = V_s_new_target - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class ExpVExpVlearning21alpha(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(ExpVExpVlearning21alpha, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, s_prime_i, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)
        V_s_prime_old = self.get_V(s_prime, s_T)

        nodeslist = list(self.maze.neighbors(s))
        Q_values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(s, c, s_T)
            if c == s_prime:
                if s_prime != s_T:
                    s_prime_neighbors = list(self.maze.neighbors(s_prime))
                    Q_s_prime_neighbors = np.zeros(len(s_prime_neighbors))
                    for j, s_prime_neighbor in enumerate(s_prime_neighbors):
                        R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                        Q_s_prime_neighbors[j] = R_s_prime_neighbor + self.gamma * self.get_V(s_prime_neighbor, s_T)
                    V_s_prime = np.sum(self.choice_probs(Q_s_prime_neighbors) * Q_s_prime_neighbors)
                else:
                    V_s_prime = self.get_V(s_prime, s_T)
                Q_values[i] = (R + self.gamma * V_s_prime)

            else:
                Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        
        V_s_new_target = np.sum(self.choice_probs(Q_values) * Q_values)

        VRPE = V_s_new_target - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        V_s_prime_new = (1 - self.alpha) * V_s_prime_old + self.alpha * V_s_prime

        self.V[s_i, s_T_i] = V_s_new
        self.V[s_prime_i, s_T_i] = V_s_prime_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class VEVlearning1(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(VEVlearning1, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = copy.deepcopy(self.get_V(s, s_T))
        V_s_prime_old = copy.deepcopy(self.get_V(s_prime, s_T))

        s_nodeslist = list(self.maze.neighbors(s))
        Q_values_s = np.zeros(len(s_nodeslist))

        for i, c in enumerate(s_nodeslist):
            R = self.reward_function(s, c, s_T)
            Q_values_s[i] = (R + self.gamma * self.get_V(c, s_T))

        #V_s_updated = (1-self.alpha)*V_s_prime_old + self.alpha * np.sum(self.choice_probs(Q_values_s) * Q_values_s)
        V_s_updated = np.sum(self.choice_probs(Q_values_s) * Q_values_s)

        #self.V[s_i, s_T_i] = V_s_updated
        
        # Now round 2
        #V_s_old = self.get_V(s, s_T)

        s_prime_nodeslist = list(self.maze.neighbors(s_prime))
        Q_values_s_prime = np.zeros(len(s_prime_nodeslist))

        for i, c in enumerate(s_nodeslist):
            R = self.reward_function(s, c, s_T)
            if c == s_prime:
                if s_prime != s_T:
                    # We expand the tree
                    for j, s_prime_neighbor in enumerate(s_prime_nodeslist):
                        R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                        if s_prime_neighbor == s:
                            V_s_prime_neighbor = V_s_updated
                        else:
                            V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)
#                        Q_s_prime_neighbors[j] = R_s_prime_neighbor + self.gamma * self.get_V(s_prime_neighbor, s_T)
                        Q_values_s_prime[j] = R_s_prime_neighbor + self.gamma * V_s_prime_neighbor

                    V_s_prime = np.sum(self.choice_probs(Q_values_s_prime) * Q_values_s_prime)
                else:
                    # s_prime is the terminal state and the episode just ended
                    # We do not expand the tree
                    V_s_prime = self.get_V(s_prime, s_T)
                Q_values_s[i] = (R + self.gamma * V_s_prime)

            else:
                Q_values_s[i] = (R + self.gamma * self.get_V(c, s_T))
        
        V_s_new_target = np.sum(self.choice_probs(Q_values_s) * Q_values_s)

        #self.V[s_i, s_T_i] = V_s_old

        VRPE = V_s_new_target - V_s_old
        #VRPE = V_s_new_target - V_s_updated

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class VEVlearning1NoUpdate(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(VEVlearning1NoUpdate, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = copy.deepcopy(self.get_V(s, s_T))
        V_s_prime_old = copy.deepcopy(self.get_V(s_prime, s_T))

        s_nodeslist = list(self.maze.neighbors(s))
        Q_values_s = np.zeros(len(s_nodeslist))

        for i, c in enumerate(s_nodeslist):
            R = self.reward_function(s, c, s_T)
            Q_values_s[i] = (R + self.gamma * self.get_V(c, s_T))

        #V_s_updated = (1-self.alpha)*V_s_prime_old + self.alpha * np.sum(self.choice_probs(Q_values_s) * Q_values_s)
        V_s_updated = np.sum(self.choice_probs(Q_values_s) * Q_values_s)

        #self.V[s_i, s_T_i] = V_s_updated
        
        # Now round 2
        #V_s_old = self.get_V(s, s_T)

        s_prime_nodeslist = list(self.maze.neighbors(s_prime))
        Q_values_s_prime = np.zeros(len(s_prime_nodeslist))

        for i, c in enumerate(s_nodeslist):
            R = self.reward_function(s, c, s_T)
            if c == s_prime:
                if s_prime != s_T:
                    # We expand the tree
                    for j, s_prime_neighbor in enumerate(s_prime_nodeslist):
                        R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                        if s_prime_neighbor == s:
                            #V_s_prime_neighbor = V_s_updated
                            V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)
                        else:
                            V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)
#                        Q_s_prime_neighbors[j] = R_s_prime_neighbor + self.gamma * self.get_V(s_prime_neighbor, s_T)
                        Q_values_s_prime[j] = R_s_prime_neighbor + self.gamma * V_s_prime_neighbor

                    V_s_prime = np.sum(self.choice_probs(Q_values_s_prime) * Q_values_s_prime)
                else:
                    # s_prime is the terminal state and the episode just ended
                    # We do not expand the tree
                    V_s_prime = self.get_V(s_prime, s_T)
                Q_values_s[i] = (R + self.gamma * V_s_prime)

            else:
                Q_values_s[i] = (R + self.gamma * self.get_V(c, s_T))
        
        V_s_new_target = np.sum(self.choice_probs(Q_values_s) * Q_values_s)

        #self.V[s_i, s_T_i] = V_s_old

        VRPE = V_s_new_target - V_s_old
        #VRPE = V_s_new_target - V_s_updated

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


#class V1stEVlearning(Agent):
#    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
#            penalty_criterium='node', penalty_type='exponential',
#            heading=None, maze_size=6, policy='softmax'):
#        super(V1stEVlearning, self).__init__(loc=loc, beta=beta, xi=xi,
#                penalty_criterium=penalty_criterium,
#                penalty_type=penalty_type, heading=heading, policy=policy)
#        
#        #TODO: remove the hack of passing maze as argument
#        self.maze = maze
#        self.maze_size = maze_size
#        self.initfun = initfun
#        self.alpha = alpha
#        self.gamma = gamma
#        # FIXME: check that I have dtype='float' in other implementations too
#        if callable(initfun):
#            self.V = self._init_V(self.initfun)
#        else: 
#            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')
#
#        # We now need to set the diagonal to zero because there is no reward to be had once
#        # s == s_T
#        np.fill_diagonal(self.V, 0)
#        
#        self.goal = None
#        self.reward_function = reward_function
#
#        if self.gamma:
#            self.evaluate = self.evaluate_w_gamma
#        else:
#            self.evaluate = self.evaluate_no_gamma
#
#    def _init_V(self, initfun):
#        V = np.empty((self.maze_size**2, self.maze_size**2))
#        for i, s in enumerate(self.maze.nodes()):
#            for j, s_T in enumerate(self.maze.nodes()):
#                V[i, j] = initfun(s, s_T)
#        return V
#        
#    def _set_loc_OneStepTD(self, node):
#        # we need to use a special _set_loc to trigger Q updates when the agent moves
#        # if it is not the very first location
#        if self.loc:
#            # it saves a bit of time for agents that don't learn
#            if self.alpha:
#                if node: # this let's us change the location to None without triggering value updates
#                    old_loc = self.loc
#                    new_loc = node
#                    if old_loc != new_loc:
#                        self.update_V(old_loc, new_loc, self.goal)
#        _set_loc(self, node)
#    
#    loc = property(_get_loc, _set_loc_OneStepTD)
#
#    def _get_node_index(self, node):
#        """Helper function to retrieve indices of nodes in a matrix.
#
#        Accepts either a node or a list of nodes
#        """
#        if type(node) is list:
#            return [list(self.maze.nodes()).index(n) for n in node]
#        else:
#            return list(self.maze.nodes()).index(node)
#        
#    def get_V(self, s, s_T):
#        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
#        """
#        s_i, s_T_i = self._get_node_index([s, s_T])
#
#        return self.V[s_i, s_T_i]
#
#    def update_V(self, s, s_prime, s_T):
#        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])
#
#        V_s_old = copy.deepcopy(self.get_V(s, s_T))
#        V_s_prime_old = copy.deepcopy(self.get_V(s_prime, s_T))
#
#        s_nodeslist = list(self.maze.neighbors(s))
#        Q_values_s = np.zeros(len(s_nodeslist))
#
#        for i, c in enumerate(s_nodeslist):
#            R = self.reward_function(s, c, s_T)
#            Q_values_s[i] = (R + self.gamma * self.get_V(c, s_T))
#
#        #V_s_updated = (1-self.alpha)*V_s_prime_old + self.alpha * np.sum(self.choice_probs(Q_values_s) * Q_values_s)
#        V_s_updated = np.sum(self.choice_probs(Q_values_s) * Q_values_s)
#
#        #self.V[s_i, s_T_i] = V_s_updated
#        
#        # Now round 2
#        #V_s_old = self.get_V(s, s_T)
#
#        s_prime_nodeslist = list(self.maze.neighbors(s_prime))
#        Q_values_s_prime = np.zeros(len(s_prime_nodeslist))
#
#        for i, c in enumerate(s_nodeslist):
#            R = self.reward_function(s, c, s_T)
#            if c == s_prime:
#                if s_prime != s_T:
#                    # We expand the tree
#                    for j, s_prime_neighbor in enumerate(s_prime_nodeslist):
#                        R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
#                        if s_prime_neighbor == s:
#                            V_s_prime_neighbor = V_s_updated
#                        else:
#                            V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)
##                        Q_s_prime_neighbors[j] = R_s_prime_neighbor + self.gamma * self.get_V(s_prime_neighbor, s_T)
#                        Q_values_s_prime[j] = R_s_prime_neighbor + self.gamma * V_s_prime_neighbor
#
#                    V_s_prime = np.sum(self.choice_probs(Q_values_s_prime) * Q_values_s_prime)
#                else:
#                    # s_prime is the terminal state and the episode just ended
#                    # We do not expand the tree
#                    V_s_prime = self.get_V(s_prime, s_T)
#                Q_values_s[i] = (R + self.gamma * V_s_prime)
#
#            else:
#                Q_values_s[i] = (R + self.gamma * self.get_V(c, s_T))
#        
#        V_s_new_target = np.sum(self.choice_probs(Q_values_s) * Q_values_s)
#
#        #self.V[s_i, s_T_i] = V_s_old
#
#        VRPE = V_s_new_target - V_s_old
#        #VRPE = V_s_new_target - V_s_updated
#
#        V_s_new = V_s_old + self.alpha * VRPE
#
#        self.V[s_i, s_T_i] = V_s_new
#
#    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
#        """Gets the next node in the node cache"""
#        # TODO: make it such that the goal is not necessarily set by this
#        # function so that self._set_loc can then update the self.Q
#        # It is a bit hacky...
#        
#        self.goal = goal
#        values = np.zeros(len(nodeslist))
#
#        for i, c in enumerate(nodeslist):
#            R = self.reward_function(self.loc, c, goal)
#            values[i] = (R + self.gamma * self.get_V(c, goal))
#
#        if penalties:
#            penalties = self.penalty_factor(nodeslist)
#
#            values -= penalties
#
#        return values
#
#    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
#        """Gets the next node in the node cache"""
#        # TODO: make it such that the goal is not necessarily set by this
#        # function so that self._set_loc can then update the self.Q
#        # It is a bit hacky...
#        self.goal = goal
#        values = np.zeros(len(nodeslist))
#
#        # save some computation time for agents that don't do learning
#        for i, c in enumerate(nodeslist):
#            values[i] = self.reward_function(self.loc, c, goal)
#
#        if penalties:
#            penalties = self.penalty_factor(nodeslist)
#
#            values -= penalties
#
#        return values
#
#    def evaluate_session(self, session, penalties=False, restore_V=True):
#        """Running a session evaluation will trigger location changes which 
#        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
#        that after each evaluation V will be altered and as a consequence
#        repeated runnings of evaluate_session() would yield different results
#        on the same session. For that reason we use this wrapper function that
#        restores V after session has been evaluated.
#        """
#        if restore_V:
#            old_V = copy.deepcopy(self.V)
#
#        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)
#
#        if restore_V:
#            self.V = old_V
#
#        return res


class TBVESARSACompound(Agent):
    """This agent uses a compound return as update target of 1/2 * 1-step + 1/2 * 2-step update
    """
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(TBVESARSACompound, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        s_nodeslist = list(self.maze.neighbors(s))
        Q_values_s = np.zeros(len(s_nodeslist))
        for j, s_neighbor in enumerate(s_nodeslist):
            R_s_neighbor = self.reward_function(s, s_neighbor, s_T)
            V_s_neighbor = self.get_V(s_neighbor, s_T)
            Q_values_s[j] = R_s_neighbor + self.gamma * V_s_neighbor
        V_s_new = np.sum(self.choice_probs(Q_values_s) * Q_values_s)

        VPE = V_s_new - V_s_old

        self.V[s_i, s_T_i] = V_s_old + self.alpha * VPE

        if s_prime != s_T:
            s_prime_nodeslist = list(self.maze.neighbors(s_prime))
            Q_values_s_prime = np.zeros(len(s_prime_nodeslist))
            for j, s_prime_neighbor in enumerate(s_prime_nodeslist):
                R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
#                if s_prime_neighbor == s:
#                    V_s_prime_neighbor = V_s_new
#                else:
#                    V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)

                V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)
                Q_values_s_prime[j] = R_s_prime_neighbor + self.gamma * V_s_prime_neighbor
            V_s_prime = np.sum(self.choice_probs(Q_values_s_prime) * Q_values_s_prime)
        else:
            # s_prime is the terminal state and the episode just ended
            # We do not expand the tree
            V_s_prime = self.get_V(s_prime, s_T)

        V_s_old = self.get_V(s, s_T)

        s_nodeslist = list(self.maze.neighbors(s))
        Q_values_s = np.zeros(len(s_nodeslist))
        for j, s_neighbor in enumerate(s_nodeslist):
            R_s_neighbor = self.reward_function(s, s_neighbor, s_T)
            if s_neighbor == s_prime:
                V_s_neighbor = V_s_prime
            else:
                V_s_neighbor = self.get_V(s_neighbor, s_T)
            Q_values_s[j] = R_s_neighbor + self.gamma * V_s_neighbor
        V_s_newest = np.sum(self.choice_probs(Q_values_s) * Q_values_s)

#        VPE = V_s_new - V_s_old
#
#        self.V[s_i, s_T_i] = V_s_old + self.alpha * VPE
#
        self.V[s_i, s_T_i] += self.alpha * (V_s_newest/2 - V_s_new/2)


    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class TBVESARSATB(Agent):
    """This agent uses a compound return as update target of 1/2 * 1-step + 1/2 * 2-step update
    """
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(TBVESARSATB, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])
        
        # We update our expectation of V(S)
        s_nodeslist = list(self.maze.neighbors(s))
        Q_values_s = np.zeros(len(s_nodeslist))
        for j, s_neighbor in enumerate(s_nodeslist):
            R_s_neighbor = self.reward_function(s, s_neighbor, s_T)
            V_s_neighbor = self.get_V(s_neighbor, s_T)
            Q_values_s[j] = R_s_neighbor + self.gamma * V_s_neighbor
        V_s_new = np.sum(self.choice_probs(Q_values_s) * Q_values_s)

        if s_prime != s_T:
            s_prime_nodeslist = list(self.maze.neighbors(s_prime))
            Q_values_s_prime = np.zeros(len(s_prime_nodeslist))
            for j, s_prime_neighbor in enumerate(s_prime_nodeslist):
                R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                if s_prime_neighbor == s:
                    # For V(S) we use the updated expectation
                    V_s_prime_neighbor = V_s_new
                else:
                    V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)
                V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)
                Q_values_s_prime[j] = R_s_prime_neighbor + self.gamma * V_s_prime_neighbor
            V_s_prime = np.sum(self.choice_probs(Q_values_s_prime) * Q_values_s_prime)
        else:
            # s_prime is the terminal state and the episode just ended
            # We do not expand the tree
            V_s_prime = self.get_V(s_prime, s_T)

        s_nodeslist = list(self.maze.neighbors(s))
        Q_values_s = np.zeros(len(s_nodeslist))
        for j, s_neighbor in enumerate(s_nodeslist):
            R_s_neighbor = self.reward_function(s, s_neighbor, s_T)
            if s_neighbor == s_prime:
                V_s_neighbor = V_s_prime
            else:
                V_s_neighbor = self.get_V(s_neighbor, s_T)
            Q_values_s[j] = R_s_neighbor + self.gamma * V_s_neighbor
        V_s_newest = np.sum(self.choice_probs(Q_values_s) * Q_values_s)

        V_s_old = self.get_V(s, s_T)
        VPE = V_s_newest - V_s_old

        self.V[s_i, s_T_i] = V_s_old + self.alpha * VPE

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class VESARSAPostTB(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(VESARSAPostTB, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, s_prime_i, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = copy.deepcopy(self.get_V(s, s_T))
        V_s_prime_old = copy.deepcopy(self.get_V(s_prime, s_T))

        # 1 We updated the estimate of V(S)
        s_nodeslist = list(self.maze.neighbors(s))
        Q_values_s = np.zeros(len(s_nodeslist))
        for i, s_neighbor in enumerate(s_nodeslist):
            R_s_neighbor = self.reward_function(s, s_neighbor, s_T)
            V_s_neighbor = self.get_V(s_neighbor, s_T)
            Q_values_s[i] = R_s_neighbor + self.gamma * V_s_neighbor
        V_s_updated = np.sum(self.choice_probs(Q_values_s) * Q_values_s)

        # 2. We compute the updated V(S') using updated V(S) and information
        # about which S'' are available
        if s_prime != s_T:
            s_prime_nodeslist = list(self.maze.neighbors(s_prime))
            Q_values_s_prime = np.zeros(len(s_prime_nodeslist))
            for j, s_prime_neighbor in enumerate(s_prime_nodeslist):
                R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                if s_prime_neighbor == s:
                    V_s_prime_neighbor = V_s_updated
                else:
                    V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)
                Q_values_s_prime[j] = R_s_prime_neighbor + self.gamma * V_s_prime_neighbor
            V_s_prime = np.sum(self.choice_probs(Q_values_s_prime) * Q_values_s_prime)
        else:
            # s_prime is the terminal state and the episode just ended
            # We do not expand the tree
            V_s_prime = self.get_V(s_prime, s_T)

        VRPE = V_s_prime - V_s_prime_old

        # 3. We do a learning step update of V(S')
#        self.V[s_prime_i, s_T_i] = V_s_prime_old + self.alpha * VRPE

        # 4. We do a tree backup learning step for V(S) using the new V(S')
        s_nodeslist = list(self.maze.neighbors(s))
        Q_values_s = np.zeros(len(s_nodeslist))
        for i, s_neighbor in enumerate(s_nodeslist):
            R_s_neighbor = self.reward_function(s, s_neighbor, s_T)
            if s_neighbor == s_prime:
                V_s_neighbor = V_s_prime
                #V_s_neighbor = self.get_V(s_neighbor, s_T)
            else:
                V_s_neighbor = self.get_V(s_neighbor, s_T)
            Q_values_s[i] = R_s_neighbor + self.gamma * V_s_neighbor
        V_s_new = np.sum(self.choice_probs(Q_values_s) * Q_values_s)

        VPE = V_s_new - V_s_old

        self.V[s_i,s_T] = V_s_old + self.alpha * VPE

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class VESARSA(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(VESARSA, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        _, s_prime_i, s_T_i = self._get_node_index([s, s_prime, s_T])

        if s_prime != s_T:
            s_prime_nodeslist = list(self.maze.neighbors(s_prime))
            Q_values_s_prime = np.zeros(len(s_prime_nodeslist))
            for j, s_prime_neighbor in enumerate(s_prime_nodeslist):
                R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)
                Q_values_s_prime[j] = R_s_prime_neighbor + self.gamma * V_s_prime_neighbor
            V_s_prime = np.sum(self.choice_probs(Q_values_s_prime) * Q_values_s_prime)
        else:
            # s_prime is the terminal state and the episode just ended
            # We do not expand the tree
            V_s_prime = self.get_V(s_prime, s_T)

        V_s_prime_old = self.get_V(s_prime, s_T)
        VRPE = V_s_prime - V_s_prime_old

        V_s_prime_new = V_s_prime_old + self.alpha * VRPE

        self.V[s_prime_i, s_T_i] = V_s_prime_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class EV(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(EV, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        s_nodeslist = list(self.maze.neighbors(s))
        Q_values_s = np.zeros(len(s_nodeslist))
        for j, s_neighbor in enumerate(s_nodeslist):
            R_s_neighbor = self.reward_function(s, s_neighbor, s_T)
            V_s_neighbor = self.get_V(s_neighbor, s_T)
            Q_values_s[j] = R_s_neighbor + self.gamma * V_s_neighbor
        V_s_new = np.sum(self.choice_probs(Q_values_s) * Q_values_s)

        VPE = V_s_new - V_s_old

        self.V[s_i, s_T_i] =  V_s_old + self.alpha * VPE

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class V(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(V, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        R_s_prime = self.reward_function(s, s_prime, s_T)
        V_s_prime = self.get_V(s_prime, s_T)
        V_s_new = R_s_prime + self.gamma * V_s_prime

        VPE = V_s_new - V_s_old

        self.V[s_i, s_T_i] =  V_s_old + self.alpha * VPE

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res

class TwoStepVESARSA(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(TwoStepVESARSA, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        if s_prime != s_T:
            s_prime_nodeslist = list(self.maze.neighbors(s_prime))
            Q_values_s_prime = np.zeros(len(s_prime_nodeslist))
            for j, s_prime_neighbor in enumerate(s_prime_nodeslist):
                R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)
                Q_values_s_prime[j] = R_s_prime_neighbor + self.gamma * V_s_prime_neighbor
            V_s_prime = np.sum(self.choice_probs(Q_values_s_prime) * Q_values_s_prime)
        else:
            # s_prime is the terminal state and the episode just ended
            # We do not expand the tree
            V_s_prime = self.get_V(s_prime, s_T)

        V_s_old = self.get_V(s, s_T)
        VPE = (self.reward_function(s, s_prime, s_T) + self.gamma * V_s_prime) - V_s_old

        self.V[s_i, s_T_i] = V_s_old + self.alpha * VPE

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class TBVESARSA(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(TBVESARSA, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        if s_prime != s_T:
            s_prime_nodeslist = list(self.maze.neighbors(s_prime))
            Q_values_s_prime = np.zeros(len(s_prime_nodeslist))
            for j, s_prime_neighbor in enumerate(s_prime_nodeslist):
                R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)
                Q_values_s_prime[j] = R_s_prime_neighbor + self.gamma * V_s_prime_neighbor
            V_s_prime = np.sum(self.choice_probs(Q_values_s_prime) * Q_values_s_prime)
        else:
            # s_prime is the terminal state and the episode just ended
            # We do not expand the tree
            V_s_prime = self.get_V(s_prime, s_T)

        V_s_old = self.get_V(s, s_T)

        s_nodeslist = list(self.maze.neighbors(s))
        Q_values_s = np.zeros(len(s_nodeslist))
        for j, s_neighbor in enumerate(s_nodeslist):
            R_s_neighbor = self.reward_function(s, s_neighbor, s_T)
            if s_neighbor == s_prime:
                V_s_neighbor = V_s_prime
            else:
                V_s_neighbor = self.get_V(s_neighbor, s_T)
            Q_values_s[j] = R_s_neighbor + self.gamma * V_s_neighbor
        V_s_new = np.sum(self.choice_probs(Q_values_s) * Q_values_s)

        VPE = V_s_new - V_s_old

        self.V[s_i, s_T_i] = V_s_old + self.alpha * VPE

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res




class VQ(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(VQ, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        _, s_prime_i, s_T_i = self._get_node_index([s, s_prime, s_T])

        if s_prime != s_T:
            s_prime_nodeslist = list(self.maze.neighbors(s_prime))
            Q_values_s_prime = np.zeros(len(s_prime_nodeslist))
            for j, s_prime_neighbor in enumerate(s_prime_nodeslist):
                R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)
                Q_values_s_prime[j] = R_s_prime_neighbor + self.gamma * V_s_prime_neighbor
            V_s_prime_max = np.max(Q_values_s_prime)
        else:
            # s_prime is the terminal state and the episode just ended
            # We do not expand the tree
            V_s_prime_max = self.get_V(s_prime, s_T)

        V_s_prime_old = self.get_V(s_prime, s_T)
        VRPE = V_s_prime_max - V_s_prime_old

        V_s_prime_new = V_s_prime_old + self.alpha * VRPE

        self.V[s_prime_i, s_T_i] = V_s_prime_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class MAXV(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(MAXV, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        s_nodeslist = list(self.maze.neighbors(s))
        Q_values_s = np.zeros(len(s_nodeslist))
        for j, s_neighbor in enumerate(s_nodeslist):
            R_s_neighbor = self.reward_function(s, s_neighbor, s_T)
            V_s_neighbor = self.get_V(s_neighbor, s_T)
            Q_values_s[j] = R_s_neighbor + self.gamma * V_s_neighbor
        V_s_new_max = np.max(Q_values_s)

        VPE = V_s_new_max - V_s_old

        self.V[s_i, s_T_i] =  V_s_old + self.alpha * VPE

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class MAXVQ(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(MAXVQ, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        if s_prime != s_T:
            s_prime_nodeslist = list(self.maze.neighbors(s_prime))
            Q_values_s_prime = np.zeros(len(s_prime_nodeslist))
            for j, s_prime_neighbor in enumerate(s_prime_nodeslist):
                R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                V_s_prime_neighbor = self.get_V(s_prime_neighbor, s_T)
                Q_values_s_prime[j] = R_s_prime_neighbor + self.gamma * V_s_prime_neighbor
            V_s_prime = np.max(Q_values_s_prime)
        else:
            # s_prime is the terminal state and the episode just ended
            # We do not expand the tree
            V_s_prime = self.get_V(s_prime, s_T)

        V_s_old = self.get_V(s, s_T)

        s_nodeslist = list(self.maze.neighbors(s))
        Q_values_s = np.zeros(len(s_nodeslist))
        for j, s_neighbor in enumerate(s_nodeslist):
            R_s_neighbor = self.reward_function(s, s_neighbor, s_T)
            if s_neighbor == s_prime:
                V_s_neighbor = V_s_prime
            else:
                V_s_neighbor = self.get_V(s_neighbor, s_T)
            Q_values_s[j] = R_s_neighbor + self.gamma * V_s_neighbor
        V_s_new = np.max(Q_values_s)

        VPE = V_s_new - V_s_old

        self.V[s_i, s_T_i] = V_s_old + self.alpha * VPE

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class VSR(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(VSR, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        R_s = self.reward_function(s, s_T)
        V_s_prime = self.get_V(s_prime, s_T)
        V_s_new = R_s + self.gamma * V_s_prime

        VPE = V_s_new - V_s_old

        self.V[s_i, s_T_i] =  V_s_old + self.alpha * VPE

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            values[i] = self.gamma * self.get_V(c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res


class MaxVmaxSymlearning(Agent):
    def __init__(self, maze, alpha, gamma, reward_function, initfun, beta=1, loc=None, xi=0,
            penalty_criterium='node', penalty_type='exponential',
            heading=None, maze_size=6, policy='softmax'):
        super(MaxVmaxSymlearning, self).__init__(loc=loc, beta=beta, xi=xi,
                penalty_criterium=penalty_criterium,
                penalty_type=penalty_type, heading=heading, policy=policy)
        
        #TODO: remove the hack of passing maze as argument
        self.maze = maze
        self.maze_size = maze_size
        self.initfun = initfun
        self.alpha = alpha
        self.gamma = gamma
        # FIXME: check that I have dtype='float' in other implementations too
        if callable(initfun):
            self.V = self._init_V(self.initfun)
        else: 
            self.V = np.full((self.maze_size**2, self.maze_size**2), self.initfun, dtype='float')

        # We now need to set the diagonal to zero because there is no reward to be had once
        # s == s_T
        np.fill_diagonal(self.V, 0)
        
        self.goal = None
        self.reward_function = reward_function

        if self.gamma:
            self.evaluate = self.evaluate_w_gamma
        else:
            self.evaluate = self.evaluate_no_gamma

    def _init_V(self, initfun):
        V = np.empty((self.maze_size**2, self.maze_size**2))
        for i, s in enumerate(self.maze.nodes()):
            for j, s_T in enumerate(self.maze.nodes()):
                V[i, j] = initfun(s, s_T)
        return V
        
    def _set_loc_OneStepTD(self, node):
        # we need to use a special _set_loc to trigger Q updates when the agent moves
        # if it is not the very first location
        if self.loc:
            # it saves a bit of time for agents that don't learn
            if self.alpha:
                if node: # this let's us change the location to None without triggering value updates
                    old_loc = self.loc
                    new_loc = node
                    if old_loc != new_loc:
                        self.update_V(old_loc, new_loc, self.goal)
        _set_loc(self, node)
    
    loc = property(_get_loc, _set_loc_OneStepTD)

    def _get_node_index(self, node):
        """Helper function to retrieve indices of nodes in a matrix.

        Accepts either a node or a list of nodes
        """
        if type(node) is list:
            return [list(self.maze.nodes()).index(n) for n in node]
        else:
            return list(self.maze.nodes()).index(node)
        
    def get_V(self, s, s_T):
        """Helper function to retrieve V(s,s_T) by looking up and using correct indices.
        """
        s_i, s_T_i = self._get_node_index([s, s_T])

        return self.V[s_i, s_T_i]

    def update_V(self, s, s_prime, s_T):
        s_i, _, s_T_i = self._get_node_index([s, s_prime, s_T])

        V_s_old = self.get_V(s, s_T)

        nodeslist = list(self.maze.neighbors(s))
        Q_values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(s, c, s_T)
            if c == s_prime:
                if s_prime != s_T:
                    s_prime_neighbors = list(self.maze.neighbors(s_prime))
                    Q_s_prime_neighbors = np.zeros(len(s_prime_neighbors))
                    for j, s_prime_neighbor in enumerate(s_prime_neighbors):
                        R_s_prime_neighbor = self.reward_function(s_prime, s_prime_neighbor, s_T)
                        Q_s_prime_neighbors[j] = R_s_prime_neighbor + self.gamma * self.get_V(s_prime_neighbor, s_T)
                    V_s_prime = Q_s_prime_neighbors.max()
                else:
                    V_s_prime = self.get_V(s_prime, s_T)
                Q_values[i] = (R + self.gamma * V_s_prime)

            else:
                Q_values[i] = (R + self.gamma * self.get_V(c, s_T))
        
        VRPE = Q_values.max() - V_s_old

        V_s_new = V_s_old + self.alpha * VRPE

        self.V[s_i, s_T_i] = V_s_new
        self.V[s_T_i, s_i] = V_s_new

    def evaluate_w_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        for i, c in enumerate(nodeslist):
            R = self.reward_function(self.loc, c, goal)
            values[i] = (R + self.gamma * self.get_V(c, goal))

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_no_gamma(self, nodeslist, goal, penalties=False):
        """Gets the next node in the node cache"""
        # TODO: make it such that the goal is not necessarily set by this
        # function so that self._set_loc can then update the self.Q
        # It is a bit hacky...
        self.goal = goal
        values = np.zeros(len(nodeslist))

        # save some computation time for agents that don't do learning
        for i, c in enumerate(nodeslist):
            values[i] = self.reward_function(self.loc, c, goal)

        if penalties:
            penalties = self.penalty_factor(nodeslist)

            values -= penalties

        return values

    def evaluate_session(self, session, penalties=False, restore_V=True):
        """Running a session evaluation will trigger location changes which 
        in turn will trigger update_V() via _set_loc_OneStepTD(). This means
        that after each evaluation V will be altered and as a consequence
        repeated runnings of evaluate_session() would yield different results
        on the same session. For that reason we use this wrapper function that
        restores V after session has been evaluated.
        """
        if restore_V:
            old_V = copy.deepcopy(self.V)

        res = super(OneStepTD, self).evaluate_session(session, penalties=penalties)

        if restore_V:
            self.V = old_V

        return res
