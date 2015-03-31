import numpy as np
from mazex.agents import HeuristicAgent, MixtureAgent

def protected_log(x):
    'Return log of x protected against giving -inf for very small values of x.'
    return np.log(((1e-200)/2)+(1-(1e-200))*x)

class Likelihood(object):
    """A class for the likelihood functions."""

    def __init__(self, session):
        """Agent should be initialized with 'loc' None.
        """
        self.session = session

    def neg_log_likelihood(self, agent):
        """Compute the likelihood of the data in the session gien the
        parameter set specified in the init of this likelihood method.
        """
        first_loc = self.session.trials[0].steps[0].loc
        agent.loc = first_loc

        log_likelihood = 0
        for t in self.session.trials:
            agent.node_visit_cache.clear()
            agent.edge_visit_cache.clear()

            for s in t.steps[:-1]: # last step is the goal
                agent.loc = s.loc
                values = agent.evaluate(s.available_nodes, t.goal)
                choice_prob = agent.choice_probs(values)[s.choice]

                #TODO: change to protected log
                log_likelihood += protected_log(choice_prob)

        return -log_likelihood

class MixtureLikelihood(Likelihood):

    def __init__(self, session):
        self.session = session

    def nll(self, params_array):
        gamma, delta, epsilon = params_array

        agent = MixtureAgent(maze = self.session.maze, beta = 1, xi = 3,
                gamma = gamma, delta = delta, epsilon = epsilon, nsteps = 3,
                loc = None)
        return self.neg_log_likelihood(agent)

    def nll_xi(self, params_array):
        xi, gamma, delta, epsilon = params_array 

        agent = MixtureAgent(maze = self.session.maze, beta = 1, xi = xi,
                gamma = gamma, delta = delta, epsilon = epsilon, nsteps = 3,
                loc = None)
        return self.neg_log_likelihood(agent)




class HeuristicLikelihood(Likelihood):

    def __init__(self, session):
        self.session = session

    def new_nll(self, params_array):
        beta, xi = params_array
        agent = HeuristicAgent(beta, xi, loc=None)
        return self.neg_log_likelihood(agent)

    '''
    def neg_log_likelihood(self, params_array):
        beta, kappa, xi = params_array
        return super(HeuristicLikelihood, self).neg_log_likelihood((beta,
            kappa, xi))
    '''

    def neg_log_likelihood_beta(self, beta, kappa, xi):
        return super(HeuristicLikelihood, self).neg_log_likelihood((beta,
            kappa, xi))

    def neg_log_likelihood_beta_xi(self, beta_xi):
        beta, xi = beta_xi
        return super(HeuristicLikelihood, self).neg_log_likelihood((beta,
            1, xi))
