import numpy as np
import mazex.agents

def protected_log(x):
    'Return log of x protected against giving -inf for very small values of x.'
    return np.log(((1e-200)/2)+(1-(1e-200))*x)

class Likelihood(object):
    """A class for the likelihood functions."""

    def __init__(self, session, agent):
        """Agent should be initialized with 'loc' None.
        """
        self.session = session
        self.agent = agent

    def neg_log_likelihood(self):
        """Compute the likelihood of the data in the session gien the
        parameter set specified in the init of this likelihood method.
        """
        first_loc = self.session.trials[0].steps[0].loc
        self.agent.loc = first_loc

        log_likelihood = 0
        for t in self.session.trials:
            self.agent.node_visit_cache.clear()
            self.agent.edge_visit_cache.clear()

            for s in t.steps[:-1]: # last step is the goal
                self.agent.loc = s.loc
                values = self.agent.evaluate(s.available_nodes, t.goal)
                choice_prob = self.agent.choice_probs(values)[s.choice]

                #TODO: change to protected log
                log_likelihood += protected_log(choice_prob)

        return -log_likelihood

class HeuristicLikelihood(Likelihood):

    def __init__(self, session):
        self.session = session

        self.agent_class = mazex.agents.HeuristicAgent

    def neg_log_likelihood(self, params_array):
        beta, kappa, xi = params_array
        return super(HeuristicLikelihood, self).neg_log_likelihood((beta,
            kappa, xi))

    def neg_log_likelihood_beta(self, beta, kappa, xi):
        return super(HeuristicLikelihood, self).neg_log_likelihood((beta,
            kappa, xi))

    def neg_log_likelihood_beta_xi(self, beta_xi):
        beta, xi = beta_xi
        return super(HeuristicLikelihood, self).neg_log_likelihood((beta,
            1, xi))
