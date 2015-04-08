import copy
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
        self.session = copy.deepcopy(session)

    def neg_log_likelihood(self, agent):
        """Compute the likelihood of the data in the session gien the
        parameter set specified in the init of this likelihood method.
        """
        first_loc = self.session.trials[0].steps[0].loc
        agent.loc = first_loc

        log_likelihood = 0
        for t in self.session.trials:
            agent.loc = None # klugey but necessary because otherwise agent
            # won't allow teleportation
            agent.node_visit_cache.clear()
            agent.edge_visit_cache.clear()

            for s in t.steps[:-1]: # last step is the goal
                agent.loc = s.loc
                values = agent.evaluate(s.available_nodes, t.goal)
                s.values.append(values) # we save it for troubleshooting
                choice_prob = agent.choice_probs(values)[s.choice]
                s.probs.append(choice_prob)

                #TODO: change to protected log
                log_likelihood += protected_log(choice_prob)

        return -log_likelihood

class MixtureLikelihood(Likelihood):

    def __init__(self, session):
        super(MixtureLikelihood, self).__init__(session)
        self.agent = MixtureAgent(maze = self.session.maze, beta = 1, xi = 0,
                gamma = None, delta = None, epsilon = None, nsteps = 2,
                loc = None)

        (self.heuristic_chosen_values, self.heuristic_values) = (
                self.agent.heuristic_agent.evaluate_session(session))
        self.sum_heuristic_values = np.nansum(self.heuristic_values, axis=1)
        
        (self.omniscient_chosen_values, self.omniscient_values) = (
                self.agent.omniscient_agent.evaluate_session(session))
        self.sum_omniscient_values = np.nansum(self.omniscient_values, axis=1)

        (self.nsteps_chosen_values, self.nsteps_values) = (
                self.agent.nsteps_agent.evaluate_session(session))
        self.sum_nsteps_values = np.nansum(self.nsteps_values, axis=1)

        #not being used yet
        self.chosen_values = np.vstack((self.heuristic_chosen_values,
            self.omniscient_chosen_values, self.nsteps_chosen_values))

    def nll(self, params_array):
        self.agent.gamma, self.agent.delta, self.agent.epsilon = params_array

        return self.neg_log_likelihood(self.agent)

    def vect_nll(self, params_array):
        gamma, delta, epsilon = params_array

        # this is essentially a vectorized softmax with beta = 1
        self.comb_chosen_values = (gamma * self.heuristic_chosen_values +
                delta * self.omniscient_chosen_values +
                epsilon * self.nsteps_chosen_values)

        self.comb_values = (gamma * self.heuristic_values +
                delta * self.omniscient_values +
                epsilon * self.nsteps_values)

        # for the denominator we have to use np.nansum instead of a normal sum
        # because for some steps there aren't 4 options and therefore the
        # 'missing' elements are nan on the values array
        
        self.choice_probs = np.exp(self.comb_chosen_values)/np.nansum(
                np.exp(self.comb_values), axis=1)

        return -np.sum(protected_log(self.choice_probs))
        
    def nll_xi(self, params_array):
        (self.agent.xi, self.agent.gamma, self.agent.delta,
                self.agent.epsilon) = params_array 

        return self.neg_log_likelihood(self.agent)

    def nll_gd(self, params_array):
        self.agent.gamma, self.agent.delta = params_array

        return self.neg_log_likelihood(self.agent)



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
