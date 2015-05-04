import copy
import numpy as np
from mazex.agents import HeuristicAgent, MixtureAgent

import math
import numba

# Not being used
@numba.vectorize('float64(float64, float64, float64, float64)', nopython=True)
def value_mean(a, b, c, d):
    """Calculate the mean of option values where absent options are NaN."""
    counts = 0
    tot = 0
    for w in (a, b, c, d):
        if not math.isnan(w):
            tot += w 
            counts += 1
    return tot/counts

# Not being used
@numba.vectorize('float64(float64, float64, float64, float64)', nopython=True)
def softmax_denominator(a, b, c, d):
    """Calculate the denominator of the softmax function."""
    tot = 0
    for w in (a, b, c, d):
        if not math.isnan(w):
            tot += np.exp(w) 
    return tot

# Not being used
@numba.guvectorize('void(float64[:,:], float64[:])', '(m,n)->(m)', nopython=True)
def softmax_denominator_vec(val_arr, denom):
    """Calculate the denominator of the softmax function."""
    for i in range(val_arr.shape[0]):
        vals = val_arr[i]
        for v in vals:
            if not math.isnan(v):
                denom[i] += np.exp(v) 

@numba.guvectorize('void(float64[:], float64[:,:], float64[:])',
'(n),(n,m)->(n)', nopython=True) 
def log_likelihoods(chosen_values, choice_values, likelihoods):
    """Calculate the log likelihood for an array of chosen_values. The output
    is saved to the array likelihoods, which has to have the same shape as
    chosen_values."""
    for i in range(chosen_values.shape[0]):
        choice_vals = choice_values[i,:]
        denominator = 0
        for v in choice_vals:
            if not math.isnan(v):
                denominator += np.exp(v) 
        likelihood = np.exp(chosen_values[i]) / denominator
        # the line below implements a protected log
        likelihoods[i] = np.log(((1e-200)/2)+(1-(1e-200))*likelihood)
        
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
        self.agent = MixtureAgent(maze = self.session.maze, beta = 1, xi = 1,
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

        (self.penalty_chosen_values, self.penalty_values) = (
                self.agent.evaluate_penalties_session(session))
        self.sum_penalty_values = np.nansum(self.nsteps_values, axis=1)

    def slow_nll(self, params_array):
        self.agent.gamma, self.agent.delta, self.agent.epsilon = params_array

        return self.neg_log_likelihood(self.agent)

    def gdex_nll(self, params_array, subtract_mean=True):
        gamma, delta, epsilon, xi = params_array

        return self.vect_nll(gamma, delta, epsilon, xi,
                subtract_mean=subtract_mean)

    def gdx_nll(self, params_array, subtract_mean=True):
        gamma, delta, xi = params_array
        epsilon = 0

        return self.vect_nll(gamma, delta, epsilon, xi,
                subtract_mean=subtract_mean)

    def gx_nll(self, params_array, subtract_mean=True):
        gamma, xi = params_array
        delta = 0
        epsilon = 0

        return self.vect_nll(gamma, delta, epsilon, xi,
                subtract_mean=subtract_mean)

    def dx_nll(self, params_array, subtract_mean=True):
        delta, xi = params_array
        gamma = 0
        epsilon = 0

        return self.vect_nll(gamma, delta, epsilon, xi,
                subtract_mean=subtract_mean)

    def x_nll(self, xi, subtract_mean=True):
        gamma = 0
        delta = 0
        epsilon = 0

        return self.vect_nll(gamma, delta, epsilon, xi,
                subtract_mean=subtract_mean)

    #@jit(nogil=True)
    def vect_nll(self, gamma, delta, epsilon, xi, subtract_mean=True):

        # this is essentially a vectorized softmax with beta = 1
        self.comb_chosen_values = (gamma * self.heuristic_chosen_values +
                delta * self.omniscient_chosen_values +
                epsilon * self.nsteps_chosen_values -
                xi * self.penalty_chosen_values)

        self.comb_values = (gamma * self.heuristic_values +
                delta * self.omniscient_values +
                epsilon * self.nsteps_values -
                xi * self.penalty_values)
        
        if subtract_mean:
            self.mean_values = np.nanmean(self.comb_values, axis=1)
            self.comb_chosen_values = self.comb_chosen_values - self.mean_values
            # before we subtract the mean to self.comb_values (has 2 dims) we have
            #  to add the empty dimension to self.mean_values but not before
            #  subtracting self.mean_values from self.comb_chosen_values!!!
            self.mean_values = self.mean_values[..., np.newaxis]
            self.comb_values = self.comb_values - self.mean_values

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
