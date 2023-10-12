import numpy as np
from mazex.agents import HeuristicAgent, MixtureAgent

import math
import numba
import numexpr as ne


# @jit(nopython=True, nogil=True)
@numba.guvectorize("void(float64[:,:], float64[:,:])", "(m,n)->(m,n)", nopython=True)
def logsoftmax(x, log_probs):
    """Adapted from https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html"""
    for i in range(x.shape[0]):
        # LogSoftMax Implementation
        max_x = np.nanmax(x[i, :])
        exp_x = np.exp(x[i, :] - max_x)
        sum_exp_x = np.nansum(exp_x)
        log_sum_exp_x = np.log(sum_exp_x)
        max_plus_log_sum_exp_x = max_x + log_sum_exp_x
        log_probs[i, :] = x[i, :] - max_plus_log_sum_exp_x


# def logsoftmax(x):
#    """Adapted from https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html
#    """
#    # LogSoftMax Implementation
#    max_x = np.nanmax(x)
#    exp_x = np.exp(x - max_x)
#    sum_exp_x = np.nansum(exp_x)
#    log_sum_exp_x = np.log(sum_exp_x)
#    max_plus_log_sum_exp_x = max_x + log_sum_exp_x
#    log_probs = x - max_plus_log_sum_exp_x
#    return log_probs

# @numba.guvectorize('void(float64[:], float64[:,:], float64[:])', '(m),(m,n)->(m)', nopython=True)
# def logsoftmax_chosen_values(chosen_values, values, log_probs):
#    """Adapted from https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html
#    """
#    for i in range(values.shape[0]):
#        # LogSoftMax Implementation
#        max_x = np.nanmax(values[i, :])
#        exp_x = np.exp(values[i, :] - max_x)
#        sum_exp_x = np.nansum(exp_x)
#        log_sum_exp_x = np.log(sum_exp_x)
#        max_plus_log_sum_exp_x = max_x + log_sum_exp_x
#        log_probs[i] = chosen_values[i] - max_plus_log_sum_exp_x


@numba.guvectorize(
    "void(float64[:], float64[:,:], float64[:])", "(m),(m,n)->(m)", nopython=True
)
def logsoftmax_chosen_values(chosen_values, values, log_probs):
    """Adapted from https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html"""
    for i in range(values.shape[0]):
        # non_nan_vals = values[i, ~np.isnan(values[i, :])]
        non_nan_vals_list = []
        for j in range(values.shape[1]):
            val = values[i, j]
            if not math.isnan(val):
                non_nan_vals_list.append(val)
        non_nan_vals = np.array(non_nan_vals_list)
        # LogSoftMax Implementation
        max_x = np.max(non_nan_vals)
        exp_x = np.exp(non_nan_vals - max_x)
        sum_exp_x = np.sum(exp_x)
        log_sum_exp_x = np.log(sum_exp_x)
        max_plus_log_sum_exp_x = max_x + log_sum_exp_x
        log_probs[i] = chosen_values[i] - max_plus_log_sum_exp_x


@numba.vectorize("float64(float64, float64, float64, float64)", nopython=True)
def value_mean(a, b, c, d):
    """Calculate the mean of option values where absent options are NaN."""
    counts = 0
    tot = 0
    for w in (a, b, c, d):
        if not math.isnan(w):
            tot += w
            counts += 1
    return tot / counts


@numba.guvectorize("void(float64[:,:], float64[:])", "(m,n)->(m)", nopython=True)
def matrix_column_mean(matrix, output):
    """Calculate the mean of option values where absent options are NaN."""
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        counts = 0
        tot = 0
        for j in range(row.shape[0]):
            w = row[j]
            if not math.isnan(w):
                tot += w
                counts += 1
        output[i] = tot / counts


# Not being used
@numba.vectorize("float64(float64, float64, float64, float64)", nopython=True)
def softmax_denominator(a, b, c, d):
    """Calculate the denominator of the softmax function."""
    tot = 0
    for w in (a, b, c, d):
        if not math.isnan(w):
            tot += np.exp(w)
    return tot


# Not being used
@numba.guvectorize("void(float64[:,:], float64[:])", "(m,n)->(m)", nopython=True)
def softmax_denominator_vec(val_arr, denom):
    """Calculate the denominator of the softmax function."""
    for i in range(val_arr.shape[0]):
        vals = val_arr[i]
        for v in vals:
            if not math.isnan(v):
                denom[i] += np.exp(v)


@numba.guvectorize(
    "void(float64[:], float64[:,:], float64[:])", "(n),(n,m)->(n)", nopython=True
)
def log_likelihoods(chosen_values, choice_values, likelihoods):
    """Calculate the log likelihood for an array of chosen_values. The output
    is saved to the array likelihoods, which has to have the same shape as
    chosen_values."""
    for i in range(chosen_values.shape[0]):
        choice_vals = choice_values[i, :]
        denominator = 0
        for v in choice_vals:
            if not math.isnan(v):
                denominator += np.exp(v)
        likelihood = np.exp(chosen_values[i]) / denominator
        # the line below implements a protected log
        likelihoods[i] = np.log(((1e-200) / 2) + (1 - (1e-200)) * likelihood)


@numba.guvectorize(
    "void(float64[:], float64[:,:], float64[:], float64[:])",
    "(n),(n,m),(n)->(n)",
    nopython=True,
)
def log_likelihoods_given_beta(chosen_values, choice_values, beta, likelihoods):
    """Calculate the log likelihood for an array of chosen_values. The output
    is saved to the array likelihoods, which has to have the same shape as
    chosen_values."""
    for i in range(chosen_values.shape[0]):
        choice_vals = choice_values[i, :]
        numerator = np.exp(beta[i] * chosen_values[i])
        denominator = np.nansum(np.exp(beta[i] * choice_vals))
        # denominator = 0
        # for v in choice_vals:
        #    if not math.isnan(v):
        #        denominator += np.exp(v)
        likelihood = numerator / denominator
        # the line below implements a protected log
        likelihoods[i] = np.log(((1e-200) / 2) + (1 - (1e-200)) * likelihood)


def protected_log(x):
    """Return log of x protected against giving -inf for very small values of
    x."""
    return np.log(((1e-200) / 2) + (1 - (1e-200)) * x)


def average_step_likelihood(nll, n_choice_steps):
    """n_choice_steps is the number of steps excluding the Step at which agent
    is at the trial's last step, usually at the goal location, i.e. This
    corresponds to the total number of decisions made in a session.
    """
    avg_step_nll = nll / n_choice_steps
    avg_step_ll = -avg_step_nll
    return np.exp(avg_step_ll)


class Likelihood(object):
    """A class for the likelihood functions."""

    def __init__(self, session):
        """Agent should be initialized with 'loc' None."""
        # self.session = copy.deepcopy(session)
        # I don't think we need to deepcopy anymore
        self.session = session

    def neg_log_likelihood(self, agent):
        """Compute the likelihood of the data in the session given the
        parameter set specified in the init of this likelihood method.
        """
        first_loc = self.session.trials[0].steps[0].loc
        agent.loc = first_loc

        log_likelihood = 0
        for t in self.session.trials:
            agent.loc = None  # klugey but necessary because otherwise agent
            # won't allow teleportation
            agent.node_visit_cache.clear()
            agent.edge_visit_cache.clear()

            for s in t.steps[:-1]:  # last step is the goal
                agent.loc = s.loc
                values = agent.evaluate(s.available_nodes, t.goal)
                s.values.append(values)  # we save it for troubleshooting
                choice_prob = agent.choice_probs(values)[s.choice]
                s.probs.append(choice_prob)

                # TODO: change to protected log
                log_likelihood += protected_log(choice_prob)

        return -log_likelihood

    def vect_neg_log_likelihood(self, agent, beta, subtract_mean=True):
        # TODO: at the moment beta is a scalar but everything is prepared for
        # it to be a vector with a different value of beta per trial
        self.chosen_values, self.values = agent.evaluate_session(self.session)

        if subtract_mean:
            # self.mean_values = np.nanmean(self.comb_values, axis=1)
            self.mean_values = matrix_column_mean(self.values)  # pyright: ignore

            self.chosen_values -= self.mean_values
            # before we subtract the mean to self.comb_values (has 2 dims) we have
            #  to add the empty dimension to self.mean_values but not before
            #  subtracting self.mean_values from self.comb_chosen_values!!!
            self.mean_values = self.mean_values[..., np.newaxis]
            self.values -= self.mean_values

        self.beta_vec = beta * np.ones_like(self.chosen_values)
        log_likes = np.zeros(self.chosen_values.shape[0])
        log_likelihoods_given_beta(
            self.chosen_values, self.values, self.beta_vec, log_likes
        )
        # log_likes = log_likelihoods_given_beta(self.chosen_values, self.values, self.beta_vec)
        return -np.sum(log_likes)

    def vect_neg_log_likelihood_max(self, agent, beta, subtract_max=True):
        # TODO: at the moment beta is a scalar but everything is prepared for
        # it to be a vector with a different value of beta per trial
        self.chosen_values, self.values = agent.evaluate_session(self.session)

        if subtract_max:
            # self.mean_values = np.nanmean(self.comb_values, axis=1)
            self.max_values = np.nanmax(self.values, axis=1)

            self.chosen_values -= self.max_values
            # before we subtract the mean to self.comb_values (has 2 dims) we have
            #  to add the empty dimension to self.mean_values but not before
            #  subtracting self.mean_values from self.comb_chosen_values!!!
            self.max_values = self.max_values[..., np.newaxis]
            self.values -= self.max_values

        self.beta_vec = beta * np.ones_like(self.chosen_values)
        log_likes = np.zeros(self.chosen_values.shape[0])
        log_likelihoods_given_beta(
            self.chosen_values, self.values, self.beta_vec, log_likes
        )
        # log_likes = log_likelihoods_given_beta(self.chosen_values, self.values, self.beta_vec)
        return -np.sum(log_likes)

    def vect_neg_log_likelihood_logsoftmax(
        self, agent, beta, subtract_max=True, remove_single_choice=True
    ):
        # TODO: at the moment beta is a scalar but everything is prepared for
        # it to be a vector with a different value of beta per trial
        self.chosen_values, self.values = agent.evaluate_session(self.session)

        if remove_single_choice:
            multiple_choice_idx = np.nonzero(
                np.count_nonzero(~np.isnan(self.values), axis=1) > 1
            )[0]
            self.values = self.values[multiple_choice_idx, :]
            self.chosen_values = self.chosen_values[multiple_choice_idx]

        if subtract_max:
            # self.mean_values = np.nanmean(self.comb_values, axis=1)
            self.max_values = np.nanmax(self.values, axis=1)

            self.chosen_values -= self.max_values
            # before we subtract the mean to self.comb_values (has 2 dims) we have
            #  to add the empty dimension to self.mean_values but not before
            #  subtracting self.mean_values from self.comb_chosen_values!!!
            self.max_values = self.max_values[..., np.newaxis]
            self.values -= self.max_values

        # log_likes = np.zeros(self.chosen_values.shape[0])
        # logsoftmax_chosen_values(beta * self.chosen_values, beta * self.values, log_likes)
        log_likes = logsoftmax_chosen_values(
            beta * self.chosen_values, beta * self.values
        )
        return -np.sum(log_likes)

    def avg_step_likelihood(self, agent, beta, subtract_mean=True):
        nll = self.vect_neg_log_likelihood(agent, beta, subtract_mean=subtract_mean)
        n_choice_steps = self.session.count_total_steps()
        return average_step_likelihood(nll, n_choice_steps)


#    def minimize_nll(self, agent_class, ):
#
#        def nll(x):
#            agent = agent_class(*x)
#
#        minimize(self.ve


class MixtureLikelihood(Likelihood):
    def __init__(self, session, nsteps=2, centrality="edge_betweenness"):
        # TODO: turn nsteps into a property and make it such that changing it
        # triggers recalculation of (only) the nsteps likelihood components
        super(MixtureLikelihood, self).__init__(session)
        self.agent = MixtureAgent(
            maze=self.session.maze,
            beta=1,
            xi=1,
            gamma=1,
            delta=1,
            epsilon=1,
            zeta=1,
            pi=1,
            nsteps=nsteps,
            centrality=centrality,
            loc=None,
        )

        (
            self.heuristic_chosen_values,
            self.heuristic_values,
        ) = self.agent.heuristic_agent.evaluate_session(session)

        (
            self.omniscient_chosen_values,
            self.omniscient_values,
        ) = self.agent.omniscient_agent.evaluate_session(session)

        (
            self.nsteps_chosen_values,
            self.nsteps_values,
        ) = self.agent.nsteps_agent.evaluate_session(session)

        (
            self.centrality_chosen_values,
            self.centrality_values,
        ) = self.agent.centrality_agent.evaluate_session(session)

        (
            self.forward_chosen_values,
            self.forward_values,
        ) = self.agent.forward_agent.evaluate_session(session)

        (
            self.penalty_chosen_values,
            self.penalty_values,
        ) = self.agent.evaluate_penalties_session(session)

    # Deprecated!
    def slow_nll(self, params_array):
        """Deprecated!"""
        self.agent.gamma, self.agent.delta, self.agent.epsilon = params_array
        return self.neg_log_likelihood(self.agent)

    def gdex_nll(self, params_array, subtract_mean=True):
        gamma, delta, epsilon, xi = params_array
        zeta = 0
        return self.vect_nll(
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            xi=xi,
            subtract_mean=subtract_mean,
        )

    def gdzx_nll(self, params_array, subtract_mean=True):
        gamma, delta, zeta, xi = params_array
        epsilon = 0
        return self.vect_nll(
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            xi=xi,
            subtract_mean=subtract_mean,
        )

    def gdezx_nll(self, params_array, subtract_mean=True):
        gamma, delta, epsilon, zeta, xi = params_array
        return self.vect_nll(
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            xi=xi,
            subtract_mean=subtract_mean,
        )

    def gex_nll(self, params_array, subtract_mean=True):
        gamma, epsilon, xi = params_array
        delta = 0
        zeta = 0
        return self.vect_nll(
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            xi=xi,
            subtract_mean=subtract_mean,
        )

    def gdx_nll(self, params_array, subtract_mean=True):
        gamma, delta, xi = params_array
        epsilon = 0
        zeta = 0
        return self.vect_nll(
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            xi=xi,
            subtract_mean=subtract_mean,
        )

    def gdpi_nll(self, params_array, subtract_mean=True):
        gamma, delta, pi = params_array
        epsilon = 0
        zeta = 0
        xi = 0
        return self.vect_nll(
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            pi=pi,
            xi=xi,
            subtract_mean=subtract_mean,
        )

    def gzx_nll(self, params_array, subtract_mean=True):
        gamma, zeta, xi = params_array
        delta = 0
        epsilon = 0
        return self.vect_nll(
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            xi=xi,
            subtract_mean=subtract_mean,
        )

    def gx_nll(self, params_array, subtract_mean=True):
        gamma, xi = params_array
        delta = 0
        epsilon = 0
        zeta = 0
        return self.vect_nll(
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            xi=xi,
            subtract_mean=subtract_mean,
        )

    def dx_nll(self, params_array, subtract_mean=True):
        delta, xi = params_array
        gamma = 0
        epsilon = 0
        zeta = 0
        return self.vect_nll(
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            xi=xi,
            subtract_mean=subtract_mean,
        )

    def x_nll(self, xi, subtract_mean=True):
        gamma = 0
        delta = 0
        epsilon = 0
        zeta = 0
        return self.vect_nll(
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            xi=xi,
            subtract_mean=subtract_mean,
        )

    def vect_nll(
        self, gamma, delta, epsilon, zeta, pi, xi, subtract_mean=True, numexpr=True
    ):
        # this is essentially a vectorized softmax with beta = 1
        if numexpr:
            shcv = self.heuristic_chosen_values
            socv = self.omniscient_chosen_values
            sncv = self.nsteps_chosen_values
            sccv = self.centrality_chosen_values
            spcv = self.penalty_chosen_values
            spicv = self.forward_chosen_values

            self.comb_chosen_values = ne.evaluate(
                "(gamma * shcv\
                    + delta * socv\
                    + epsilon * sncv\
                    + zeta * sccv\
                    + pi * spicv\
                    - xi * spcv)"
            )

            shv = self.heuristic_values
            sov = self.omniscient_values
            snv = self.nsteps_values
            scv = self.centrality_values
            spv = self.penalty_values
            spiv = self.forward_values

            self.comb_values = ne.evaluate(
                "(gamma * shv\
                    + delta * sov\
                    + epsilon * snv\
                    + zeta * scv\
                    + pi * spiv\
                    - xi * spv)"
            )
        else:
            self.comb_chosen_values = (
                gamma * self.heuristic_chosen_values
                + delta * self.omniscient_chosen_values
                + epsilon * self.nsteps_chosen_values
                + zeta * self.centrality_chosen_values
                + pi * self.forward_chosen_values
                - xi * self.penalty_chosen_values
            )

            self.comb_values = (
                gamma * self.heuristic_values
                + delta * self.omniscient_values
                + epsilon * self.nsteps_values
                + zeta * self.centrality_values
                + pi * self.forward_values
                - xi * self.penalty_values
            )

        if subtract_mean:
            # self.mean_values = np.nanmean(self.comb_values, axis=1)
            self.mean_values = matrix_column_mean(self.comb_values)

            self.comb_chosen_values = self.comb_chosen_values - self.mean_values
            # before we subtract the mean to self.comb_values (has 2 dims) we have
            #  to add the empty dimension to self.mean_values but not before
            #  subtracting self.mean_values from self.comb_chosen_values!!!
            self.mean_values = self.mean_values[..., np.newaxis]
            self.comb_values = self.comb_values - self.mean_values

        # for the denominator we have to use np.nansum instead of a normal sum
        # because for some steps there aren't 4 options and therefore the
        # 'missing' elements are nan on the values array
        #
        # Below is an alternative, cleaner but slower form of the same
        # calculation
        # self.choice_probs = np.exp(self.comb_chosen_values)/np.nansum(
        #        np.exp(self.comb_values), axis=1)
        #
        # denominator = softmax_denominator(self.comb_values[:,0],
        # self.comb_values[:,1], self.comb_values[:,2], self.comb_values[:,3])
        # self.choice_probs = np.exp(self.comb_chosen_values) / denominator
        # return -np.sum(protected_log(self.choice_probs))

        log_likes = np.empty(self.comb_chosen_values.shape[0])
        log_likelihoods(self.comb_chosen_values, self.comb_values, log_likes)

        return -np.sum(log_likes)


class HeuristicLikelihood(Likelihood):
    def __init__(self, session):
        self.session = session

    def new_nll(self, params_array):
        beta, xi = params_array
        agent = HeuristicAgent(beta, xi, loc=None)
        return self.neg_log_likelihood(agent)

    """
    def neg_log_likelihood(self, params_array):
        beta, kappa, xi = params_array
        return super(HeuristicLikelihood, self).neg_log_likelihood((beta,
            kappa, xi))
    """

    def neg_log_likelihood_beta(self, beta, kappa, xi):
        return super(HeuristicLikelihood, self).neg_log_likelihood((beta, kappa, xi))

    def neg_log_likelihood_beta_xi(self, beta_xi):
        beta, xi = beta_xi
        return super(HeuristicLikelihood, self).neg_log_likelihood((beta, 1, xi))
