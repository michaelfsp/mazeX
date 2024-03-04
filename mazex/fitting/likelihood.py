import numpy as np
from mazex.agents import HeuristicAgent, MixtureAgent
from mazex.data.load import (
    load_steps_multi,
    compute_features,
    load_session_from_steps_df,
)
from mazex.utils import (
    pad_with_nan,
    list_to_cardinal_columns,
    maze_loc_to_cardinal_dict,
)

import math
import numba
import numexpr as ne

import pandas as pd

from functools import partial

idx = pd.IndexSlice


def BIC(n_params, total_choices, nll_opt):
    """Compute the Bayesian Information Criterion"""
    BIC = n_params * np.log(total_choices) - 2 * (-nll_opt)
    return BIC


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
    avg_step_nll = nll / float(n_choice_steps)
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
    def __init__(
        self,
        session,
        nsteps=2,
        centrality="edge_betweenness",
        penalty_criterium="node",
        pandas_interface=False,
    ):
        # TODO: turn nsteps into a property and make it such that changing it
        # triggers recalculation of (only) the nsteps likelihood components
        super(MixtureLikelihood, self).__init__(session)
        init_loc = session.trials[0].steps[0].loc
        init_heading = session.trials[0].steps[0].heading

        # HACK: this if should be removed
        if penalty_criterium in ("node, edge_unidir", "node_lag2"):
            penalty_criterium_agent = penalty_criterium
        else:
            penalty_criterium_agent = "node"

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
            penalty_criterium=penalty_criterium_agent,
            loc=init_loc,
            heading=init_heading,
        )

        if pandas_interface:
            session_df = load_steps_multi([session])
            session_df = compute_features(session_df)
            self.session_df = session_df

            self.heuristic_chosen_values = np.cos(np.array(session_df.angle_to_goal))
            # we have to pad lists with nans because different lists in the list of lists have different lengths
            all_angle_to_goal_padded = pad_with_nan(
                session_df.all_angle_to_goal, max_len=4
            )
            self.heuristic_values = np.cos(np.array(all_angle_to_goal_padded))

            self.forward_chosen_values = np.cos(np.array(session_df.angle_to_heading))
            # we have to pad lists with nans because different lists in the list of lists have different lengths
            all_angle_to_heading_padded = pad_with_nan(
                session_df.all_angle_to_heading, max_len=4
            )
            self.forward_values = np.cos(np.array(all_angle_to_heading_padded))

            angle_to_heading_array = session_df.angle_to_heading.to_numpy()

            self.nobackward_chosen_values = np.where(
                angle_to_heading_array == np.pi, -1, 1
            )

            # we set the values smaller than np.pi to 1 and keep np.pi and np.nan as they are
            self.nobackward_values = np.where(
                all_angle_to_heading_padded < np.pi, 1, all_angle_to_heading_padded
            )

            # we set np.pi to -1
            self.nobackward_values = np.where(
                self.nobackward_values == np.pi, -1, self.nobackward_values
            )

            if penalty_criterium == "node":
                self.constant_penalty_chosen_values = np.array(
                    session_df.chosen_node_visited
                )
                self.constant_penalty_values = np.array(
                    pad_with_nan(session_df.available_node_visited)
                )

            elif penalty_criterium == "edge_unidir":
                self.constant_penalty_chosen_values = np.array(
                    session_df.chosen_dir_edge_visited
                )
                self.constant_penalty_values = np.array(
                    pad_with_nan(session_df.available_dir_edge_visited)
                )

            if penalty_criterium == "node_lag2":
                self.constant_penalty_chosen_values = np.array(
                    session_df.chosen_node_visited_lag2
                )
                self.constant_penalty_values = np.array(
                    pad_with_nan(session_df.available_node_visited_lag2)
                )

            self.omniscient_chosen_values = -np.array(session_df.geodesic_delta)
            self.omniscient_values = -np.array(
                pad_with_nan(session_df.all_geodesic_deltas)
            )

            # HACK: this is a temporary fix
            (
                self.nsteps_chosen_values,
                self.nsteps_values,
            ) = self.agent.nsteps_agent.evaluate_session(session)

            (
                self.centrality_chosen_values,
                self.centrality_values,
            ) = self.agent.centrality_agent.evaluate_session(session)

            self.agent.penalty_type = "exponential"
            (
                self.penalty_chosen_values,
                self.penalty_values,
            ) = self.agent.evaluate_penalties_session(session)

            self.penalty_chosen_values = (2 * self.penalty_chosen_values) - 1
            self.penalty_values = (2 * self.penalty_values) - 1

        else:
            #            (
            #                self.heuristic_chosen_values,
            #                self.heuristic_values,
            #            ) = self.agent.heuristic_agent.evaluate_session(session)
            #
            #            (
            #                self.omniscient_chosen_values,
            #                self.omniscient_values,
            #            ) = self.agent.omniscient_agent.evaluate_session(session)
            #
            #            (
            #                self.nsteps_chosen_values,
            #                self.nsteps_values,
            #            ) = self.agent.nsteps_agent.evaluate_session(session)
            #
            #            (
            #                self.centrality_chosen_values,
            #                self.centrality_values,
            #            ) = self.agent.centrality_agent.evaluate_session(session)
            #
            #            (
            #                self.forward_chosen_values,
            #                self.forward_values,
            #            ) = self.agent.forward_agent.evaluate_session(session)

            (
                heuristicv,
                omniscientv,
                nstepsv,
                centralityv,
                forwardv,
            ) = self.agent.evaluate_session_subagents(session)

            (
                self.heuristic_chosen_values,
                self.heuristic_values,
            ) = heuristicv

            (
                self.omniscient_chosen_values,
                self.omniscient_values,
            ) = omniscientv

            (
                self.nsteps_chosen_values,
                self.nsteps_values,
            ) = nstepsv

            (
                self.centrality_chosen_values,
                self.centrality_values,
            ) = centralityv

            (
                self.forward_chosen_values,
                self.forward_values,
            ) = forwardv

            self.nobackward_chosen_values = np.where(
                self.forward_chosen_values == np.cos(np.pi), -1, 1
            )

            # we set the values smaller than np.pi to 1 and keep np.pi and np.nan as they are
            self.nobackward_values = np.where(
                self.forward_values > np.cos(np.pi), 1, self.forward_values
            )

            # we set np.pi to -1
            self.nobackward_values = np.where(
                self.forward_values == np.cos(np.pi), -1, self.nobackward_values
            )

            self.agent.penalty_type = "constant"
            (
                self.constant_penalty_chosen_values,
                self.constant_penalty_values,
            ) = self.agent.evaluate_penalties_session(session)

            self.constant_penalty_chosen_values = (
                2 * self.constant_penalty_chosen_values
            ) - 1
            self.constant_penalty_values = (2 * self.constant_penalty_values) - 1

            self.agent.penalty_type = "exponential"
            (
                self.penalty_chosen_values,
                self.penalty_values,
            ) = self.agent.evaluate_penalties_session(session)

            self.penalty_chosen_values = (2 * self.penalty_chosen_values) - 1
            self.penalty_values = (2 * self.penalty_values) - 1

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

    def gdpipsi_nll(self, params_array, subtract_mean=True):
        gamma, delta, pi, psi = params_array
        epsilon = 0
        zeta = 0
        xi = 0
        return self.vect_nll(
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            zeta=zeta,
            pi=pi,
            psi=psi,
            xi=xi,
            subtract_mean=subtract_mean,
        )

    def gdphipsi_nll(self, params_array, subtract_mean=True):
        gamma, delta, phi, psi = params_array
        return self.vect_nll(
            gamma=gamma,
            delta=delta,
            phi=phi,
            psi=psi,
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
        self,
        gamma=0,
        delta=0,
        epsilon=0,
        zeta=0,
        pi=0,
        phi=0,
        psi=0,
        xi=0,
        subtract_mean=True,
        numexpr=True,
    ):
        # this is essentially a vectorized softmax with beta = 1
        if numexpr:
            shcv = self.heuristic_chosen_values
            socv = self.omniscient_chosen_values
            sncv = self.nsteps_chosen_values
            sccv = self.centrality_chosen_values
            spcv = self.penalty_chosen_values
            spicv = self.forward_chosen_values
            sphicv = self.nobackward_chosen_values
            spsicv = self.constant_penalty_chosen_values

            self.comb_chosen_values = ne.evaluate(
                "(gamma * shcv\
                    + delta * socv\
                    + epsilon * sncv\
                    + zeta * sccv\
                    + pi * spicv\
                    + phi * sphicv\
                    - psi * spsicv\
                    - xi * spcv)"
            )

            shv = self.heuristic_values
            sov = self.omniscient_values
            snv = self.nsteps_values
            scv = self.centrality_values
            spv = self.penalty_values
            spiv = self.forward_values
            sphiv = self.nobackward_values
            spsiv = self.constant_penalty_values

            self.comb_values = ne.evaluate(
                "(gamma * shv\
                    + delta * sov\
                    + epsilon * snv\
                    + zeta * scv\
                    + pi * spiv\
                    + phi * sphiv\
                    - psi * spsiv\
                    - xi * spv)"
            )
        else:
            self.comb_chosen_values = (
                gamma * self.heuristic_chosen_values
                + delta * self.omniscient_chosen_values
                + epsilon * self.nsteps_chosen_values
                + zeta * self.centrality_chosen_values
                + pi * self.forward_chosen_values
                + phi * self.nobackward_chosen_values
                - psi * self.constant_penalty_chosen_values
                - xi * self.penalty_chosen_values
            )

            self.comb_values = (
                gamma * self.heuristic_values
                + delta * self.omniscient_values
                + epsilon * self.nsteps_values
                + zeta * self.centrality_values
                + pi * self.forward_values
                + phi * self.nobackward_values
                - psi * self.constant_penalty_values
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


def vnav_vals(session_df, mazes_dict, func=np.cos):
    if func is None:

        def renormalize_angles(angle):
            return (2.0 * (-angle / np.pi)) + 1.0

        func = renormalize_angles

    vnav_chosen_vals = pd.Series(
        func(np.array(session_df.angle_to_goal)), index=session_df.index, name="vnav"
    )

    vnav_vals = list_to_cardinal_columns(
        session_df, "all_angle_to_goal", mazes_dict, new_column_name="vnav"
    )
    vnav_vals = func(vnav_vals)

    return (vnav_chosen_vals, vnav_vals)


def antibackward_vals(session_df, mazes_dict):
    # antibackward_chosen_vals = pd.Series(
    #     np.where(session_df.angle_to_heading == np.pi, -1.0, 1.0),
    #     index=session_df.index,
    #     name="antibackward",
    # )
    #
    antibackward_vals = list_to_cardinal_columns(
        session_df, "all_angle_to_heading", mazes_dict, new_column_name="antibackward"
    )

    # when angle is np.pi we set it to -1 (turn around)
    antibackward_vals[antibackward_vals == np.pi] = -1.0
    # when angle is != np.pi we set it to 1 (no need to turn around)
    antibackward_vals[antibackward_vals >= 0] = 1.0

    chosen_vals = []
    for i, row in session_df.iterrows():
        chosen_vals.append(
            antibackward_vals.loc[i, ("antibackward", row.cardinal_choice)]
        )

    antibackward_chosen_vals = pd.Series(
        chosen_vals, index=session_df.index, name="antibackward"
    )

    return (antibackward_chosen_vals, antibackward_vals)


def antiturn_vals(session_df, mazes_dict):
    # antiturn_chosen_vals = pd.Series(
    #     np.where(session_df.angle_to_heading == np.pi / 2, -1.0, 1.0),
    #     index=session_df.index,
    #     name="antiturn",
    # )
    antiturn_vals = list_to_cardinal_columns(
        session_df, "all_angle_to_heading", mazes_dict, new_column_name="antiturn"
    )

    antiturn_vals[antiturn_vals == np.pi / 2] = -1.0
    # when angle is != np.pi we set it to 1 (no need to turn around)
    antiturn_vals[antiturn_vals >= 0] = 1.0
    # # when angle is != np.pi we set it to 1 (no need to turn around)
    # antiturn_vals.where(antiturn_vals == np.pi / 2, -1.0, inplace=True)
    # # when angle is np.pi we set it to -1 (turn around)
    # antiturn_vals.mask(antiturn_vals == np.pi / 2, 1.0, inplace=True)

    chosen_vals = []
    for i, row in session_df.iterrows():
        chosen_vals.append(antiturn_vals.loc[i, ("antiturn", row.cardinal_choice)])

    antiturn_chosen_vals = pd.Series(
        chosen_vals, index=session_df.index, name="antiturn"
    )

    return (antiturn_chosen_vals, antiturn_vals)


def antiL_vals(session_df, mazes_dict):
    # antiturn_chosen_vals = pd.Series(
    #     np.where(session_df.angle_to_heading == np.pi / 2, -1.0, 1.0),
    #     index=session_df.index,
    #     name="antiturn",
    # )
    antiL_vals = list_to_cardinal_columns(
        session_df, "all_angle_to_heading", mazes_dict, new_column_name="antiL"
    )

    antiL_vals[antiL_vals == np.pi / 2] = -1.0
    # when angle is != np.pi we set it to 1 (no need to turn around)
    antiL_vals[antiL_vals >= 0] = 1.0
    # # when angle is != np.pi we set it to 1 (no need to turn around)
    # antiturn_vals.where(antiturn_vals == np.pi / 2, -1.0, inplace=True)
    # # when angle is np.pi we set it to -1 (turn around)
    # antiturn_vals.mask(antiturn_vals == np.pi / 2, 1.0, inplace=True)

    for i in range(len(antiL_vals)):
        row = antiL_vals.iloc[i, :]
        if any(row == -1):
            heading = session_df.iloc[i, :]["heading"]
            dfloc = row.name
            if heading == "v":
                antiL_vals.loc[dfloc, ("antiL", "W")] *= -1.0
            elif heading == "<":
                antiL_vals.loc[dfloc, ("antiL", "N")] *= -1.0
            elif heading == "^":
                antiL_vals.loc[dfloc, ("antiL", "E")] *= -1.0
            if heading == ">":
                antiL_vals.loc[dfloc, ("antiL", "S")] *= -1.0

    chosen_vals = []
    for i, row in session_df.iterrows():
        chosen_vals.append(antiL_vals.loc[i, ("antiL", row.cardinal_choice)])

    antiL_chosen_vals = pd.Series(chosen_vals, index=session_df.index, name="antiL")

    return (antiL_chosen_vals, antiL_vals)


def antiR_vals(session_df, mazes_dict):
    # antiturn_chosen_vals = pd.Series(
    #     np.where(session_df.angle_to_heading == np.pi / 2, -1.0, 1.0),
    #     index=session_df.index,
    #     name="antiturn",
    # )
    antiR_vals = list_to_cardinal_columns(
        session_df, "all_angle_to_heading", mazes_dict, new_column_name="antiR"
    )

    antiR_vals[antiR_vals == np.pi / 2] = -1.0
    # when angle is != np.pi we set it to 1 (no need to turn around)
    antiR_vals[antiR_vals >= 0] = 1.0
    # # when angle is != np.pi we set it to 1 (no need to turn around)
    # antiturn_vals.where(antiturn_vals == np.pi / 2, -1.0, inplace=True)
    # # when angle is np.pi we set it to -1 (turn around)
    # antiturn_vals.mask(antiturn_vals == np.pi / 2, 1.0, inplace=True)

    for i in range(len(antiR_vals)):
        row = antiR_vals.iloc[i, :]
        if any(row == -1):
            heading = session_df.iloc[i, :]["heading"]
            dfloc = row.name
            if heading == "v":
                antiR_vals.loc[dfloc, ("antiR", "E")] *= -1.0
            elif heading == "<":
                antiR_vals.loc[dfloc, ("antiR", "S")] *= -1.0
            elif heading == "^":
                antiR_vals.loc[dfloc, ("antiR", "W")] *= -1.0
            if heading == ">":
                antiR_vals.loc[dfloc, ("antiR", "N")] *= -1.0

    chosen_vals = []
    for i, row in session_df.iterrows():
        chosen_vals.append(antiR_vals.loc[i, ("antiR", row.cardinal_choice)])

    antiR_chosen_vals = pd.Series(chosen_vals, index=session_df.index, name="antiR")

    return (antiR_chosen_vals, antiR_vals)


def visited_node_lag2_vals(session_df, mazes_dict):
    """Warning: it is the symmetrical of what i had done before. This is to be added and not subtracted as a penalty. I.e., actual penalties are now negative values"""

    visited_node_lag2_chosen_vals = pd.Series(
        -session_df.chosen_node_visited_lag2,
        index=session_df.index,
        name="visited_node_lag2",
    ).astype("float")

    visited_node_lag2_vals = -list_to_cardinal_columns(
        session_df,
        "available_node_visited_lag2",
        mazes_dict,
        new_column_name="visited_node_lag2",
    )

    visited_node_lag2_vals = visited_node_lag2_vals.astype("float")

    return (visited_node_lag2_chosen_vals, visited_node_lag2_vals)


def visited_node_exp_vals(session_df, mazes_dict):
    """Warning: it is the symmetrical of what i had done before. This is to be added and not subtracted as a penalty. I.e., actual penalties are now negative values"""

    session_from_df = load_session_from_steps_df(session_df, mazes_dict)

    like = MixtureLikelihood(
        session_from_df, penalty_criterium="node", pandas_interface=True
    )

    visited_node_exp_chosen_vals = pd.Series(
        -like.penalty_chosen_values, index=session_df.index, name="visited_node_exp"
    )

    session_df["available_node_visited_exp"] = (-like.penalty_values).tolist()
    visited_node_exp_vals = list_to_cardinal_columns(
        session_df,
        "available_node_visited_exp",
        mazes_dict,
        new_column_name="visited_node_exp",
    )

    return (visited_node_exp_chosen_vals, visited_node_exp_vals)


def optimal_vals(session_df, mazes_dict):
    chosen_vals = pd.Series(
        -session_df.geodesic_delta, index=session_df.index, name="optimal"
    ).astype("float")

    vals = -list_to_cardinal_columns(
        session_df, "all_geodesic_deltas", mazes_dict, new_column_name="optimal"
    )
    vals = vals.astype("float")

    return (chosen_vals, vals)


def euclidean_vals(session_df, mazes_dict):
    chosen_vals = pd.Series(
        -session_df.euclidean_delta, index=session_df.index, name="euclidean"
    )

    vals = -list_to_cardinal_columns(
        session_df, "all_euclidean_delta", mazes_dict, new_column_name="euclidean"
    )

    return (chosen_vals, vals)


def get_chosen_vals_from_vals(vals, session_df, label):
    chosen_vals = []
    for (i, df_row), (j, vals_row) in zip(session_df.iterrows(), vals.iterrows()):
        # vals is a MultiIndex so we need feature_name and cardinal direction as indices
        chosen_vals.append(vals_row.loc[label, df_row.cardinal_choice])

    chosen_vals = pd.Series(chosen_vals, index=session_df.index, name=f"{label}")
    return chosen_vals


def thigmotaxis_vals(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="thigmotaxis"
    )

    def thigmotaxis(location):
        # pd.isna doesn't fail when it is a tuple, unlike np.isnan
        if pd.isna(location):
            return np.nan
        else:
            if location[0] in [0, 5] or location[1] in [0, 5]:
                # it's a border
                val = 1.0
            else:
                val = -1.0
            return val

    vals = vals.map(thigmotaxis)

    chosen_vals = get_chosen_vals_from_vals(vals, session_df, label="thigmotaxis")

    return (chosen_vals, vals)


def renormalize_geodesic_deltas(geodelta_array):
    if type(geodelta_array) is not np.ndarray:
        geodelta_array = np.array(geodelta_array)
    idxs_available_choices = np.nonzero(~np.isnan(geodelta_array))[0].tolist()
    # if all non nan values are the same or there is only one nan value
    if np.nanstd(geodelta_array) == 0:
        normalized_deltas = np.full(4, np.nan)
        if np.any(geodelta_array[idxs_available_choices] > 0):
            normalized_deltas[idxs_available_choices] = -1
        elif np.any(geodelta_array[idxs_available_choices] == 0):
            normalized_deltas[idxs_available_choices] = 0
        elif np.any(geodelta_array[idxs_available_choices] < 0):
            normalized_deltas[idxs_available_choices] = +1
    else:
        # we make the worst option, i.e. the one resulting in the highest increase of geodesic distance, 0.0
        geodelta_array -= np.nanmax(geodelta_array)
        # we make the best option, i.e. the one resulting in the highest decrease of distance, 1.0
        geodelta_array /= np.nanmin(geodelta_array)
        # we shift and scale
        normalized_deltas = 2 * (geodelta_array - 0.5)

    return normalized_deltas.tolist()


def prior_maze_vals(session_df, mazes_dict, prior_dict):
    chosen_vals = []
    vals = {"W": [], "S": [], "N": [], "E": []}
    for i, row in session_df.iterrows():
        prior_maze = prior_dict[row.maze_name]
        current_maze = mazes_dict[row.maze_name]

        cardinal_dict = maze_loc_to_cardinal_dict(current_maze, row.location)

        prior_geodesic_deltas = []
        for node_choice in cardinal_dict.values():
            if node_choice:
                prior_geodesic_deltas.append(
                    prior_maze.distances_dictionary[node_choice][row.goal]
                    - prior_maze.distances_dictionary[row.location][row.goal]
                )
            else:
                prior_geodesic_deltas.append(np.nan)

        normalized_prior_deltas = renormalize_geodesic_deltas(prior_geodesic_deltas)

        for i, cardinal_dir in enumerate(cardinal_dict.keys()):
            vals[cardinal_dir].append(normalized_prior_deltas[i])

        # chosen_vals.append(
        #     prior_maze.distances_dictionary[row.choice][row.goal]
        #     - prior_maze.distances_dictionary[row.location][row.goal]
        # )

        chosen_vals.append(vals[row.cardinal_choice][-1])

    chosen_vals = pd.Series(chosen_vals, index=session_df.index, name="prior_maze")
    vals = pd.DataFrame(vals, index=session_df.index)

    column_tuples = [
        ("prior_maze", "W"),
        ("prior_maze", "S"),
        ("prior_maze", "N"),
        ("prior_maze", "E"),
    ]
    columns_index = pd.MultiIndex.from_tuples(column_tuples)
    vals.columns = columns_index

    return (chosen_vals, vals)


def prior_maze_overlap_vals(session_df, mazes_dict, prior_dict):
    chosen_vals = []
    vals = {"W": [], "S": [], "N": [], "E": []}
    for i, row in session_df.iterrows():
        prior_maze = prior_dict[row.maze_name]
        current_maze = mazes_dict[row.maze_name]

        cardinal_dict = maze_loc_to_cardinal_dict(current_maze, row.location)

        prior_geodesic_deltas = []
        for node_choice in cardinal_dict.values():
            if node_choice:
                prior_maze_geodesic_delta = (
                    prior_maze.distances_dictionary[node_choice][row.goal]
                    - prior_maze.distances_dictionary[row.location][row.goal]
                )
                if abs(prior_maze_geodesic_delta) == 1:
                    prior_geodesic_deltas.append(
                        prior_maze.distances_dictionary[node_choice][row.goal]
                        - prior_maze.distances_dictionary[row.location][row.goal]
                    )
                else:
                    prior_geodesic_deltas.append(0)
            else:
                prior_geodesic_deltas.append(np.nan)

        # In the case of this variation of prior_maze componente, this step of renormalization
        # is only here to invert the value of this feature, compared to the geodesic delta, i.e.,
        # when the delta is negative the evaluation as per the feature is negative
        # in this case it's just 0 (for no overlap), -1 mapped to +1, and +1 mapped to -1
        normalized_prior_deltas = renormalize_geodesic_deltas(prior_geodesic_deltas)

        for i, cardinal_dir in enumerate(cardinal_dict.keys()):
            vals[cardinal_dir].append(normalized_prior_deltas[i])

        # chosen_vals.append(
        #     prior_maze.distances_dictionary[row.choice][row.goal]
        #     - prior_maze.distances_dictionary[row.location][row.goal]
        # )

        chosen_vals.append(vals[row.cardinal_choice][-1])

    chosen_vals = pd.Series(
        chosen_vals, index=session_df.index, name="prior_maze_overlap"
    )
    vals = pd.DataFrame(vals, index=session_df.index)

    column_tuples = [
        ("prior_maze_overlap", "W"),
        ("prior_maze_overlap", "S"),
        ("prior_maze_overlap", "N"),
        ("prior_maze_overlap", "E"),
    ]
    columns_index = pd.MultiIndex.from_tuples(column_tuples)
    vals.columns = columns_index

    return (chosen_vals, vals)


def betweenness_vals(session_df, mazes_dict):
    chosen_vals = []

    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="betweenness"
    )

    maze_name = session_df.maze_name.unique().tolist()[0]
    maze = mazes_dict[maze_name]

    def loc_to_betweenness_scaled(loc):
        if loc is np.nan:
            return np.nan
        else:
            min_betweenness = min(list(maze.betweenness_centrality.values()))
            max_betweenness = max(list(maze.betweenness_centrality.values()))
            loc_betweenness = maze.betweenness_centrality[loc]
            # we bring it to scale of [0, 1]
            scaled_betweenness = loc_betweenness - min_betweenness
            scaled_betweenness = scaled_betweenness / (
                max_betweenness - min_betweenness
            )
            # we bring it to scale [-1, 1]
            scaled_betweenness = (scaled_betweenness * 2) - 1

            return scaled_betweenness

    vals = vals.map(loc_to_betweenness_scaled)

    for i, row in session_df.iterrows():
        chosen_vals.append(vals.loc[i, ("betweenness", row.cardinal_choice)])

    chosen_vals = pd.Series(chosen_vals, index=session_df.index, name="betweenness")

    return (chosen_vals, vals)


def degree_vals(session_df, mazes_dict):
    chosen_vals = []

    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="degree"
    )

    maze_name = session_df.maze_name.unique().tolist()[0]
    maze = mazes_dict[maze_name]

    def loc_to_degree(loc):
        if loc is np.nan:
            return np.nan
        else:
            # min_betweenness = min(list(maze.betweenness_centrality.values()))
            # max_betweenness = max(list(maze.betweenness_centrality.values()))
            # loc_betweenness = maze.betweenness_centrality[loc]
            # # we bring it to scale of [0, 1]
            # scaled_betweenness = loc_betweenness - min_betweenness
            # scaled_betweenness = scaled_betweenness / (
            #     max_betweenness - min_betweenness
            # )
            # # we bring it to scale [-1, 1]
            # scaled_betweenness = (scaled_betweenness * 2) - 1

            return maze.degree[loc]

    vals = vals.map(loc_to_degree)

    for i, row in session_df.iterrows():
        chosen_vals.append(vals.loc[i, ("degree", row.cardinal_choice)])

    chosen_vals = pd.Series(chosen_vals, index=session_df.index, name="degree")

    return (chosen_vals, vals)


def manhattan_distance_multiplier(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="manhattan_dist_goal"
    )
    vals[~pd.isna(vals)] = 1.0

    vals = session_df.manhattan_dist_goal.to_numpy("float")[:, np.newaxis] * vals

    chosen_vals = get_chosen_vals_from_vals(
        vals, session_df, label="manhattan_dist_goal"
    )

    return (chosen_vals, vals)


def session_goal_block_multiplier(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="session_goal_block"
    )
    vals[~pd.isna(vals)] = 1.0

    vals = session_df.goal_block.to_numpy("float")[:, np.newaxis] * vals

    chosen_vals = get_chosen_vals_from_vals(
        vals, session_df, label="session_goal_block"
    )

    return (chosen_vals, vals)


def maze_day_multiplier(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="maze_day"
    )
    vals[~pd.isna(vals)] = 1.0

    vals = session_df.maze_day.to_numpy("float")[:, np.newaxis] * vals

    chosen_vals = get_chosen_vals_from_vals(vals, session_df, label="maze_day")

    return (chosen_vals, vals)


def maze_loc_count_multiplier(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="maze_loc_count"
    )
    vals[~pd.isna(vals)] = 1.0

    vals = session_df.maze_loc_count.to_numpy("float")[:, np.newaxis] * vals

    chosen_vals = get_chosen_vals_from_vals(vals, session_df, label="maze_loc_count")

    return (chosen_vals, vals)


def maze_loc_goal_count_multiplier(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="maze_loc_goal_count"
    )
    vals[~pd.isna(vals)] = 1.0

    vals = session_df.maze_loc_goal_count.to_numpy("float")[:, np.newaxis] * vals

    chosen_vals = get_chosen_vals_from_vals(
        vals, session_df, label="maze_loc_goal_count"
    )

    return (chosen_vals, vals)


def maze_goal_count_multiplier(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="maze_goal_count"
    )
    vals[~pd.isna(vals)] = 1.0

    vals = session_df.maze_goal_count.to_numpy("float")[:, np.newaxis] * vals

    chosen_vals = get_chosen_vals_from_vals(vals, session_df, label="maze_goal_count")

    return (chosen_vals, vals)


def maze_step_count_multiplier(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="maze_step_count"
    )
    vals[~pd.isna(vals)] = 1.0

    vals = session_df.maze_step_count.to_numpy("float")[:, np.newaxis] * vals

    chosen_vals = get_chosen_vals_from_vals(vals, session_df, label="maze_step_count")

    return (chosen_vals, vals)


def session_step_count_multiplier(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="session_step_count"
    )
    vals[~pd.isna(vals)] = 1.0

    vals = session_df.session_step_count.to_numpy("float")[:, np.newaxis] * vals

    chosen_vals = get_chosen_vals_from_vals(
        vals, session_df, label="session_step_count"
    )

    return (chosen_vals, vals)


def session_trial_count_multiplier(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="session_trial_count"
    )
    vals[~pd.isna(vals)] = 1.0

    vals = session_df.session_trial_count.to_numpy("float")[:, np.newaxis] * vals

    chosen_vals = get_chosen_vals_from_vals(
        vals, session_df, label="session_trial_count"
    )

    return (chosen_vals, vals)


def session_log_trial_count_multiplier(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df,
        "available_nodes",
        mazes_dict,
        new_column_name="session_log_trial_count",
    )
    vals[~pd.isna(vals)] = 1.0

    vals = session_df.session_trial_count.to_numpy("float")[:, np.newaxis] * vals
    # we add one so that we can take the log for the very first trial
    vals[~pd.isna(vals)] += 1.0
    vals = vals.map(np.log)

    chosen_vals = get_chosen_vals_from_vals(
        vals, session_df, label="session_log_trial_count"
    )

    return (chosen_vals, vals)


def bias_W(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="bias_W"
    )
    vals.loc[pd.notna(vals.loc[:, ("bias_W", "W")]), ("bias_W", "W")] = 1.0
    for other_dir in ("N", "S", "E"):
        vals.loc[
            pd.notna(vals.loc[:, ("bias_W", other_dir)]), ("bias_W", other_dir)
        ] = 0.0

    chosen_vals = get_chosen_vals_from_vals(vals, session_df, label="bias_W")

    return (chosen_vals, vals)


def bias_N(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="bias_N"
    )
    vals.loc[pd.notna(vals.loc[:, ("bias_N", "N")]), ("bias_N", "N")] = 1.0
    for other_dir in ("W", "S", "E"):
        vals.loc[
            pd.notna(vals.loc[:, ("bias_N", other_dir)]), ("bias_N", other_dir)
        ] = 0.0

    chosen_vals = get_chosen_vals_from_vals(vals, session_df, label="bias_N")

    return (chosen_vals, vals)


def bias_E(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="bias_E"
    )
    vals.loc[pd.notna(vals.loc[:, ("bias_E", "E")]), ("bias_E", "E")] = 1.0
    for other_dir in ("N", "S", "W"):
        vals.loc[
            pd.notna(vals.loc[:, ("bias_E", other_dir)]), ("bias_E", other_dir)
        ] = 0.0

    chosen_vals = get_chosen_vals_from_vals(vals, session_df, label="bias_E")

    return (chosen_vals, vals)


def bias_S(session_df, mazes_dict):
    vals = list_to_cardinal_columns(
        session_df, "available_nodes", mazes_dict, new_column_name="bias_S"
    )
    vals.loc[pd.notna(vals.loc[:, ("bias_S", "S")]), ("bias_S", "S")] = 1.0
    for other_dir in ("N", "E", "W"):
        vals.loc[
            pd.notna(vals.loc[:, ("bias_S", other_dir)]), ("bias_S", other_dir)
        ] = 0.0

    chosen_vals = get_chosen_vals_from_vals(vals, session_df, label="bias_S")

    return (chosen_vals, vals)


def vals_retriever(df, feature, mazes_dict, prior_dict):
    # TODO: Make this function able to retrieve interaction features
    feature_func_dict = {
        "vnav_cos": partial(vnav_vals, mazes_dict=mazes_dict, func=np.cos),
        "antibackward": partial(antibackward_vals, mazes_dict=mazes_dict),
        "visited_node_lag2": partial(visited_node_lag2_vals, mazes_dict=mazes_dict),
        "visited_node_exp": partial(visited_node_exp_vals, mazes_dict=mazes_dict),
        "optimal": partial(optimal_vals, mazes_dict=mazes_dict),
        "prior_maze": partial(
            prior_maze_vals, mazes_dict=mazes_dict, prior_dict=prior_dict
        ),
        "prior_maze_overlap": partial(
            prior_maze_overlap_vals, mazes_dict=mazes_dict, prior_dict=prior_dict
        ),
        "betweenness": partial(betweenness_vals, mazes_dict=mazes_dict),
        "degree": partial(degree_vals, mazes_dict=mazes_dict),
        "euclidean": partial(euclidean_vals, mazes_dict=mazes_dict),
        "vnav": partial(vnav_vals, mazes_dict=mazes_dict, func=None),
        "antiturn": partial(antiturn_vals, mazes_dict=mazes_dict),
        "thigmotaxis": partial(thigmotaxis_vals, mazes_dict=mazes_dict),
        "maze_day": partial(maze_day_multiplier, mazes_dict=mazes_dict),
        "session_goal_block": partial(
            session_goal_block_multiplier, mazes_dict=mazes_dict
        ),
        "manhattan_dist_goal": partial(
            manhattan_distance_multiplier, mazes_dict=mazes_dict
        ),
        "maze_loc_count": partial(maze_loc_count_multiplier, mazes_dict=mazes_dict),
        "maze_loc_goal_count": partial(
            maze_loc_goal_count_multiplier, mazes_dict=mazes_dict
        ),
        "maze_goal_count": partial(maze_goal_count_multiplier, mazes_dict=mazes_dict),
        "maze_step_count": partial(maze_step_count_multiplier, mazes_dict=mazes_dict),
        "session_step_count": partial(
            session_step_count_multiplier, mazes_dict=mazes_dict
        ),
        "session_trial_count": partial(
            session_trial_count_multiplier, mazes_dict=mazes_dict
        ),
        "session_log_trial_count": partial(
            session_log_trial_count_multiplier, mazes_dict=mazes_dict
        ),
        "bias_W": partial(bias_W, mazes_dict=mazes_dict),
        "bias_N": partial(bias_N, mazes_dict=mazes_dict),
        "bias_E": partial(bias_E, mazes_dict=mazes_dict),
        "bias_S": partial(bias_S, mazes_dict=mazes_dict),
        "antiR": partial(antiR_vals, mazes_dict=mazes_dict),
        "antiL": partial(antiL_vals, mazes_dict=mazes_dict),
    }
    if "&" in feature:
        # it's an interaction feature
        feat1, feat2 = feature.split("&")
        cvs1, vs1 = vals_retriever(df, feat1, mazes_dict, prior_dict)
        cvs2, vs2 = vals_retriever(df, feat2, mazes_dict, prior_dict)
        chosen_vals = cvs1.to_numpy("float") * cvs2.to_numpy("float")
        vals = vs1.to_numpy("float") * vs2.to_numpy("float")

        column_tuples = [
            (feature, "W"),
            (feature, "S"),
            (feature, "N"),
            (feature, "E"),
        ]
        columns_index = pd.MultiIndex.from_tuples(column_tuples)
        chosen_vals = pd.Series(data=chosen_vals, index=df.index, name=feature)
        vals = pd.DataFrame(data=vals, index=df.index, columns=columns_index)

    else:
        chosen_vals, vals = feature_func_dict[feature](df)

    return (chosen_vals, vals)


def nll_func(params_array, chosen_vals, vals):
    chosen_vals_comb = (params_array * chosen_vals).sum(axis=-1)
    vals_comb = (params_array * vals).sum(axis=-1).T
    logprobs = logsoftmax_chosen_values(chosen_vals_comb, vals_comb)

    return -np.sum(logprobs)


def step_like_func(params_array, chosen_vals, vals):
    chosen_vals_comb = (params_array * chosen_vals).sum(axis=-1)
    vals_comb = (params_array * vals).sum(axis=-1).T
    logprobs = logsoftmax_chosen_values(chosen_vals_comb, vals_comb)

    return np.exp(logprobs)


def single_day_nll_func_generator(
    df,
    features_list,
    mazes_dict,
    prior_dict=None,
    standardize=False,
    feature_means=None,
    feature_stds=None,
    debug=False,
):
    chosen_vals_array = []
    vals_array = []
    chosen_vals = []
    vals = []

    single_day_out = {}

    for feature in features_list:
        cvs, vs = vals_retriever(
            df=df, feature=feature, mazes_dict=mazes_dict, prior_dict=prior_dict
        )
        chosen_vals_array.append(cvs.to_numpy("float"))
        vals_array.append(vs.to_numpy("float"))
        chosen_vals.append(cvs)
        vals.append(vs)

    chosen_vals_array = np.array(chosen_vals_array).T
    vals_array = np.array(vals_array).T

    if standardize:
        if feature_means is None:
            feature_means = np.nanmean(vals_array, axis=(0, 1))
        if feature_stds is None:
            feature_stds = np.nanstd(vals_array, axis=(0, 1))

        chosen_vals_array = chosen_vals_array - feature_means
        chosen_vals_array = chosen_vals_array / feature_stds

        vals_array = vals_array - feature_means
        vals_array = vals_array / feature_stds
    if debug:
        # we use the dataframe created by pd.concat just to have the right index and labels in place but then substitute the values by those which are used for  nll_func such that we have the exact same values, with normalization, etc. as it went into the optimization
        # breakpoint()
        chosen_vals = pd.concat(chosen_vals, axis=1)
        chosen_vals[:] = chosen_vals_array

        vals = pd.concat(vals, axis=1)
        flabel_list = vals.columns.get_level_values(0).unique()
        for i, flabel in enumerate(flabel_list):
            vals.loc[:, idx[flabel, :]] = vals_array[:, :, i].T

        chosen_vals["maze_name"] = df.maze_name.unique()[0]
        chosen_vals["maze_day"] = df.maze_day.unique()[0]
        chosen_vals["subject"] = df.subject.unique()[0]
        vals["maze_name"] = df.maze_name.unique()[0]
        vals["maze_day"] = df.maze_day.unique()[0]
        vals["subject"] = df.subject.unique()[0]

        single_day_out["chosen_vals"] = chosen_vals
        single_day_out["vals"] = vals

    single_day_out["nll_func"] = partial(
        nll_func, chosen_vals=chosen_vals_array, vals=vals_array
    )
    single_day_out["step_like_func"] = partial(
        step_like_func, chosen_vals=chosen_vals_array, vals=vals_array
    )
    return single_day_out


def nll_func_generator(
    df,
    private_features_list,  # includes both with and without interaction
    mazes_dict,
    shared_features_list=None,
    prior_dict=None,
    standardize=True,
    shared_feature_global_standardization=True,
    debug=False,
):
    """Returns a function that inputs feature coefficients and outputs NLL"""
    if shared_features_list is None:
        output = single_day_nll_func_generator(
            df=df,
            features_list=private_features_list,
            mazes_dict=mazes_dict,
            prior_dict=prior_dict,
            standardize=standardize,
            debug=debug,
        )

    else:
        output = {}
        n_days = len(df.maze_day.unique())
        assert n_days >= 9
        assert df.subject.nunique() == 1
        private_and_shared_features_list = private_features_list + shared_features_list
        n_private_features = len(private_features_list)

        nll_funcs = []
        step_like_funcs = []
        chosen_vals = []
        vals = []

        # We calculate the mean and SD for the shared features
        shared_feature_means = []
        shared_feature_stds = []
        for shared_feature in shared_features_list:
            scvs, svs = vals_retriever(df, shared_feature, mazes_dict, prior_dict)
            shared_feature_means.append(np.nanmean(svs))
            shared_feature_stds.append(np.nanstd(svs))
            # print(
            #     f"{df.maze_name.unique()[0]}\t{shared_feature}: {np.nanmean(svs, axis=0)}"
            # )
            # print(
            #     f"{df.maze_name.unique()[0]}\t{shared_feature}: {np.nanstd(svs, axis=0)}"
            # )

        for day in df.maze_day.unique():
            day_df = df[df["maze_day"] == day].copy()

            if shared_feature_global_standardization:
                private_feature_means = []
                private_feature_stds = []
                for private_feature in private_features_list:
                    pcvs, pvs = vals_retriever(
                        day_df,
                        private_feature,
                        mazes_dict,
                        prior_dict,
                    )
                    private_feature_means.append(np.nanmean(pvs))
                    private_feature_stds.append(np.nanstd(pvs))
                    # print(
                    #     f"{df.maze_name.unique()[0]}\t{private_feature}: {np.nanmean(pvs, axis=0)}"
                    # )
                    # print(
                    #     f"{df.maze_name.unique()[0]}\t{private_feature}: {np.nanstd(pvs, axis=0)}"
                    # )

                private_and_shared_feature_means = np.array(
                    private_feature_means + shared_feature_means
                )
                private_and_shared_feature_stds = np.array(
                    private_feature_stds + shared_feature_stds
                )
            else:
                private_and_shared_feature_means = None
                private_and_shared_feature_stds = None

            # print(private_and_shared_feature_means)

            single_day_output = single_day_nll_func_generator(
                df=day_df,
                features_list=private_and_shared_features_list,
                mazes_dict=mazes_dict,
                prior_dict=prior_dict,
                standardize=standardize,
                feature_means=private_and_shared_feature_means,
                feature_stds=private_and_shared_feature_stds,
                debug=debug,
            )
            nll_func = single_day_output["nll_func"]
            step_like_func = single_day_output["step_like_func"]
            if debug:
                chosen_vals.append(single_day_output["chosen_vals"])
                vals.append(single_day_output["vals"])

            nll_funcs.append(nll_func)
            step_like_funcs.append(step_like_func)

        def shared_nll_func(params_array):
            # the private params are sorted as (day1_feature1, day1_feature2, day1_feature3, day2_feature1, day2_feature2, day2_feature3, day3_feature1, ...)
            private_params_array = params_array[0 : n_private_features * n_days]
            # the last params in the array are the ones shared across days
            shared_params_array = params_array[n_private_features * n_days :]

            global_nll = 0.0

            for i in range(n_days):
                first_day_param_index = i * n_private_features
                day_private_params = private_params_array[
                    first_day_param_index : first_day_param_index + n_private_features
                ]
                day_private_and_shared_params = np.concatenate(
                    (day_private_params, shared_params_array)
                )
                global_nll += nll_funcs[i](day_private_and_shared_params)
            return global_nll

        def all_days_step_like_func(params_array):
            # the private params are sorted as (day1_feature1, day1_feature2, day1_feature3, day2_feature1, day2_feature2, day2_feature3, day3_feature1, ...)
            private_params_array = params_array[0 : n_private_features * n_days]
            # the last params in the array are the ones shared across days
            shared_params_array = params_array[n_private_features * n_days :]

            all_days_step_like = []
            for i in range(n_days):
                first_day_param_index = i * n_private_features
                day_private_params = private_params_array[
                    first_day_param_index : first_day_param_index + n_private_features
                ]
                day_private_and_shared_params = np.concatenate(
                    (day_private_params, shared_params_array)
                )
                all_days_step_like.append(
                    step_like_funcs[i](day_private_and_shared_params).ravel()
                )
            return np.concatenate(all_days_step_like)

        output["shared_nll_func"], output["nll_funcs"], output["step_like_func"] = (
            shared_nll_func,
            nll_funcs,
            all_days_step_like_func,
        )
        if debug:
            (
                output["chosen_vals"],
                output["vals"],
            ) = (chosen_vals, vals)
    return output
