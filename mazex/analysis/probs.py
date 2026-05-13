"""
Analysis methods that don't rely on fitting should come here
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import random

from mazex.exceptions import *
from mazex.analysis.iterators import (
    first_exit,
    first_exit_over_steps,
    first_edge_crossing,
    orthogonal_planning_vectornav,
)
from mazex.agents import HeuristicAgent, OmniscientAgent
import mazex.utils as utils
import mazex as mx
import networkx as nx
import numpy as np
import pylab as pl
import scipy.optimize

import seaborn as sns

import shapely.geometry as sg
import shapely.affinity as sa
import shapely.ops as so
import descartes
from tqdm import tqdm
import matplotlib as mpl


_ARROWS_PLOT2_PARALLEL_STATE = {}


def _arrows_plot2_simulation_coords(session, spatial_distance, first_exit_only):
    if first_exit_only:
        return prob_plan_prob_vec(first_exit(session), spatial_distance=spatial_distance)
    return prob_plan_prob_vec(
        session.steps(),
        spatial_distance=spatial_distance,
        discard_backtracked=False,
        only_forward=False,
    )


def _arrows_plot2_simulate_beta(args):
    beta, max_total_steps, seed = args
    maze = _ARROWS_PLOT2_PARALLEL_STATE["maze"]
    goalz = _ARROWS_PLOT2_PARALLEL_STATE["goalz"]
    spatial_distance = _ARROWS_PLOT2_PARALLEL_STATE["spatial_distance"]
    first_exit_only = _ARROWS_PLOT2_PARALLEL_STATE["first_exit_only"]

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

    hcoords = []
    mcoords = []

    for goals in goalz:
        hagent = HeuristicAgent(
            beta=beta, xi=5, penalty_type="exponential", heading="^", loc=(0, 0)
        )
        hsession = mx.SimSessionGivenGoals(
            hagent, maze, goals, max_total_steps=max_total_steps
        )
        hsession.run(progress_bar=False)
        hcoords.append(
            _arrows_plot2_simulation_coords(
                hsession, spatial_distance, first_exit_only
            )
        )

        magent = OmniscientAgent(maze, beta=beta, xi=5, heading="^", loc=(0, 0))
        msession = mx.SimSessionGivenGoals(
            magent, maze, goals, max_total_steps=max_total_steps
        )
        msession.run(progress_bar=False)
        mcoords.append(
            _arrows_plot2_simulation_coords(
                msession, spatial_distance, first_exit_only
            )
        )

    return beta, np.array(hcoords), np.array(mcoords)


def _arrows_plot2_default_n_jobs():
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _arrows_plot2_process_pool(n_jobs):
    return ProcessPoolExecutor(
        max_workers=n_jobs, mp_context=mp.get_context("fork")
    )


def shapely_ellipse(center, major_axis, minor_axis, rotation=0.0):
    el = sg.Point(0.0, 0.0).buffer(1.0)  # a unit circle
    el = sa.scale(el, major_axis / 2, minor_axis / 2)  # make it elliptic
    el = sa.rotate(el, rotation)
    el = sa.translate(el, *center)
    return el


def prob(step, null="random"):
    if null == "random":
        # now we need to count how many edges correspond to the chosen
        # direction, but only in case there are more edges available than
        # directions in the distances space
        edges_per_direction = np.zeros(4)
        for n in step.available_nodes:
            edges_per_direction[
                utils.to_dist_direction(n, step.loc, step.trial.maze, step.trial.goal)
            ] += 1

        return edges_per_direction / len(step.available_nodes)

    else:
        raise NotSupported


def arrows_plot(sessions, maze, controls="greedy", colors="dark"):
    # TODO: let user input control msession and hsession as function inputs

    sns.set(context="notebook", font_scale=1.3, rc={"axes.titleweight": "bold"})

    # Params
    arrow_width = 0.004

    if colors == "dark":
        data_color = sns.color_palette("Dark2")[3]
        planning_color = sns.color_palette("Dark2")[0]
        heuristic_color = sns.color_palette("Dark2")[1]

    elif colors == "light":
        data_color = "#E78CC4"
        planning_color = "#3CCF97"
        heuristic_color = "#F97F42"

    # Analyze and plot data
    pl.figure()
    indices = []

    if controls == "greedy":
        plot_data = True
    elif controls == "stochastic":
        plot_data = False

    for session in sessions:
        dx_dy, nums = prob_m_vs_prob_g(
            first_edge_crossing(session),
            geometric_distance="euclidian",
            plot=plot_data,
            # geometric_distance='euclidian', plot=plot_data, hold=True,
            facecolor=data_color,
            width=arrow_width,
            head_width=9 * arrow_width,
        )

        indices.append(dx_dy)

    indices = np.array(indices)
    dx, dy = np.mean(indices, axis=0)
    dx_err, dy_err = np.std(indices, axis=0)

    # Generate controls
    if controls == "greedy":
        goals = list(maze.randomgoals(10000))

        hagent = HeuristicAgent(
            beta=100, xi=3, penalty_type="exponential", heading="^", loc=(0, 0)
        )

        hsession = mx.SimSessionGivenGoals(hagent, maze, goals)
        hsession.run()

        magent = OmniscientAgent(maze, beta=100, xi=3, heading="^", loc=(0, 0))

        msession = mx.SimSessionGivenGoals(magent, maze, goals)
        msession.run()

        # Analyze controls
        prob_m_vs_prob_g(
            first_edge_crossing(msession),
            geometric_distance="euclidian",
            plot=True,
            # hold=True,
            label="Planning agent",
            facecolor=planning_color,
            width=arrow_width,
            head_width=9 * arrow_width,
        )

        prob_m_vs_prob_g(
            first_edge_crossing(hsession),
            geometric_distance="euclidian",
            plot=True,
            # hold=True,
            label="Heuristic agent",
            facecolor=heuristic_color,
            width=arrow_width,
            head_width=9 * arrow_width,
        )

    elif controls == "stochastic":
        betas = np.logspace(-5, 4, 30, base=np.e)
        # betas = np.logspace(-3, 4, 20, base=np.e)
        # betas = np.logspace(-3, 2, 120, base=np.e)

        n_goals_sessions = []
        n_steps_sessions = []
        # goals_base = []
        for session in sessions:
            n_goals_sessions.append(len(session.trials))
            n_steps_sessions.append(session.count_total_steps())
            # sessions_goals = []
            # for trial in session.trials:
            #    session_goals.append trial.goal
            # goals_base.append(session_goals)

        n_goals = 5 * int(np.average(n_goals_sessions))
        print(n_goals)
        max_total_steps = np.average(n_steps_sessions)
        print(" ", max_total_steps)
        # n_goals = 1000
        # ctrl_size = 20
        ctrl_size = 50

        goalz = [list(maze.randomgoals(n_goals)) for x in range(ctrl_size)]

        # for the confidence region
        hcenters = []
        mcenters = []

        hlambdas = []
        mlambdas = []
        hvecs = []
        mvecs = []

        hellipses = []
        mellipses = []
        for beta in tqdm(betas, desc="betas"):
            # print(beta)
            n_sims = 0
            hsims = []
            msims = []

            hcoords = []
            mcoords = []

            for goals in tqdm(goalz, desc="goals"):
                # print(beta)
                # TODO: adapt xi per beta
                hagent = HeuristicAgent(
                    beta=beta, xi=5, penalty_type="exponential", heading="^", loc=(0, 0)
                )
                hsession = mx.SimSessionGivenGoals(
                    hagent, maze, goals, max_total_steps=max_total_steps
                )
                hsession.run()
                hsims.append(hsession)
                hcoords.append(
                    prob_m_vs_prob_g(
                        first_edge_crossing(hsession),
                        geometric_distance="euclidian",
                        plot=False,
                    )[0].tolist()
                )

                magent = OmniscientAgent(maze, beta=beta, xi=5, heading="^", loc=(0, 0))

                msession = mx.SimSessionGivenGoals(
                    magent, maze, goals, max_total_steps=max_total_steps
                )
                msession.run()
                msims.append(msession)
                mcoords.append(
                    prob_m_vs_prob_g(
                        first_edge_crossing(msession),
                        geometric_distance="euclidian",
                        plot=False,
                    )[0].tolist()
                )

            hcoords = np.array(hcoords)
            # h_indices = np.argsort(hcoords[:,0])
            mcoords = np.array(mcoords)

            # pl.plot(hcoords[:,0], hcoords[:,1], linewidth=0, marker='o',
            #        color=heuristic_color)
            # pl.plot(mcoords[:,0], mcoords[:,1], linewidth=0, marker='o',
            #        color=planning_color)

            hcov = np.cov(hcoords[:, 0], hcoords[:, 1])
            hlambda, hv = np.linalg.eig(hcov)
            hlambda = np.sqrt(hlambda)
            hlambdas.append(hlambda)
            hvecs.append(hv)
            hcenters.append(np.mean(hcoords, axis=0))

            mcov = np.cov(mcoords[:, 0], mcoords[:, 1])
            mlambda, mv = np.linalg.eig(mcov)
            mlambda = np.sqrt(mlambda)
            mlambdas.append(mlambda)
            mvecs.append(mv)
            mcenters.append(np.mean(mcoords, axis=0))

            # borrowing code from http://stackoverflow.com/questions/34702067/how-can-i-get-the-outline-of-the-overlapping-patches-in-matplotlib-without-maths
            # useful codes:
            # https://gist.github.com/urschrei/6436526
            # http://stackoverflow.com/questions/28732845/combine-two-pyplot-patches-for-legend
            j = 2
            hell_shapely = shapely_ellipse(
                np.mean(hcoords, axis=0),
                hlambda[0] * j * 2,
                hlambda[1] * j * 2,
                np.rad2deg(np.arccos(hv[0, 0])),
            )
            hellipses.append(hell_shapely)

            mell_shapely = shapely_ellipse(
                np.mean(mcoords, axis=0),
                (mlambda[0] * j * 2) + 1e-10,
                (mlambda[1] * j * 2) + 1e-10,
                np.rad2deg(np.arccos(mv[0, 0])),
            )
            mellipses.append(mell_shapely)

        sns.set_palette("Dark2", 8)
        xs = np.array([indices[:, 0]])
        ys = np.array([indices[:, 1]])

        pmarkers = pl.plot(xs, ys, linewidth=0, marker="o", label="Data")
        p3 = pmarkers[-1]

        all_hellipses = so.cascaded_union(hellipses)
        if type(all_hellipses) is sg.multipolygon.MultiPolygon:
            hcollection = [
                descartes.PolygonPatch(
                    h, ec="none", fc=heuristic_color, alpha=0.5, zorder=0
                )
                for h in all_hellipses
            ]
            ax = pl.gca()
            ax.add_collection(
                mpl.collections.PatchCollection(hcollection, match_original=True)
            )
        else:
            hell = descartes.PolygonPatch(
                all_hellipses, ec="none", fc=heuristic_color, alpha=0.5, zorder=0
            )
            ax = pl.gca()
            ax.add_artist(hell)

        hcenters = [c.tolist() for c in hcenters]
        hcenters.insert(0, (0, 0))
        hcenters = np.array(hcenters)
        (p2_,) = pl.plot(
            hcenters[:, 0],
            hcenters[:, 1],
            color=heuristic_color,
            linestyle="--",
            zorder=1,
        )

        all_mellipses = so.cascaded_union(mellipses)
        if type(all_mellipses) is sg.multipolygon.MultiPolygon:
            mcollection = [
                descartes.PolygonPatch(
                    m, ec="none", fc=planning_color, alpha=0.5, zorder=0
                )
                for m in all_mellipses
            ]
            ax = pl.gca()
            ax.add_collection(
                mpl.collections.PatchCollection(mcollection, match_original=True)
            )
        else:
            mell = descartes.PolygonPatch(
                all_mellipses, ec="none", fc=planning_color, alpha=0.5, zorder=0
            )
            ax = pl.gca()
            ax.add_artist(mell)

        mcenters = [c.tolist() for c in mcenters]
        mcenters.insert(0, (0, 0))
        mcenters = np.array(mcenters)
        (p1_,) = pl.plot(
            mcenters[:, 0],
            mcenters[:, 1],
            color=planning_color,
            linestyle="--",
            zorder=1,
        )

        """
        #hsegments_vectors = []
        #for i in np.arange(len(hcenters))[1:-1]:
        #    hsegments_vectors.append(hcenters[i+1] - hcenters[i])
        #hsegments_vectors.insert(0, hsegments_vectors[0])
        #hsegments_vectors.insert(-1, hsegments_vectors[-1])
        origin= np.array([0,0])
        hline = hcenters[-1] - origin

        hvecs_selected_i = []
        for i, vec in enumerate(hvecs):
            angles = []
# The eigenvectors are the columns of the matrix. If we transpose then
# we can simply iterate like this
            vec = vec.transpose()
            for v in vec:
                angles.append(
                    np.arccos(np.dot(v, hline) / (
                        np.linalg.norm(v)*np.linalg.norm(
                            hline))))
            hvecs_selected_i.append(np.argmax(angles))
            import ipdb; ipdb.set_trace()

        sigma = 3
        h_boundaries_plus = [hcenters[i]+sigma*hlambdas[i][hvecs_selected_i[i]]*hvecs[i][
            hvecs_selected_i[i]] for i in range(len(hcenters))]

        h_boundaries_minus =[hcenters[i]-sigma*hlambdas[i][hvecs_selected_i[i]]*hvecs[i][
            hvecs_selected_i[i]] for i in range(len(hcenters))]

        #pl.plot(h_boundaries_plus[:,0], h_boundaries_plus[:,1], 'b')
        #pl.plot(h_boundaries_minus[:,0], h_boundaries_plus[:,1], 'k')
        for x, y in h_boundaries_plus:
            pl.plot(x,y, 'b', linewidth=0, marker='o')

        for x, y in h_boundaries_minus:
            pl.plot(x,y, 'k', linewidth=0, marker='o')

        pl.show()
        import ipdb; ipdb.set_trace()

        #from shapely.geometry import Polygon
        #from shapely.ops import cascaded_union
        #heshapely = [Polygon(ellipse.get_path().vertices) for ellipse in hellipses]

        #union = cascaded_union(heshapely)

        #from descartes import PolygonPatch
        #union_patch = PolygonPatch(union, fc='b', ec='b', alpha=0.5)
        #ax = pl.gca()
        #ax.add_patch(union_patch)
        """

    # Generate objects so legend can be autogenerated
    p1 = pl.Rectangle((0, 0), 1, 1, fc=planning_color)
    p2 = pl.Rectangle((0, 0), 1, 1, fc=heuristic_color)
    if controls == "greedy":
        p3 = pl.Rectangle((0, 0), 1, 1, fc=data_color)

    # Legend and axis labels
    if controls == "greedy":
        pl.legend(
            [p1, p2, p3],
            ["Planning (simulation)", "Vector nav. (simulation)", "Data"],
            loc=2,
        )
    if controls == "stochastic":
        pl.legend(
            [p1, p2, p3], ["Planning (95% CR)", "Vector nav. (95% CR)", "Data"], loc=2
        )

    pl.xlabel("Planning index", weight="bold")
    pl.ylabel("Vector navigation index", weight="bold")
    pl.title("Planning vs vector navigation choice index")
    pl.gca().set_aspect("equal")
    pl.xlim(0, 1.02)
    pl.ylim(0, 1.02)


def arrows_plot2(
    sessions,
    maze,
    spatial_distance="euclidian",
    first_exit_only=True,
    colors="dark",
    n_jobs=None,
    ctrl_size=100,
    show_progress=True,
):
    # TODO: let user input control msession and hsession as function inputs

    sns.set(context="notebook", font_scale=1.3, rc={"axes.titleweight": "bold"})

    # Params
    if colors == "dark":
        data_color = sns.color_palette("Dark2")[3]
        planning_color = sns.color_palette("Dark2")[0]
        heuristic_color = sns.color_palette("Dark2")[1]

    elif colors == "light":
        data_color = "#E78CC4"
        planning_color = "#3CCF97"
        heuristic_color = "#F97F42"

    # Analyze and plot data
    pl.figure()
    indices = []

    n_jobs = _arrows_plot2_default_n_jobs() if n_jobs is None else n_jobs
    n_jobs = max(1, int(n_jobs))

    for session in sessions:
        if first_exit_only:
            dx_dy = prob_plan_prob_vec(
                first_exit(session), spatial_distance=spatial_distance
            )
        else:
            dx_dy = prob_plan_prob_vec(
                session.steps(),
                spatial_distance=spatial_distance,
                discard_backtracked=False,
                only_forward=False,
            )

        indices.append(dx_dy)

    indices = np.array(indices)
    dx, dy = np.mean(indices, axis=0)
    dx_err, dy_err = np.std(indices, axis=0)

    # Generate controls
    betas = np.logspace(-2, 3.5, 40, base=np.e)
    # betas = np.logspace(-3, 4, 20, base=np.e)
    # betas = np.logspace(-3, 2, 120, base=np.e)

    n_goals_sessions = []
    n_steps_sessions = []
    # goals_base = []
    for session in sessions:
        n_goals_sessions.append(len(session.trials))
        n_steps_sessions.append(session.count_total_steps())
        # sessions_goals = []
        # for trial in session.trials:
        #    session_goals.append trial.goal
        # goals_base.append(session_goals)

    n_goals = 5 * int(np.average(n_goals_sessions))
    print(n_goals)
    max_total_steps = np.average(n_steps_sessions)
    print(" ", max_total_steps)
    # n_goals = 1000
    # ctrl_size = 20
    goalz = [list(maze.randomgoals(n_goals)) for x in range(ctrl_size)]

    # for the confidence region
    hcenters = []
    mcenters = []

    hlambdas = []
    mlambdas = []
    hvecs = []
    mvecs = []

    hellipses = []
    mellipses = []
    seeds = np.random.SeedSequence().generate_state(len(betas), dtype=np.uint32)
    simulation_args = [
        (beta, max_total_steps, int(seed))
        for beta, seed in zip(betas, seeds)
    ]
    _ARROWS_PLOT2_PARALLEL_STATE.clear()
    _ARROWS_PLOT2_PARALLEL_STATE.update(
        {
            "maze": maze,
            "goalz": goalz,
            "spatial_distance": spatial_distance,
            "first_exit_only": first_exit_only,
        }
    )

    if n_jobs == 1:
        simulation_results = map(_arrows_plot2_simulate_beta, simulation_args)
    else:
        pool = _arrows_plot2_process_pool(n_jobs)
        simulation_results = pool.map(_arrows_plot2_simulate_beta, simulation_args)

    if show_progress:
        simulation_results = tqdm(
            simulation_results, total=len(simulation_args), desc="betas"
        )

    try:
        for beta, hcoords, mcoords in simulation_results:
            # h_indices = np.argsort(hcoords[:,0])

            # pl.plot(hcoords[:,0], hcoords[:,1], linewidth=0, marker='o',
            #        color=heuristic_color)
            # pl.plot(mcoords[:,0], mcoords[:,1], linewidth=0, marker='o',
            #        color=planning_color)

            hcov = np.cov(hcoords[:, 0], hcoords[:, 1])
            hlambda, hv = np.linalg.eig(hcov)
            hlambda = np.sqrt(hlambda)
            hlambdas.append(hlambda)
            hvecs.append(hv)
            hcenters.append(np.mean(hcoords, axis=0))

            mcov = np.cov(mcoords[:, 0], mcoords[:, 1])
            mlambda, mv = np.linalg.eig(mcov)
            mlambda = np.sqrt(mlambda)
            mlambdas.append(mlambda)
            mvecs.append(mv)
            mcenters.append(np.mean(mcoords, axis=0))

            # borrowing code from http://stackoverflow.com/questions/34702067/how-can-i-get-the-outline-of-the-overlapping-patches-in-matplotlib-without-maths
            # useful codes:
            # https://gist.github.com/urschrei/6436526
            # http://stackoverflow.com/questions/28732845/combine-two-pyplot-patches-for-legend
            j = 2
            hell_shapely = shapely_ellipse(
                np.mean(hcoords, axis=0),
                hlambda[0] * j * 2,
                hlambda[1] * j * 2,
                np.rad2deg(np.arccos(hv[0, 0])),
            )
            hellipses.append(hell_shapely)

            mell_shapely = shapely_ellipse(
                np.mean(mcoords, axis=0),
                (mlambda[0] * j * 2) + 1e-10,
                (mlambda[1] * j * 2) + 1e-10,
                np.rad2deg(np.arccos(mv[0, 0])),
            )
            mellipses.append(mell_shapely)
    finally:
        if n_jobs != 1:
            pool.shutdown()
        _ARROWS_PLOT2_PARALLEL_STATE.clear()

    sns.set_palette("Dark2", 8)
    xs = np.array([indices[:, 0]])
    ys = np.array([indices[:, 1]])

    pmarkers = pl.plot(xs, ys, linewidth=0, marker="o", label="Data")
    p3 = pmarkers[-1]

    all_hellipses = so.cascaded_union(hellipses)
    if type(all_hellipses) is sg.multipolygon.MultiPolygon:
        hcollection = [
            descartes.PolygonPatch(
                h, ec="none", fc=heuristic_color, alpha=0.5, zorder=0
            )
            for h in all_hellipses
        ]
        ax = pl.gca()
        ax.add_collection(
            mpl.collections.PatchCollection(hcollection, match_original=True)
        )
    else:
        hell = descartes.PolygonPatch(
            all_hellipses, ec="none", fc=heuristic_color, alpha=0.5, zorder=0
        )
        ax = pl.gca()
        ax.add_artist(hell)

    hcenters = [c.tolist() for c in hcenters]
    hcenters.insert(0, (0, 0))
    hcenters = np.array(hcenters)
    (p2_,) = pl.plot(
        hcenters[:, 0], hcenters[:, 1], color=heuristic_color, linestyle="--", zorder=1
    )

    all_mellipses = so.cascaded_union(mellipses)
    if type(all_mellipses) is sg.multipolygon.MultiPolygon:
        mcollection = [
            descartes.PolygonPatch(m, ec="none", fc=planning_color, alpha=0.5, zorder=0)
            for m in all_mellipses
        ]
        ax = pl.gca()
        ax.add_collection(
            mpl.collections.PatchCollection(mcollection, match_original=True)
        )
    else:
        mell = descartes.PolygonPatch(
            all_mellipses, ec="none", fc=planning_color, alpha=0.5, zorder=0
        )
        ax = pl.gca()
        ax.add_artist(mell)

    mcenters = [c.tolist() for c in mcenters]
    mcenters.insert(0, (0, 0))
    mcenters = np.array(mcenters)
    (p1_,) = pl.plot(
        mcenters[:, 0], mcenters[:, 1], color=planning_color, linestyle="--", zorder=1
    )

    # Generate objects so legend can be autogenerated
    p1 = pl.Rectangle((0, 0), 1, 1, fc=planning_color)
    p2 = pl.Rectangle((0, 0), 1, 1, fc=heuristic_color)

    # Legend and axis labels
    pl.legend(
        [p1, p2, p3], ["Planning (95% CR)", "Vector nav. (95% CR)", "Data"], loc=2
    )

    pl.xlabel("Planning index", weight="bold")
    pl.ylabel("Vector navigation index", weight="bold")
    pl.title("Planning vs vector navigation choice index")
    pl.gca().set_aspect("equal")
    pl.xlim(0, 1.02)
    pl.ylim(0, 1.02)


def orthogonal_arrows_plot(sessions, maze, colors="dark"):
    # TODO: let user input control msession and hsession as function inputs

    sns.set(context="notebook", font_scale=1.3, rc={"axes.titleweight": "bold"})

    # Params
    if colors == "dark":
        data_color = sns.color_palette("Dark2")[3]
        planning_color = sns.color_palette("Dark2")[0]
        heuristic_color = sns.color_palette("Dark2")[1]

    elif colors == "light":
        data_color = "#E78CC4"
        planning_color = "#3CCF97"
        heuristic_color = "#F97F42"

    # Analyze and plot data
    indices = []

    for session in sessions:
        steps_generator = orthogonal_planning_vectornav(first_exit(session), maze)

        dx_dy, nums = frac_m(
            steps_generator, geometric_distance="euclidian", plot=False
        )

        indices.append(dx_dy)

    indices = np.array(indices)
    dx, dy = np.mean(indices, axis=0)
    dx_err, dy_err = np.std(indices, axis=0)

    pl.xlabel("Planning index", weight="bold")
    pl.ylabel("Vector navigation index", weight="bold")
    pl.title("Planning vs vector navigation choice index")
    pl.gca().set_aspect("equal")
    pl.xlim(0, 1.02)
    pl.ylim(0, 1.02)

    return indices


def avg_vector_difference(
    session,
    geometric_distance="euclidian",
    exclusion_criterium="node",
    plot=False,
    **kwargs
):
    chosen_vectors = []
    available_mean_vectors = []
    sum_vector = np.zeros(2)
    step_count = 0

    if exclusion_criterium == "node":
        step_iterator = first_exit(session)

    elif exclusion_criterium == "edge":
        step_iterator = first_edge_crossing(session)

    if geometric_distance == "manhattan":
        unit_vectors = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])

    for s in step_iterator:
        if s.previous:
            forward_nodes = [a for a in s.available_nodes if a != s.previous.loc]
        else:
            forward_nodes = []

        if len(forward_nodes) > 1:
            step_count += 1

            if geometric_distance == "manhattan":
                available = np.array([np.nan] * 4)
                for a in forward_nodes:
                    available[
                        utils.to_dist_direction(a, s.loc, session.maze, s.trial.goal)
                    ] = np.nansum(
                        (
                            available[
                                utils.to_dist_direction(
                                    a, s.loc, session.maze, s.trial.goal
                                )
                            ],
                            1,
                        )
                    )

                available_vectors = available[..., np.newaxis] * unit_vectors
                chosen_vector = unit_vectors[
                    utils.to_dist_direction(
                        s.subsequent.loc, s.loc, session.maze, s.trial.goal
                    )
                ]

            elif geometric_distance == "euclidian":
                current_geodesic_dist = session.maze.distances_dictionary[s.loc][
                    s.trial.goal
                ]
                current_euclidian_dist = utils.euclidian_distance(s.loc, s.trial.goal)
                available_vectors = []
                for a in forward_nodes:
                    choice_geodesic_dist = session.maze.distances_dictionary[a][
                        s.trial.goal
                    ]
                    choice_euclidian_dist = utils.euclidian_distance(a, s.trial.goal)

                    available_vectors.append(
                        (
                            choice_geodesic_dist - current_geodesic_dist,
                            choice_euclidian_dist - current_euclidian_dist,
                        )
                    )
                available_vectors = np.array(available_vectors)

                # we compute the chosen vector
                chosen_geodesic_dist = session.maze.distances_dictionary[
                    s.subsequent.loc
                ][s.trial.goal]
                chosen_euclidian_dist = utils.euclidian_distance(
                    s.subsequent.loc, s.trial.goal
                )

                d_geodesic_dist = chosen_geodesic_dist - current_geodesic_dist
                d_euclidian_dist = chosen_euclidian_dist - current_euclidian_dist

                chosen_vector = np.array([d_geodesic_dist, d_euclidian_dist])

            available_mean_vector = np.nanmean(available_vectors, axis=0)
            available_mean_vectors.append(list(available_mean_vector))

            chosen_vectors.append(list(chosen_vector))

            sum_vector += chosen_vector - available_mean_vector

    normed_sum_vector = sum_vector / step_count
    if plot:
        pl.arrow(0, 0, *normed_sum_vector, **kwargs)
        # we compute the right xlim and ylim
        lower_bound = min(np.floor(normed_sum_vector.min()), -1)
        upper_bound = max(np.ceil(normed_sum_vector.max()), 1)

        pl.ylim(lower_bound, upper_bound)
        pl.xlim(lower_bound, upper_bound)
        ax = pl.gca()
        ax.set_aspect(1, adjustable="box")
        pl.grid(True)
        pl.tight_layout()

        # return chosen_vectors, available_mean_vectors, sum_vector/step_count
    return normed_sum_vector


def avg_fraction(session, exclusion_criterium="node"):
    available = np.zeros(4)
    chosen = np.zeros(4)

    if exclusion_criterium == "node":
        step_iterator = first_exit(session)

    elif exclusion_criterium == "edge":
        step_iterator = first_edge_crossing(session)

    for s in step_iterator:
        for a in s.available_nodes:
            available[
                utils.to_dist_direction(a, s.loc, session.maze, s.trial.goal)
            ] += 1
        chosen[
            utils.to_dist_direction(s.subsequent.loc, s.loc, session.maze, s.trial.goal)
        ] += 1
    return chosen, available


def fraction_dir_vector(session, exclusion_criterium="node"):
    chosen, available = avg_fraction(session, exclusion_criterium)
    fraction = chosen / available

    unit_vectors = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
    fraction_vectors = fraction[..., np.newaxis] * unit_vectors
    fraction_vector = np.sum(fraction_vectors, axis=0)

    tan_vector = fraction_vector[1] / fraction_vector[0]
    angle = np.arctan(tan_vector)

    return fraction_vector, angle


def frac_m(steps, geometric_distance="euclidian", plot=True, hold=None, **kwargs):
    """Calculate the average change in Geodesic and Manhattan distance on the
    trials when the agent had both the option to reduce or increase either
    distance."""
    # [delta_Geodesic_distance, delta_Manhattan_distance]
    unit_vectors = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])

    # _M = Manhattan / Euclidian distance
    available_M = 0

    count_vector = 0
    # _G = Geodesic = Maze = Shortest path distance
    available_G = 0

    count_planning = 0

    if geometric_distance == "euclidian":
        to_dist_direction = utils.to_dist_direction_euclidian

    elif geometric_distance == "manhattan":
        to_dist_direction = utils.to_dist_direction

    for s in steps:
        if s.previous:
            # we start by removing the node the agent just came from, from the
            #  list of available_nodes. we call that forward_nodes
            forward_nodes = [a for a in s.available_nodes if a != s.previous.loc]
        else:
            forward_nodes = s.available_nodes

        # TODO: Exclude the trials when the animal actually decided to go
        # backward!
        if len(forward_nodes) > 1:  # and s.subsequent.loc in forward_nodes:
            # we change from physical coordinates of the forward nodes to
            #  changes in the geodesical and physical distance
            directions = np.array(
                [
                    unit_vectors[
                        to_dist_direction(a, s.loc, s.trial.maze, s.trial.goal)
                    ]
                    for a in forward_nodes
                ]
            )

            # We check if there were options along the Geodesical axis
            # import ipdb; ipdb.set_trace() # BREAKPOINT
            #            if np.any(directions[:,0] == -1) and np.any(directions[:,0] == 1):
            #                available_G += 1 # a simple counter
            #                d_G, d_M = unit_vectors[to_dist_direction(s.subsequent.loc, s.loc,
            #                        s.trial.maze, s.trial.goal)]
            #
            #                #import ipdb; ipdb.set_trace()
            #                if d_M == -1:
            #                    count_vector += 1
            #
            #                if d_G == -1:
            #                    count_planning += 1
            #
            #            if np.any(directions[:,1] == -1) and np.any(directions[:,1] == 1):
            #                available_M += 1 # a simple counter
            #                d_G, d_M = unit_vectors[to_dist_direction(s.subsequent.loc, s.loc,
            #                        s.trial.maze, s.trial.goal)]
            #
            #                if d_M == -1:
            #                    count_vector += 1
            #
            #                if d_G == -1:
            #                    count_planning += 1

            if np.any(directions[:, 0] == -1) and np.any(directions[:, 0] == 1):
                available_G += 1  # a simple counter
                # import ipdb; ipdb.set_trace()

            if np.any(directions[:, 1] == -1) and np.any(directions[:, 1] == 1):
                available_M += 1  # a simple counter

            d_G, d_M = unit_vectors[
                to_dist_direction(s.subsequent.loc, s.loc, s.trial.maze, s.trial.goal)
            ]

            if d_M == -1:
                count_vector += 1

            if d_G == -1:
                count_planning += 1
    if plot:
        # we need to save the matplotlib artist for the arrow object for legend
        #  building purposes
        arrow_artist = pl.arrow(
            0,
            0,
            S_d_G / S_best_forward_G,
            S_d_M / S_best_forward_M,
            # hold=hold,
            edgecolor="none",
            length_includes_head=True,
            **kwargs
        )

        if kwargs:
            if "label" in kwargs:
                # now we create the legend in this complex way because arrow
                #  does not support automatic legend building
                from mazex.drawing import make_legend_arrow
                import matplotlib.patches as mpatches
                from matplotlib.legend_handler import HandlerPatch

                pl.legend(
                    [arrow_artist],
                    [kwargs["label"]],
                    handler_map={
                        mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow)
                    },
                )
                pl.gca().set_aspect("equal")
                pl.xlim(0, 1.1)
                pl.ylim(0, 1.1)
                pl.tight_layout()

    return np.array([count_planning, count_vector]), (available_G, available_M)


def prob_m_vs_prob_g(
    steps,
    geometric_distance="euclidian",
    plot=True,
    **kwargs
    # steps, geometric_distance="euclidian", plot=True, hold=None, **kwargs
):
    """Calculate the average change in Geodesic and Manhattan distance on the
    trials when the agent had both the option to reduce or increase either
    distance."""
    # [delta_Geodesic_distance, delta_Manhattan_distance]
    unit_vectors = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])

    # _M = Manhattan / Euclidian distance
    S_d_M = 0
    S_best_forward_M = 0
    available_M = 0
    # _G = Geodesic = Maze = Shortest path distance
    S_d_G = 0
    S_best_forward_G = 0
    available_G = 0

    if geometric_distance == "euclidian":
        to_dist_direction = utils.to_dist_direction_euclidian

    elif geometric_distance == "manhattan":
        to_dist_direction = utils.to_dist_direction

    for s in steps:
        if s.previous:
            # we start by removing the node the agent just came from, from the
            #  list of available_nodes. we call that forward_nodes
            forward_nodes = [a for a in s.available_nodes if a != s.previous.loc]
        else:
            forward_nodes = s.available_nodes

        # TODO: Exclude the trials when the animal actually decided to go
        # backward!
        if (
            len(forward_nodes) > 1 and s.loc != s.trial.goal
        ):  # and s.subsequent.loc in forward_nodes:
            # we change from physical coordinates of the forward nodes to
            #  changes in the geodesical and physical distance
            directions = np.array(
                [
                    unit_vectors[
                        to_dist_direction(a, s.loc, s.trial.maze, s.trial.goal)
                    ]
                    for a in forward_nodes
                ]
            )

            # We check if there were options along the Geodesical axis
            # import ipdb; ipdb.set_trace() # BREAKPOINT
            if np.any(directions[:, 0] == -1) and np.any(directions[:, 0] == 1):
                available_G += 1  # a simple counter
                d_G, d_M = unit_vectors[
                    to_dist_direction(
                        s.subsequent.loc, s.loc, s.trial.maze, s.trial.goal
                    )
                ]

                #                i_dir_available_nodes = [to_dist_direction(n, s.loc,
                #                    s.trial.maze, s.trial.goal) for n in s.available_nodes]
                i_dir_available_nodes = [
                    to_dist_direction(n, s.loc, s.trial.maze, s.trial.goal)
                    for n in forward_nodes
                ]
                d_GM_available_nodes = unit_vectors[i_dir_available_nodes]
                # expected value of d_G
                E_d_G = np.mean(d_GM_available_nodes[:, 0])
                S_best_forward_G += -1 - E_d_G

                S_d_G += d_G - E_d_G

            if np.any(directions[:, 1] == -1) and np.any(directions[:, 1] == 1):
                available_M += 1  # a simple counter
                d_G, d_M = unit_vectors[
                    to_dist_direction(
                        s.subsequent.loc, s.loc, s.trial.maze, s.trial.goal
                    )
                ]

                #                i_dir_available_nodes = [to_dist_direction(n, s.loc,
                #                    s.trial.maze, s.trial.goal) for n in s.available_nodes]
                i_dir_available_nodes = [
                    to_dist_direction(n, s.loc, s.trial.maze, s.trial.goal)
                    for n in forward_nodes
                ]

                d_GM_available_nodes = unit_vectors[i_dir_available_nodes]
                # expected value of d_G
                E_d_M = np.mean(d_GM_available_nodes[:, 1])
                S_best_forward_M += -1 - E_d_M

                S_d_M += d_M - E_d_M
        # import ipdb; ipdb.set_trace()

    if plot:
        # we need to save the matplotlib artist for the arrow object for legend
        #  building purposes
        arrow_artist = pl.arrow(
            0,
            0,
            S_d_G / S_best_forward_G,
            S_d_M / S_best_forward_M,
            # hold=hold,
            edgecolor="none",
            length_includes_head=True,
            **kwargs
        )

        if kwargs:
            if "label" in kwargs:
                # now we create the legend in this complex way because arrow
                #  does not support automatic legend building
                from mazex.drawing import make_legend_arrow
                import matplotlib.patches as mpatches
                from matplotlib.legend_handler import HandlerPatch

                pl.legend(
                    [arrow_artist],
                    [kwargs["label"]],
                    handler_map={
                        mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow)
                    },
                )
                pl.gca().set_aspect("equal")
                pl.xlim(0, 1.1)
                pl.ylim(0, 1.1)
                pl.tight_layout()

    # import ipdb; ipdb.set_trace()
    return np.array([S_d_G / S_best_forward_G, S_d_M / S_best_forward_M]), (
        available_G,
        available_M,
    )


def prob_plan_prob_vec(
    steps,
    spatial_distance="euclidian",
    subtract_random=True,
    discard_backtracked=True,
    only_forward=True,
    neither_plan_nor_nav=False,
):
    planning_opportunities = 0
    planning_choices = 0
    planning_choices_random_choice = 0

    vector_opportunities = 0
    vector_choices = 0
    vector_choices_random_choice = 0

    if neither_plan_nor_nav:
        neither_chosen = 0

    for s in mx.analysis.iterators.exclude_goal(steps):
        if s.previous:
            if discard_backtracked and s.previous.loc == s.subsequent.loc:
                # by continuing to the next iteration of the for loop we skip
                # over steps where the agent went to where it just came from
                continue
            else:
                if only_forward:
                    available_nodes = [
                        a for a in s.available_nodes if a != s.previous.loc
                    ]
                else:
                    available_nodes = [a for a in s.available_nodes]

        else:
            available_nodes = s.available_nodes

        if len(available_nodes) > 1:
            # we check if there are preferred and non-preferred options for planning
            if (
                len(
                    set(
                        [
                            s.trial.maze.distances_dictionary[s.trial.goal][f]
                            for f in available_nodes
                        ]
                    )
                )
                > 1
            ):
                planning_opportunities += 1

                n_planning_options = 0
                for f in available_nodes:
                    if (
                        s.trial.maze.distances_dictionary[s.trial.goal][f]
                        < s.trial.maze.distances_dictionary[s.trial.goal][s.loc]
                    ):
                        n_planning_options += 1
                planning_choices_random_choice += n_planning_options / len(
                    available_nodes
                )

                if (
                    s.trial.maze.distances_dictionary[s.trial.goal][s.subsequent.loc]
                    < s.trial.maze.distances_dictionary[s.trial.goal][s.loc]
                ):
                    planning_choices += 1

            if spatial_distance == "manhattan":
                available_spatial_dists = np.array(
                    [
                        mx.utils.manhattan_distance(s.trial.goal, f)
                        for f in available_nodes
                    ]
                )

                if len(set(available_spatial_dists)) > 1:
                    vector_opportunities += 1

                    n_vector_options = 0
                    for f in available_nodes:
                        if mx.utils.manhattan_distance(s.trial.goal, f) == min(
                            available_spatial_dists
                        ):
                            n_vector_options += 1
                    vector_choices_random_choice += n_vector_options / len(
                        available_nodes
                    )

                    if mx.utils.manhattan_distance(
                        s.trial.goal, s.subsequent.loc
                    ) == min(available_spatial_dists):
                        vector_choices += 1

            if spatial_distance == "euclidian":
                available_spatial_dists = np.array(
                    [
                        mx.utils.euclidian_distance(s.trial.goal, f)
                        for f in available_nodes
                    ]
                )

                if len(set(available_spatial_dists)) > 1:
                    vector_opportunities += 1

                    n_vector_options = 0
                    for f in available_nodes:
                        if mx.utils.euclidian_distance(s.trial.goal, f) == min(
                            available_spatial_dists
                        ):
                            n_vector_options += 1
                    vector_choices_random_choice += n_vector_options / len(
                        available_nodes
                    )

                    if mx.utils.euclidian_distance(
                        s.trial.goal, s.subsequent.loc
                    ) == min(available_spatial_dists):
                        vector_choices += 1

            if neither_plan_nor_nav:
                # FIXME: this option should only to be used with orthogonal_planning_vectornav iterator!
                if (
                    len(
                        list(
                            mx.analysis.iterators.orthogonal_planning_vectornav(
                                [s],
                                maze=s.trial.maze,
                                distance=spatial_distance,
                                either_plan_or_nav=True,
                            )
                        )
                    )
                    == 0
                ):
                    # In this case neither the planning nor the vector navigation option was chosen
                    neither_chosen += 1

    if subtract_random:
        return (
            (planning_choices - planning_choices_random_choice)
            / (planning_opportunities - planning_choices_random_choice),
            (vector_choices - vector_choices_random_choice)
            / (vector_opportunities - vector_choices_random_choice),
        )
    else:
        if neither_plan_nor_nav:
            if planning_opportunities != vector_opportunities:
                raise Exception(
                    "Option neither_plan_nor_nav should only be used in conjunction with iterator for orthogonal_planning_vectornav with option either_plan_or_nav=True"
                )
            else:
                return (
                    planning_choices / planning_opportunities,
                    vector_choices / vector_opportunities,
                    neither_chosen / planning_opportunities,
                )
        else:
            return (
                planning_choices / planning_opportunities,
                vector_choices / vector_opportunities,
            )


# TODO: take the common core of dist_dir_prob() nd avg_dist_dir_prob() into
#   separate function
def dist_dir_prob(session, point, first_exit_only=True):
    """Calculate the average direction of choices in distances metric for a
    single point on the distances space.

    It outputs 'normalized surprisal'

    Parameters:
    -----------
        first_exit_only : Boolean
            Consider for the analysis only the first transition out of a node
    """
    # TODO: Spit an error if the point does not exist
    # FIXME: These storage of maze and goal in a step will be deprecated
    maze = session.trial.maze

    surprisals = np.zeros(4)
    availability = np.zeros(4)

    surprisals_counter = np.zeros(4)
    availability_counter = np.zeros(4)

    for t in session.trials:
        try:
            nodes_on_point = maze.distances_to_coord[point][t.goal]
        except KeyError:  # this point doesn't exist on this trial
            continue  # ignores the remaining calculations for the current
        # iteration of this for loop

        if first_exit_only:
            step_indices = [t.path.index(n) for n in nodes_on_point if n in t.path]
        else:
            step_indices = []
            for n in nodes_on_point:
                # This list comprehension finds all occurrences of n in list
                # t.path, unlinke t.path.index() which returns only the first
                # occurrence
                # += 'appends' it to the list step_indices
                step_indices += [i for i, x in enumerate(t.path) if x == n]

        # just to save on text we will save here the parameters we need to
        # call the function to_dist_coord()

        for i in step_indices:
            step = t.steps[i]
            try:
                # will fail if it is the last in the path
                next_step = t.steps[i + 1]
            except IndexError:
                continue  # ignore current step and go to next...
            # makes sense to not break instead in case the trial was aborted
            # before goal was reached.
            dist_params = (step.loc, t.maze, t.goal)

            available_directions = []
            for a in step.available_nodes:
                available_directions.append(utils.to_dist_direction(a, *dist_params))
            # list() so I can use output as an index
            # set() gets only the unique indices in the list
            # NOTE: if in the same step two maze arms represent the same
            # direction in the distances space, that direction is counted as
            # available only once
            availability_indices = list(set(available_directions))
            # [[]] makes use of advanced indexing
            availability_counter[[availability_indices]] += 1
            availability[[availability_indices]] += surprisal(step)

            surprisals_counter[
                utils.to_dist_direction(next_step.loc, *dist_params)
            ] += 1

            surprisals[
                utils.to_dist_direction(next_step.loc, *dist_params)
            ] += surprisal(step)

    """
    # These don't really make as much sense anymore but will leave them here
    # for now
    if normalized == 'simple':
        prob /= prob.max()

    elif normalized == 'moves':
        prob /= surprisals_counter

    elif normalized == 'prob':
        prob /= prob.sum()
    """
    return surprisals / surprisals_counter, availability / availability_counter


def avg_dist_dir_prob(session, plot=False):
    """Calculate the average direction of choices in distances metric."""
    moves = np.zeros(4)
    availability = np.zeros(4)

    for t in session.trials:
        # just to save on text we will save here the parameters we need to
        # call the function to_dist_coord()
        for i in range(len(t.steps) - 1):  # len(t.steps)-1 to exclude the goal
            step = t.steps[i]
            next_step = t.steps[i + 1]
            dist_params = (step.loc, t.maze, t.goal)

            availability_indices = []
            for a in step.available_nodes:
                availability_indices.append(utils.to_dist_direction(a, *dist_params))
            availability_indices = list(set(availability_indices))
            # [[]] activates advanced indexing
            availability[[availability_indices]] += 1

            moves[utils.to_dist_direction(next_step.loc, *dist_params)] += 1
    prob = moves / availability

    if plot:
        # we will use these signs to make the arrows point in the right
        # direction
        dx_sign = (1, 1, -1, -1)
        dy_sign = (1, -1, -1, 1)
        # Will be used for the plotting: hold until the last arrow is drawn
        # and then stop the holding
        hold = (True, True, True, False)
        for i in range(len(prob)):
            # we calculate the distance on x/y axis
            proj_axis = np.sqrt(prob[i] ** 2 / 2)
            dx = dx_sign[i] * proj_axis
            dy = dy_sign[i] * proj_axis
            # pl.arrow(0, 0, dx, dy, hold[i], lw=2)
            pl.arrow(0, 0, dx, dy, lw=2)

        pl.xlim(-1.2, 1.2)
        pl.ylim(-1.2, 1.2)
        ax = pl.gca()
        ax.set_aspect("equal")
        pl.xlabel("$\Delta$Z")
        pl.ylabel("$\Delta$M")
        pl.tight_layout()

    else:
        return prob


def per_dist_avg_dir_prob(
    session, first_exit_only=True, availability="plot", plot=False
):
    """Calculate the probability of direction choice on the distances space.
    Right now the way the algorithm is the following:

       For every point in the distances space, calculate for every direction
       number of times it is available (availability) and the number of times
       it is taken (moves) and return the 'probability' as moves/availability.
    """
    # session.maze will return a list if not all trials use the same maze
    # in which case it will be iterable
    max_dists = np.array(
        [utils.max_distances(t.maze, t.goal) for t in session.trials]
    ).max(0)

    dir_surprisals_dims = max_dists + 1
    dir_surprisals_dims = dir_surprisals_dims.tolist()
    dir_surprisals_dims.append(4)
    dir_surprisals = np.zeros(dir_surprisals_dims)
    dir_availabilities = np.zeros(dir_surprisals_dims)

    x, y = np.meshgrid(range(dir_surprisals.shape[0]), range(dir_surprisals.shape[1]))
    xy = np.array([x.flatten(), y.flatten()]).transpose()

    for dist_coord in xy:
        # import pdb; pdb.set_trace()
        surpr, avail = dist_dir_prob(
            session, tuple(dist_coord), first_exit_only=first_exit_only
        )
        dir_surprisals[dist_coord[0], dist_coord[1], :] = surpr
        dir_availabilities[dist_coord[0], dist_coord[1], :] = avail

    if not plot:
        return dir_surprisals, dir_availabilities
    else:
        distances_plot(
            session.trial.maze, [t.goal for t in session.trials], plot=True, zorder=10
        )
        pl.hold(True)
        dx_sign = (1, 1, -1, -1)
        dy_sign = (1, -1, -1, 1)
        # scale factor that will make the maximum length of the arrow 0.5 when
        # projected to either x or y axis. 0.75 = np.sqrt(1**2/2), and 1 is
        # obviously the maximum...
        # FIXME

        if availability == "normalizer":
            # maximum would be 1, since the surprisals are normalized
            # this is simply the value of the projection of a vector of length
            # 1 and angle 45 degr. onto x or y axes...
            max_proj = np.sqrt(1 / 2)
        else:
            # if not normalized the maximum surprisal is 2
            max_proj = np.sqrt((2**2) / 2)

        # this is simply a scaling factor so that the arrows don't overlap on
        # the plot
        scale = 0.5 / max_proj

        for direction in range(0, 4):
            surprisals_xy_proj = (
                scale
                * np.sqrt(dir_surprisals[:, :, direction] ** 2 / 2)
                .transpose()
                .flatten()
            )
            pl.quiver(
                xy[:, 0],
                xy[:, 1],
                dx_sign[direction] * surprisals_xy_proj,
                dy_sign[direction] * surprisals_xy_proj,
                units="xy",
                scale=1,
                color="b",
                zorder=2,
            )

            if availability == "plot":
                availabilities_xy_proj = (
                    scale
                    * np.sqrt(dir_availabilities[:, :, direction] ** 2 / 2)
                    .transpose()
                    .flatten()
                )
                pl.quiver(
                    xy[:, 0],
                    xy[:, 1],
                    dx_sign[direction] * availabilities_xy_proj,
                    dy_sign[direction] * availabilities_xy_proj,
                    units="xy",
                    scale=1,
                    color="r",
                    zorder=1,
                )


def heatmap(session, **kwargs):
    plot = True
    goals = [t.step.goal for t in session.trials]

    heatarray_shape = np.array(utils.max_distances(session.trial.step.maze, goals)) + 1

    heatarray = np.zeros(heatarray_shape)

    for t in session.trials:
        for s in t.path[:-1]:
            heatarray[utils.to_dist_coord(s, t.step.maze, t.step.goal)] += 1

    maze = session.trial.step.maze
    max_distances_all = utils.max_distances(maze, goals)

    dist_shape = np.array(max_distances_all) + 1
    distances = np.zeros(dist_shape)

    for goal in goals:
        for n in maze.nodes():
            new_coord = utils.to_dist_coord(n, maze, goal)
            distances[new_coord] += 1

    if len(goals) == 1:
        # This turns every element of d into an int
        distances = distances.astype(int)

    else:
        distances = distances / len(goals)

    # it is a bit hacky but it produces the plot we want... good for now
    mesh = np.meshgrid(range(dist_shape[1]), range(dist_shape[0]))
    y, x = mesh
    x = x.flatten()
    y = y.flatten()

    if plot:
        ax = pl.gca()
        ax.set_aspect("equal")
        # multiplies the value to obtain number of pixels in the surface
        area_factor = 30 * 15 * np.pi
        pl.scatter(
            x.flatten(),
            y.flatten(),
            s=(area_factor * heatarray.flatten()) / heatarray.sum(),
            c="k",
            **kwargs
        )

        # Very ugly hack just to get some legend working
        bullet_sizes = np.linspace(0.001, np.max(heatarray / heatarray.sum()), 4)
        for b in bullet_sizes:
            pl.scatter(-1, -1, s=area_factor * b, c="k", label="%f" % b, **kwargs)
        pl.legend(loc="upper left", scatterpoints=1, title="<nodes>")

        pl.xlim([-0.5, max_distances_all[0] + 0.5])
        pl.ylim([-0.5, max_distances_all[1] + 0.5])
        pl.xticks(range(0, max_distances_all[0] + 1))
        pl.yticks(range(0, max_distances_all[1] + 1))
        pl.grid()
        ax.set_axisbelow(True)  # sends the grid to the 'bottom layer'
        pl.xlabel("Maze distance to goal")
        pl.ylabel("Manhattan distance to goal")

    else:
        return distances, x, y
