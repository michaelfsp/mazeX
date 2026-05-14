import copy
import warnings

import numpy as np
import pandas as pd
import rsync
from tqdm import tqdm
from tqdm.contrib import tenumerate

from mazex import Session, Step, Trial
from mazex.data.arduino import ArduinoDataFile
from mazex.data.bonsai import BonsaiData
from mazex.data.regionmapping import RegionMapping
from mazex.utils import (
    angle_between_node_and_heading,
    angle_between_nodes,
    euclidian_distance,
    manhattan_distance,
    maze_loc_choice_to_cardinal,
    signed_angle_between_node_and_heading,
)


def load_session(
    background_file,
    background_PNG,
    arduino_file_path,
    bonsai_metadata_file_path,
    dlc_coords_file_path,
    maze,
    date=None,
    mouse=None,
    experiment_day=None,
    maze_day=None,
    fps=60,
):
    """Build Session object with data for that mouse and experiment_day"""
    # TODO: turn region_mapping into a function argument
    region_mapping = RegionMapping(background_file, background_PNG)

    # FIXME: turn these into arguments
    min_trial_duration = 1000  # ms
    step_duration_threshold = 700000  # ms
    jitter_tolerance = 2000  # ms
    discard_whole_trials = True  # If True, then trials with long steps will not be trimmed but entirely discarded
    max_trial_duration = 10000000

    session_step_count = 0
    n_shortenings = 0
    discarded_steps_too_slow = 0
    shortened_trials = 0
    discarded_trials_has_slow_step = 0
    discarded_trials_empty = 0
    discarded_trials_neg_duration = 0
    discarded_trials_not_consumed = 0

    arduino_data = ArduinoDataFile(arduino_file_path)

    bonsai_data = BonsaiData(
        dlc_coords_file_path,
        bonsai_metadata_file_path,
        metadata_delimiter=",",
        fps=fps,
        background_file=background_file,
        background_PNG=background_PNG,
        maze=maze,
        frame_number_column=0,
        timestamps_column=1,
        sync_column=2,
        hardware_timestamps=True,
    )

    fail = False

    if len(arduino_data.syncon_event_times) == 0:
        warnings.warn(
            "Arduino sync signal missing for mouse {} on day {}".format(mouse, date)
        )
        fail = True

    elif len(bonsai_data.syncon_event_times) == 0:
        warnings.warn(
            "Bonsai video sync signal missing for mouse {} on day {}".format(
                mouse, date
            )
        )
        fail = True

    else:
        try:
            aligner = rsync.Rsync_aligner(
                arduino_data.syncon_event_times.ravel().astype("int"),
                bonsai_data.syncon_event_times.ravel().astype("int")[
                    0 : arduino_data.syncon_event_times.size
                ],
                chunk_size=5,
                plot=True,
            )
        except:
            warnings.warn(
                "Alignment failure for mouse {} on date {}".format(mouse, date)
            )

            fail = True

    if fail:
        # import ipdb
        #
        # ipdb.set_trace()
        return None

    goals = [
        region_mapping.address_to_coord_dic[poke_address]
        for poke_address in arduino_data.reward_events[:, 1]
    ]

    session = Session(
        date=date, subject=mouse, experiment_day=experiment_day, maze_day=maze_day
    )

    # iterate over trials
    trial_count = 0
    search_t_start = None  # for the very first trial of the session the time to reconstruct past steps from, as for determininig the previous_loc of the initial step, is the beginning of the session. For that we need to pass None
    for i, tgoal in enumerate(zip(arduino_data.reward_event_times[1:-1], goals[1:-1])):
        # counters
        trial_count += 1

        # we get the time of reward and the current goal but to reconstruct trajectory we need to look 'backward'
        t_arduino_arrival_goal, goal = tgoal
        previous_goal = goals[
            i
        ]  # this does correspond to previous goal as we start iteration over second element of goals
        previous_goal_t = arduino_data.reward_event_times[i]

        # we are starting to iterate over goals[1:] so we need to add 1 to the enumeration counter
        # make up for the fact that we are starting iteration on the first and not the zeroth goal
        hardware_trial_count = i + 1
        block_size = 36
        goal_block = hardware_trial_count // block_size

        trial = Trial(
            maze,
            goal,
            delay_period=arduino_data.delay_period,
            # we need to add 1 because we are already ignoring the very first trial in our enumeration
            # trial_count is already added by 1
            goal_block=goal_block,
            hardware_trial_count=hardware_trial_count,
        )

        trial_begin_t_video = aligner.A_to_B(
            previous_goal_t + arduino_data.delay_period
        )
        trial_end_t_video = aligner.A_to_B(t_arduino_arrival_goal)

        port_visit_times_video, visited_ports = bonsai_data.get_trajectory(
            trial_begin_t_video, trial_end_t_video, maze
        )

        # FIXME: for a reason we don't yet understand sometimes the list of
        # visited_ports is empty. I suspect it is the case for trials where
        # reward is adjacent or the centroid gets lost. Needs investigation
        # and removal of this catch if
        if len(visited_ports) == 0:
            continue

        # TODO: search_t_start should move to at least the previous trial of the same session to save time not reconstructing every trajectory from the beginning

        last_port_before_trial_begin = bonsai_data.get_last_loc_before_trial_start(
            t_start=trial_begin_t_video,
            t_stop=trial_end_t_video,
            search_t_start=search_t_start,
        )

        # we save the start time of this trial so we can use if to truncate the segment prior to next trial for which we reconstruct the trajectory and infer locations visited such as to find previous_loc
        search_t_start = trial_begin_t_video

        #
        # if visited_ports[0] == previous_goal:
        #     all_visited_ports.extend(visited_ports[1:])
        # else:
        #     all_visited_ports.extend(visited_ports)
        # now we convert the time at which visits to a port are detected
        # back to Arduino time
        port_visit_times_arduino = aligner.B_to_A(port_visit_times_video)

        # FIXME: there could be a discrepancy between port_visit_times[0]
        # and the t_start we are using here: previous_goal_t +
        # arduino_data.delay_period
        #
        # FIXME:
        initial_step = Step(
            loc=visited_ports[0],
            trial=trial,
            time=previous_goal_t + arduino_data.delay_period,
            previous_loc=last_port_before_trial_begin,
        )

        trial.steps.append(initial_step)

        # iterate over steps

        for t_arrival_at_port, loc in zip(
            port_visit_times_arduino[1:], visited_ports[1:]
        ):
            session_step_count += 1
            # TODO: reorganize these messy ifs

            if (t_arrival_at_port - trial.steps[-1].time) < step_duration_threshold:
                # Step duration less than the threshold
                step = Step(
                    loc, trial, previous=trial.steps[-1], time=t_arrival_at_port
                )
                trial.steps.append(step)

            else:
                # The animal spent more than the threshold time at a given port
                if trial.steps[-1].loc != previous_goal:
                    # only discard if this was not the first step. They are allowed to spend longer at the previous port drinking
                    if not discard_whole_trials:
                        trial.shortened = (
                            True  # used to count trials that were shortened
                        )
                        n_shortenings += 1
                        discarded_steps_too_slow += len(trial.steps)
                        trial.steps = []
                        initial_step = Step(
                            loc=loc, trial=trial, time=t_arrival_at_port
                        )
                        trial.steps.append(initial_step)

                    else:
                        # the whole trial is discarded
                        discarded_trials_has_slow_step += 1
                        break

                else:
                    # it's the first step so it still gets included as normal
                    step = Step(
                        loc, trial, previous=trial.steps[-1], time=t_arrival_at_port
                    )
                    trial.steps.append(step)

            if loc == goal:
                # the animal is at the reward port
                # now we need to check that he actually poked and got the reward:
                trial.steps[-1].time = copy.deepcopy(t_arduino_arrival_goal)

                if (
                    np.abs(t_arduino_arrival_goal - t_arrival_at_port)
                    < jitter_tolerance
                ):
                    trial.steps[
                        -1
                    ].time = t_arduino_arrival_goal  # for the last step we use the arduino time 'cause it is more accurate

                    if min_trial_duration <= trial.duration <= max_trial_duration:
                        if trial.shortened:
                            shortened_trials += 1

                        if len(trial.steps) >= 1:
                            # discard empty trials where all previous steps to goal were discarded
                            session.trials.append(trial)

                        else:
                            discarded_trials_empty += 1
                    else:
                        discarded_trials_neg_duration += 1

                else:
                    # the animal didn't consume reward or there is another more exotic problem
                    discarded_trials_not_consumed += 1
                break

        # import ipdb
        #
        # ipdb.set_trace()
    # if len(session.trials) == 0:
    #     import ipdb
    #
    #     ipdb.set_trace()
    #     return None

    return session


# def load_steps(
#     background_file,
#     background_PNG,
#     arduino_file_path,
#     bonsai_metadata_file_path,
#     dlc_coords_file_path,
#     maze,
#     date=None,
#     mouse=None,
#     experiment_day=None,
#     maze_day=None,
#     fps=60,
# ):
#     session = load_session(
#         background_file=background_file,
#         background_PNG=background_PNG,
#         arduino_file_path=arduino_file_path,
#         bonsai_metadata_file_path=bonsai_metadata_file_path,
#         dlc_coords_file_path=dlc_coords_file_path,
#         maze=maze,
#         date=date,
#         mouse=mouse,
#         experiment_day=experiment_day,
#         maze_day=maze_day,
#         fps=fps,
#     )
#
#     steps = load_steps_multi([session])
#
#     return steps
#
#
# -----------------Load steps from sessions of multiple subjects---------------
def load_steps_multi(sessions):
    """Multisession Multisubject."""

    data = []

    for session in tqdm(sessions, position=0, desc="Loading Sessions"):
        maze_distances_dict = session.maze.distances_dictionary
        maze_optimal_choice_dict = session.maze.optimal_choice_dic
        maze_degree_dict = session.maze.degree

        if session.date:
            date_time = pd.to_datetime("{}-{}-{}".format(*session.date))
        else:
            date_time = None

        session_step_count = 0

        for i_t, t in tenumerate(
            session.trials, position=1, leave=False, desc="Processing Trial"
        ):
            for i_s, s in enumerate(t.steps[:-1]):
                step_duration = t.steps[i_s + 1].time - t.steps[i_s].time
                geodesic_distance_goal = maze_distances_dict[s.loc][s.trial.goal]

                available_nodes = s.available_nodes
                optimal_choice = maze_optimal_choice_dict[s.loc][s.trial.goal]

                chosen = available_nodes[s.choice]

                if chosen in optimal_choice:
                    choice_was_optimal = 1
                    geodesic_delta = -1
                else:
                    choice_was_optimal = 0
                    geodesic_delta = 1

                all_geodesic_deltas = [
                    -1 if node in optimal_choice else 1 for node in s.available_nodes
                ]

                all_degree = [maze_degree_dict[node] for node in s.available_nodes]

                next_step = t.steps[i_s + 1]

                data.append(
                    [
                        date_time,
                        session.experiment_day,
                        session.maze_day,
                        session.subject,
                        np.float64(s.time),
                        np.float64(next_step.time),
                        session.maze.name,
                        s.loc,
                        s.trial.goal,
                        s.trial.goal_block,
                        s.trial.hardware_trial_count,
                        available_nodes,
                        available_nodes[s.choice],
                        maze_loc_choice_to_cardinal(
                            session.maze, s.loc, available_nodes[s.choice]
                        ),
                        optimal_choice,
                        s.heading,
                        i_s,
                        session_step_count,
                        i_t,
                        step_duration,
                        geodesic_distance_goal,
                        choice_was_optimal,
                        geodesic_delta,
                        all_geodesic_deltas,
                        all_degree,
                        np.float64(t.delay_period),
                    ]
                )

                session_step_count += 1

    return pd.DataFrame(
        data,
        columns=[
            "date",
            "experiment_day",
            "maze_day",
            "subject",
            "time",
            "time_next_step",
            "maze_name",
            "location",
            "goal",
            "goal_block",
            "hardware_trial_count",
            "available_nodes",
            "choice",
            "cardinal_choice",
            "optimal_choice",
            "heading",
            "trial_step_count",
            "session_step_count",
            "session_trial_count",
            "step_duration",
            "geodesic_dist_goal",
            "choice_was_optimal",
            "geodesic_delta",
            "all_geodesic_deltas",
            "all_degree",
            "trial_delay_period",
        ],
    ).astype(
        {
            "maze_name": "string",
            "subject": "string",
            "cardinal_choice": "string",
            "heading": "string",
        }
    )


# -----------------Load trials from sessions of multiple subjects---------------
def load_trials_and_steps_multi(sessions):
    """Multisession Multisubject."""
    trial_data = []
    step_data = []

    for session in tqdm(sessions, position=0, desc="Loading Sessions"):
        maze_distances_dict = session.maze.distances_dictionary
        maze_optimal_choice_dict = session.maze.optimal_choice_dic

        if session.date:
            date_time = pd.to_datetime("{}-{}-{}".format(*session.date))
        else:
            date_time = None

        session_step_count = 0

        for i_t, t in tenumerate(
            session.trials, position=1, leave=False, desc="Processing Trial"
        ):
            n_optimal_steps = 0
            n_non_optimal_steps = 0

            for i_s, s in enumerate(t.steps[:-1]):
                step_duration = t.steps[i_s + 1].time - t.steps[i_s].time
                geodesic_distance_goal = maze_distances_dict[s.loc][s.trial.goal]

                available_nodes = s.available_nodes
                optimal_choice = maze_optimal_choice_dict[s.loc][s.trial.goal]

                chosen = available_nodes[s.choice]

                if chosen in optimal_choice:
                    choice_was_optimal = 1
                    geodesic_delta = -1
                    n_optimal_steps += 1
                else:
                    choice_was_optimal = 0
                    geodesic_delta = 1
                    n_non_optimal_steps += 1

                step_data.append(
                    [
                        date_time,
                        session.experiment_day,
                        session.maze_day,
                        session.subject,
                        s.time,
                        session.maze.name,
                        s.loc,
                        s.trial.goal,
                        s.trial.goal_block,
                        s.trial.hardware_trial_count,
                        available_nodes,
                        chosen,
                        optimal_choice,
                        s.heading,
                        i_s,
                        session_step_count,
                        i_t,
                        step_duration,
                        geodesic_distance_goal,
                        choice_was_optimal,
                        geodesic_delta,
                    ]
                )
                session_step_count += 1

            trial_geodesic_distance_goal = maze_distances_dict[t.steps[0].loc][t.goal]
            trial_data.append(
                [
                    date_time,
                    session.experiment_day,
                    session.maze_day,
                    session.subject,
                    t.t_stop - t.t_start,
                    session.maze.name,
                    t.steps[0].loc,
                    t.goal,
                    t.goal_block,
                    t.hardware_trial_count,
                    i_t,
                    trial_geodesic_distance_goal,
                    t.excess_steps(),
                    n_optimal_steps,
                    n_non_optimal_steps,
                ]
            )

    trials_df = pd.DataFrame(
        trial_data,
        columns=[
            "date",
            "experiment_day",
            "maze_day",
            "subject",
            "duration",
            "maze_name",
            "location",
            "goal",
            "goal_block",
            "hardware_trial_count",
            "session_trial_count",
            "geodesic_dist_goal",
            "excess_steps",
            "n_optimal_steps",
            "n_non_optimal_steps",
        ],
    )
    steps_df = pd.DataFrame(
        step_data,
        columns=[
            "date",
            "experiment_day",
            "maze_day",
            "subject",
            "time",
            "maze_name",
            "location",
            "goal",
            "goal_block",
            "hardware_trial_count",
            "available_nodes",
            "choice",
            "optimal_choice",
            "heading",
            "trial_step_count",
            "session_step_count",
            "session_trial_count",
            "step_duration",
            "geodesic_dist_goal",
            "choice_was_optimal",
            "geodesic_delta",
        ],
    )

    return steps_df, trials_df


def load_trials_multi(sessions):
    """Multisession Multisubject."""

    data = []

    for session in tqdm(sessions, position=0, desc="Loading Sessions"):
        maze_distances_dict = session.maze.distances_dictionary

        if session.date:
            date_time = pd.to_datetime("{}-{}-{}".format(*session.date))
        else:
            date_time = None

        for i_t, t in tenumerate(
            session.trials, position=1, leave=False, desc="Processing Trial"
        ):
            geodesic_distance_goal = maze_distances_dict[t.steps[0].loc][t.goal]

            data.append(
                [
                    date_time,
                    session.experiment_day,
                    session.maze_day,
                    session.subject,
                    t.t_stop - t.t_start,
                    session.maze.name,
                    t.steps[0].loc,
                    t.goal,
                    t.goal_block,
                    t.hardware_trial_count,
                    i_t,
                    geodesic_distance_goal,
                    t.excess_steps(),
                ]
            )

    return pd.DataFrame(
        data,
        columns=[
            "date",
            "experiment_day",
            "maze_day",
            "subject",
            "duration",
            "maze_name",
            "location",
            "goal",
            "goal_block",
            "hardware_trial_count",
            "session_trial_count",
            "geodesic_dist_goal",
            "excess_steps",
        ],
    )


def row_computations(choice, goal, location, heading, available_nodes):
    n_available_nodes = len(available_nodes)
    manhattan_dist_to_goal = manhattan_distance(location, goal)
    euclidean_dist_to_goal = euclidian_distance(location, goal)

    manhattan_delta = manhattan_distance(choice, goal) - manhattan_dist_to_goal
    euclidean_delta = euclidian_distance(choice, goal) - euclidean_dist_to_goal
    angle_to_goal = angle_between_nodes(location, choice, goal)
    angle_to_heading = angle_between_node_and_heading(location, choice, heading)
    signed_angle_to_heading = signed_angle_between_node_and_heading(
        location, choice, heading
    )

    all_manhattan_deltas = [
        manhattan_distance(choice, goal) - manhattan_dist_to_goal
        for choice in available_nodes
    ]
    all_euclidean_deltas = [
        euclidian_distance(choice, goal) - euclidean_dist_to_goal
        for choice in available_nodes
    ]
    all_angles_to_goal = [
        angle_between_nodes(location, choice, goal) for choice in available_nodes
    ]
    all_angles_to_heading = [
        angle_between_node_and_heading(location, choice, heading)
        for choice in available_nodes
    ]
    all_signed_angles_to_heading = [
        signed_angle_between_node_and_heading(location, choice, heading)
        for choice in available_nodes
    ]

    return (
        manhattan_delta,
        euclidean_delta,
        angle_to_goal,
        angle_to_heading,
        signed_angle_to_heading,
        n_available_nodes,
        manhattan_dist_to_goal,
        euclidean_dist_to_goal,
        np.array(all_manhattan_deltas, dtype=object),
        np.array(all_euclidean_deltas, dtype=object),
        np.array(all_angles_to_goal, dtype=object),
        np.array(all_angles_to_heading, dtype=object),
        np.array(all_signed_angles_to_heading, dtype=object),
    )


row_computations_vec = np.vectorize(
    row_computations  # , signature="(),(),(),(),()->(),(),(),(),(),(),(),(),(),(),()"
)


def compute_state_features(steps_df):
    """Computes statistical features over steps

    Inputs:
        steps_df    :   DataFrame with steps rows as produced by load_steps_multi

        mazes_dic   :   dictionary containing maze objects for maze names appearing in steps_df column 'maze_name'
    """
    # We take care of step local features first, i.e., features for which we only
    # need stepwise info to compute

    newfeatures = row_computations_vec(
        steps_df.choice,
        steps_df.goal,
        steps_df.location,
        steps_df.heading,
        steps_df.available_nodes,
    )

    (
        lst_manhattan_delta,
        lst_euclidean_delta,
        lst_angle_to_goal,
        lst_angle_to_heading,
        lst_signed_angle_to_heading,
        lst_n_available_nodes,
        lst_manhattan_dist_to_goal,
        lst_euclidean_dist_to_goal,
        lst_all_manhattan_deltas,
        lst_all_euclidean_deltas,
        lst_all_angle_to_goal,
        lst_all_angle_to_heading,
        lst_all_signed_angles_to_heading,
    ) = newfeatures

    # the lst_all_* features are, each of them, an array of arrays. we will turn it into a list of lists
    lst_all_manhattan_deltas = [row.tolist() for row in lst_all_manhattan_deltas]
    lst_all_euclidean_deltas = [row.tolist() for row in lst_all_euclidean_deltas]
    lst_all_angle_to_goal = [row.tolist() for row in lst_all_angle_to_goal]
    lst_all_angle_to_heading = [row.tolist() for row in lst_all_angle_to_heading]
    lst_all_signed_angles_to_heading = [
        row.tolist() for row in lst_all_signed_angles_to_heading
    ]

    steps_df_with_features = steps_df  # .copy()

    steps_df_with_features["manhattan_delta"] = lst_manhattan_delta
    steps_df_with_features["euclidean_delta"] = lst_euclidean_delta
    steps_df_with_features["angle_to_goal"] = lst_angle_to_goal
    steps_df_with_features["angle_to_heading"] = lst_angle_to_heading
    steps_df_with_features["signed_angle_to_heading"] = lst_signed_angle_to_heading
    steps_df_with_features["n_available_nodes"] = lst_n_available_nodes
    steps_df_with_features["manhattan_dist_goal"] = lst_manhattan_dist_to_goal
    steps_df_with_features["euclidean_dist_goal"] = lst_euclidean_dist_to_goal
    steps_df_with_features["all_manhattan_delta"] = lst_all_manhattan_deltas
    steps_df_with_features["all_euclidean_delta"] = lst_all_euclidean_deltas
    steps_df_with_features["all_angle_to_goal"] = lst_all_angle_to_goal
    steps_df_with_features["all_angle_to_heading"] = lst_all_angle_to_heading
    steps_df_with_features["all_signed_angle_to_heading"] = (
        lst_all_signed_angles_to_heading
    )

    return steps_df_with_features


def compute_maze_path_features(maze_steps_df):
    # Now we compute features that need global information, i.e., information from
    # other steps in the session, other sessions, etc.
    steps_df = maze_steps_df.copy()
    steps_df["maze_loc_count"] = steps_df.groupby(
        ["location", "subject", "maze_name"]
    ).cumcount()
    steps_df["maze_loc_goal_count"] = steps_df.groupby(
        ["location", "goal", "subject", "maze_name"]
    ).cumcount()
    steps_df["maze_goal_count"] = steps_df.groupby(
        ["goal", "subject", "maze_name"]
    ).cumcount()

    # we first copy the session_trial_count and only then add the number of trials of past sessions to get the global trial count for a given maze
    steps_df.loc[:, "maze_trial_count"] = steps_df["session_trial_count"]
    subject_maze_trial_counts_per_session = (
        steps_df.groupby(["subject", "maze_name", "maze_day"])[
            "session_trial_count"
        ].max()
        + 1
    )

    for subject_maze_session in subject_maze_trial_counts_per_session.index:
        subject, maze_name, maze_day = subject_maze_session
        n_trials_per_session = subject_maze_trial_counts_per_session[
            (subject, maze_name)
        ]
        sum_trials_up_to_this_session = n_trials_per_session.cumsum()
        past_trials_to_add_to_each_session = (
            sum_trials_up_to_this_session - n_trials_per_session
        )
        past_trials_to_add_to_this_session = past_trials_to_add_to_each_session[
            maze_day
        ]

        steps_df.loc[
            (steps_df["subject"] == subject)
            & (steps_df["maze_name"] == maze_name)
            & (steps_df["maze_day"] == maze_day),
            "maze_trial_count",
        ] += past_trials_to_add_to_this_session

    # we do the same for the step count
    steps_df.loc[:, "maze_step_count"] = steps_df["session_step_count"]
    subject_maze_step_counts_per_session = (
        steps_df.groupby(["subject", "maze_name", "maze_day"])[
            "session_step_count"
        ].max()
        + 1
    )

    for subject_maze_session in subject_maze_step_counts_per_session.index:
        subject, maze_name, maze_day = subject_maze_session
        n_steps_per_session = subject_maze_step_counts_per_session[(subject, maze_name)]
        sum_steps_up_to_this_session = n_steps_per_session.cumsum()
        past_steps_to_add_to_each_session = (
            sum_steps_up_to_this_session - n_steps_per_session
        )
        past_steps_to_add_to_this_session = past_steps_to_add_to_each_session[maze_day]

        steps_df.loc[
            (steps_df["subject"] == subject)
            & (steps_df["maze_name"] == maze_name)
            & (steps_df["maze_day"] == maze_day),
            "maze_step_count",
        ] += past_steps_to_add_to_this_session

    return steps_df


def compute_trial_path_features(steps_df):
    steps_df = steps_df  # .copy()
    steps_df["trial_loc_count"] = steps_df.groupby(
        ["location", "maze_day", "session_trial_count", "subject", "maze_name"]
    ).cumcount()

    # compute revisit values for nodes and edges
    lst_dir_edge_chosen = []
    for index, row in steps_df.iterrows():
        lst_dir_edge_chosen.append((row["location"], row["choice"]))

    steps_df["dir_edge_chosen"] = lst_dir_edge_chosen

    lst_chosen_dir_edge_visited = []
    lst_available_dir_edge_visited = []
    lst_chosen_node_visited = []
    lst_available_node_visited = []

    lst_chosen_dir_edge_visited_lag2 = []
    lst_available_dir_edge_visited_lag2 = []
    lst_chosen_node_visited_lag2 = []
    lst_available_node_visited_lag2 = []

    for group_name, trial_df in steps_df.groupby(
        ["maze_day", "session_trial_count", "subject", "maze_name"]
    ):
        trial_first_idx = trial_df.index[0]
        for index, row in trial_df.iterrows():
            idx_chosen_node = row["available_nodes"].index(row["choice"])
            # we check if the chosen edge was already crossed in this direction
            available_dir_edge_visited = []
            available_dir_edge_visited_lag2 = []
            for node in row["available_nodes"]:
                available_edge = (row["location"], node)

                # the logic might seem inverted (checking elementwise if the elements of the list of dir_edges_chosen during current trial is in the list of length 1 with the choice and testing if there is any hit) but it is correct
                if (
                    trial_df.loc[trial_first_idx : index - 1, "dir_edge_chosen"]
                    .isin([available_edge])
                    .any()
                ):
                    available_dir_edge_visited.append(1)
                else:
                    available_dir_edge_visited.append(-1)

                # now the same for a lag=2, which means we exclude the last chosen "dir_edge"

                if (
                    trial_df.loc[trial_first_idx : index - 2, "dir_edge_chosen"]
                    .isin([available_edge])
                    .any()
                ):
                    available_dir_edge_visited_lag2.append(1)
                else:
                    available_dir_edge_visited_lag2.append(-1)

            lst_available_dir_edge_visited.append(available_dir_edge_visited)
            lst_available_dir_edge_visited_lag2.append(available_dir_edge_visited_lag2)
            lst_chosen_dir_edge_visited.append(
                available_dir_edge_visited[idx_chosen_node]
            )
            lst_chosen_dir_edge_visited_lag2.append(
                available_dir_edge_visited_lag2[idx_chosen_node]
            )

            # now the same for all the available nodes
            available_node_visited = []
            available_node_visited_lag2 = []
            for node in row["available_nodes"]:
                if (
                    trial_df.loc[trial_first_idx : index - 1, "location"]
                    .isin([node])
                    .any()
                ):
                    available_node_visited.append(1)
                else:
                    available_node_visited.append(-1)

                # now the same for a lag=2, which means we exclude the node where the agent just came from because it is indissociable from the nobackward
                if (
                    trial_df.loc[trial_first_idx : index - 2, "location"]
                    .isin([node])
                    .any()
                ):
                    available_node_visited_lag2.append(1)
                else:
                    available_node_visited_lag2.append(-1)

            lst_available_node_visited.append(available_node_visited)
            lst_chosen_node_visited.append(available_node_visited[idx_chosen_node])

            lst_available_node_visited_lag2.append(available_node_visited_lag2)
            lst_chosen_node_visited_lag2.append(
                available_node_visited_lag2[idx_chosen_node]
            )

    steps_df["chosen_dir_edge_visited"] = lst_chosen_dir_edge_visited
    steps_df["available_dir_edge_visited"] = lst_available_dir_edge_visited
    steps_df["chosen_dir_edge_visited_lag2"] = lst_chosen_dir_edge_visited_lag2
    steps_df["available_dir_edge_visited_lag2"] = lst_available_dir_edge_visited_lag2
    steps_df["chosen_node_visited"] = lst_chosen_node_visited
    steps_df["available_node_visited"] = lst_available_node_visited
    steps_df["chosen_node_visited_lag2"] = lst_chosen_node_visited_lag2
    steps_df["available_node_visited_lag2"] = lst_available_node_visited_lag2

    return steps_df


def compute_features(steps_df):
    steps_df = steps_df.copy()
    steps_df = compute_state_features(steps_df)
    steps_df = compute_trial_path_features(steps_df)
    # steps_df_with_maze_path_features = compute_maze_path_features(
    #     steps_df_with_state_features
    # )
    # trial path features only need to have data from the whole trial but not from previous trials
    return steps_df


def datetime_to_list(dt):
    year, month, day, *_ = dt.timetuple()
    time_list = [str(year).zfill(4), str(month).zfill(2), str(day).zfill(2)]
    return time_list


def infer_previous_loc_from_loc_heading(loc, heading):
    if heading == "o":
        return None
    elif heading == ">":
        return (loc[0] - 1, loc[1])
    elif heading == "<":
        return (loc[0] + 1, loc[1])
    elif heading == "^":
        return (loc[0], loc[1] - 1)
    elif heading == "v":
        return (loc[0], loc[1] + 1)


def load_session_from_steps_df(steps_df, mazes_dic, check_consistency=False):
    # load mandatory metadata for Session
    date = datetime_to_list(steps_df.date.unique()[0])
    subject = steps_df.subject.unique()[0]
    experiment_day = int(steps_df.experiment_day.unique()[0])
    maze_day = int(steps_df.maze_day.unique()[0])

    session = Session(
        date=date, subject=subject, experiment_day=experiment_day, maze_day=maze_day
    )

    # load additional metadata
    maze_name = steps_df.maze_name.unique()[0]

    for trial_number, trial_df in steps_df.groupby("session_trial_count"):
        # TODO: add delay_period attribute
        goal = trial_df.goal.unique()[0]

        trial = Trial(maze=mazes_dic[maze_name], goal=goal)
        trial.delay_period = trial_df.trial_delay_period.unique()[0]

        previous_step = None
        for ind, row in trial_df.iterrows():
            step = Step(
                loc=row.location,
                trial=trial,
                time=np.float64(row.time),
                previous=previous_step,
                previous_loc=infer_previous_loc_from_loc_heading(
                    row.location, row.heading
                ),
            )

            trial.steps.append(step)

            previous_step = step

        # add the very last step of the trial
        # row is going to be the last row of the DataFrame containing only the steps of current trial
        # and ind will be its index
        last_step_row = trial_df.iloc[-1]
        last_step = Step(
            loc=last_step_row.choice,
            trial=trial,
            time=np.float64(last_step_row.time_next_step),
            previous=previous_step,
            previous_loc=last_step_row.location,
        )

        trial.steps.append(last_step)

        session.trials.append(trial)

    return session
