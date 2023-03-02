import copy
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tenumerate

from mazex.data.regionmapping import RegionMapping
from mazex.data.arduino import ArduinoDataFile
from mazex.data.bonsai import BonsaiData
from mazex import Session, Trial, Step
from mazex.utils import manhattan_distance, euclidian_distance, angle_between_nodes, angle_between_node_and_heading

import rsync


def load_session(background_file, background_PNG, arduino_file_path, bonsai_metadata_file_path, dlc_coords_file_path,
                 maze, date=None, mouse=None, experiment_day=None, maze_day=None, fps=60):
    '''Build Session object with data for that mouse and experiment_day
    '''
    #TODO: turn region_mapping into a function argument
    region_mapping = RegionMapping(background_file, background_PNG)

    # FIXME: turn these into arguments
    min_trial_duration = 1000 #ms
    step_duration_threshold = 700000  #ms
    jitter_tolerance = 2000 #ms
    discard_whole_trials = True # If True, then trials with long steps will not be trimmed but entirely discarded
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
        dlc_coords_file_path, bonsai_metadata_file_path,
        metadata_delimiter=',', fps=fps, background_file=background_file,
        background_PNG=background_PNG, maze=maze, frame_number_column=0,
        timestamps_column=1, sync_column=2, hardware_timestamps=True)

    if len(arduino_data.syncon_event_times)==0:
        warnings.warn("Arduino sync signal missing for mouse {} on day {}".format(mouse, date))
        return None

    if len(bonsai_data.syncon_event_times)==0:
        warnings.warn("Bonsai video sync signal missing for mouse {} on day {}".format(mouse, date))
        return None

    try:
        aligner = rsync.Rsync_aligner(
            arduino_data.syncon_event_times.ravel().astype('int'),
            bonsai_data.syncon_event_times.ravel().astype('int')[0:arduino_data.syncon_event_times.size],
            chunk_size=5, plot=True)
    except:
        warnings.warn("Alignment failure for mouse {} on date {}".format(mouse,date))
        return None

    goals = [region_mapping.address_to_coord_dic[poke_address] for poke_address in arduino_data.reward_events[:,1]]

    session = Session(date=date, subject=mouse, experiment_day=experiment_day, maze_day=maze_day)

    #iterate over trials
    trial_count = 0
    for i, tgoal in enumerate(zip(arduino_data.reward_event_times[1:-1], goals[1:-1])):
        #counters
        trial_count += 1

        # we get the time of reward and the current goal but to reconstruct trajectory we need to look 'backward'
        t_arduino_arrival_goal, goal = tgoal
        previous_goal = goals[i] #this does correspond to previous goal as we start iteration over second element of goals
        previous_goal_t = arduino_data.reward_event_times[i]

        trial = Trial(maze, goal, delay_period = arduino_data.delay_period)

        trial_begin_t_video = aligner.A_to_B(previous_goal_t + arduino_data.delay_period)
        trial_end_t_video = aligner.A_to_B(t_arduino_arrival_goal)

        port_visit_times_video, visited_ports = bonsai_data.get_trajectory(trial_begin_t_video, trial_end_t_video, maze)

        # FIXME: for a reason we don't yet understand sometimes the list of
        # visited_ports is empty. I suspect it is the case for trials where
        # reward is adjacent or the centroid gets lost. Needs investigation
        # and removal of this catch if
        if len(visited_ports) == 0:
            continue
        # now we convert the time at which visits to a port are detected
        # back to Arduino time
        port_visit_times_arduino = aligner.B_to_A(port_visit_times_video)

        # FIXME: there could be a discrepancy between port_visit_times[0]
        # and the t_start we are using here: previous_goal_t +
        # arduino_data.delay_period
        #
        # FIXME:
        initial_step = Step(loc=visited_ports[0], trial=trial, time=previous_goal_t + arduino_data.delay_period)

        trial.steps.append(initial_step)

        #iterate over steps

        for t_arrival_at_port, loc in zip(port_visit_times_arduino[1:], visited_ports[1:]):
            session_step_count += 1
            #TODO: reorganize these messy ifs

            if (t_arrival_at_port - trial.steps[-1].time) < step_duration_threshold:
                # Step duration less than the threshold
                step = Step(loc, trial, previous=trial.steps[-1], time=t_arrival_at_port)
                trial.steps.append(step)

            else:
                # The animal spent more than the threshold time at a given port
                if trial.steps[-1].loc != previous_goal:
                    #only discard if this was not the first step. They are allowed to spend longer at the previous port drinking
                    if not discard_whole_trials:
                        trial.shortened = True #used to count trials that were shortened
                        n_shortenings += 1
                        discarded_steps_too_slow += len(trial.steps)
                        trial.steps = []
                        initial_step = Step(loc=loc, trial=trial, time=t_arrival_at_port)
                        trial.steps.append(initial_step)

                    else:
                        #the whole trial is discarded
                        discarded_trials_has_slow_step += 1
                        break

                else:
                    #it's the first step so it still gets included as normal
                    step = Step(loc, trial, previous = trial.steps[-1], time=t_arrival_at_port)
                    trial.steps.append(step)

            if loc == goal:
                # the animal is at the reward port
                # now we need to check that he actually poked and got the reward:
                trial.steps[-1].time = copy.deepcopy(t_arduino_arrival_goal)

                if np.abs(t_arduino_arrival_goal - t_arrival_at_port) < jitter_tolerance:
                    trial.steps[-1].time = t_arduino_arrival_goal # for the last step we use the arduino time 'cause it is more accurate

                    if  min_trial_duration <= trial.duration <= max_trial_duration:
                        if trial.shortened:
                            shortened_trials += 1

                        if len(trial.steps) >= 3:
                            #discard empty trials where all previous steps to goal were discarded
                            session.trials.append(trial)

                        else:
                            discarded_trials_empty += 1
                    else:
                        discarded_trials_neg_duration += 1

                else:
                    #the animal didn't consume reward or there is another more exotic problem
                    discarded_trials_not_consumed += 1
                break

    return session


#-----------------Load steps from sessions of multiple subjects---------------
def load_steps_multi(sessions):
    """Multisession Multisubject.
    """

    data = []

    for session in tqdm(sessions, position=0, desc='Loading Sessions'):

        maze_distances_dict = session.maze.distances_dictionary
        maze_optimal_choice_dict = session.maze.optimal_choice_dic

        if session.date:
            date_time = pd.to_datetime('{}-{}-{}'.format(*session.date))
        else:
            date_time = None

        session_step_count = 0

        for i_t, t in tenumerate(session.trials, position=1, leave=False, desc='Processing Trial'):

            for i_s, s in enumerate(t.steps[:-1]):

                step_duration = t.steps[i_s+1].time - t.steps[i_s].time
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

                data.append([date_time, session.experiment_day, session.maze_day, session.subject, s.time, session.maze.name, s.loc, s.trial.goal, available_nodes, available_nodes[s.choice], optimal_choice, s.heading, i_s, session_step_count, i_t, step_duration, geodesic_distance_goal, choice_was_optimal, geodesic_delta])

                session_step_count += 1

    return pd.DataFrame(data, columns=['date', 'experiment_day', 'maze_day', 'subject', 'time', 'maze_name', 'location', 'goal', 'available_nodes', 'choice', 'optimal_choice', 'heading', 'trial_step_count', 'session_step_count', 'session_trial_count', 'step_duration', 'geodesic_dist_goal', 'choice_was_optimal', 'geodesic_delta'])


#-----------------Load trials from sessions of multiple subjects---------------
def load_trials_and_steps_multi(sessions):
    """Multisession Multisubject.
    """
    trial_data = []
    step_data = []

    for session in tqdm(sessions, position=0, desc='Loading Sessions'):

        maze_distances_dict = session.maze.distances_dictionary
        maze_optimal_choice_dict = session.maze.optimal_choice_dic

        if session.date:
            date_time = pd.to_datetime('{}-{}-{}'.format(*session.date))
        else:
            date_time = None

        session_step_count = 0

        for i_t, t in tenumerate(session.trials, position=1, leave=False, desc='Processing Trial'):

            n_optimal_steps = 0
            n_non_optimal_steps = 0

            for i_s, s in enumerate(t.steps[:-1]):

                step_duration = t.steps[i_s+1].time - t.steps[i_s].time
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

                step_data.append([date_time, session.experiment_day, session.maze_day, session.subject, s.time, session.maze.name, s.loc, s.trial.goal, available_nodes, chosen, optimal_choice, s.heading, i_s, session_step_count, i_t, step_duration, geodesic_distance_goal, choice_was_optimal, geodesic_delta])
                session_step_count += 1

            trial_geodesic_distance_goal = maze_distances_dict[t.steps[0].loc][t.goal]
            trial_data.append([date_time, session.experiment_day, session.maze_day, session.subject, t.t_stop - t.t_start, session.maze.name, t.steps[0].loc, t.goal, i_t, trial_geodesic_distance_goal, t.excess_steps(), n_optimal_steps, n_non_optimal_steps])

    trials_df = pd.DataFrame(trial_data, columns=['date', 'experiment_day', 'maze_day', 'subject', 'duration', 'maze_name', 'location', 'goal', 'session_trial_count', 'geodesic_dist_goal', 'excess_steps', 'n_optimal_steps', 'n_non_optimal_steps'])
    steps_df = pd.DataFrame(step_data, columns=['date', 'experiment_day', 'maze_day', 'subject', 'time', 'maze_name', 'location', 'goal', 'available_nodes', 'choice', 'optimal_choice', 'heading', 'trial_step_count', 'session_step_count', 'session_trial_count', 'step_duration', 'geodesic_dist_goal', 'choice_was_optimal', 'geodesic_delta']) 

    return steps_df, trials_df



def load_trials_multi(sessions):
    """Multisession Multisubject.
    """

    data = []

    for session in tqdm(sessions, position=0, desc='Loading Sessions'):

        maze_distances_dict = session.maze.distances_dictionary

        if session.date:
            date_time = pd.to_datetime('{}-{}-{}'.format(*session.date))
        else:
            date_time = None

        for i_t, t in tenumerate(session.trials, position=1, leave=False, desc='Processing Trial'):

            geodesic_distance_goal = maze_distances_dict[t.steps[0].loc][t.goal]

            data.append([date_time, session.experiment_day, session.maze_day, session.subject, t.t_stop - t.t_start, session.maze.name, t.steps[0].loc, t.goal, i_t, geodesic_distance_goal, t.excess_steps()])

    return pd.DataFrame(data, columns=['date', 'experiment_day', 'maze_day', 'subject', 'duration', 'maze_name', 'location', 'goal', 'session_trial_count', 'geodesic_dist_goal', 'excess_steps'])


def row_computations(choice, goal, location, heading, available_nodes):
    manhattan_dist_to_goal = manhattan_distance(location, goal)
    euclidean_dist_to_goal = euclidian_distance(location, goal)

    manhattan_delta = manhattan_distance(choice, goal) - manhattan_dist_to_goal
    euclidean_delta = euclidian_distance(choice, goal) - euclidean_dist_to_goal
    angle_to_goal = angle_between_nodes(location, choice, goal)
    angle_to_heading = angle_between_node_and_heading(location, choice, heading)
    n_available_nodes = len(available_nodes)

    return (manhattan_delta, euclidean_delta, angle_to_goal, angle_to_heading, n_available_nodes, manhattan_dist_to_goal, euclidean_dist_to_goal)


row_computations_vec = np.vectorize(row_computations)


def compute_features(steps_df):
    """Computes statistical features over steps

    Inputs:
        steps_df    :   DataFrame with steps rows as produced by load_steps_multi

        mazes_dic   :   dictionary containing maze objects for maze names appearing in steps_df column 'maze_name'
    """
    # We take care of step local features first, i.e., features for which we only
    # need stepwise info to compute

    newfeatures = row_computations_vec(steps_df.choice, steps_df.goal, steps_df.location, steps_df.heading, steps_df.available_nodes)

    lst_manhattan_delta, lst_euclidean_delta, lst_angle_to_goal, lst_angle_to_heading, lst_n_available_nodes, lst_manhattan_dist_to_goal, lst_euclidean_dist_to_goal = newfeatures

    steps_df_with_features = steps_df

    steps_df_with_features['manhattan_delta'] = lst_manhattan_delta
    steps_df_with_features['euclidean_delta'] = lst_euclidean_delta
    steps_df_with_features['angle_to_goal'] = lst_angle_to_goal
    steps_df_with_features['angle_to_heading'] = lst_angle_to_heading
    steps_df_with_features['n_available_nodes'] = lst_n_available_nodes
    steps_df_with_features['manhattan_dist_goal'] = lst_manhattan_dist_to_goal
    steps_df_with_features['euclidean_dist_goal'] = lst_euclidean_dist_to_goal
    
    # Now we compute features that need global information, i.e., information from
    # other steps in the session, other sessions, etc.

    steps_df_with_features['loc_count'] = steps_df.groupby(["location", "subject", "maze_name"]).cumcount()
    steps_df_with_features['loc_goal_count'] = steps_df.groupby(["location", "goal", "subject", "maze_name"]).cumcount()
    steps_df_with_features['goal_count'] = steps_df.groupby(["goal", "subject", "maze_name"]).cumcount()
    steps_df_with_features['trial_loc_count'] = steps_df.groupby(["location", "maze_day", "session_trial_count", "subject", "maze_name"]).cumcount()

    # we first copy the session_trial_count and only then add the number of trials of past sessions to get the global trial count for a given maze
    steps_df_with_features.loc[:, 'maze_trial_count'] = steps_df_with_features['session_trial_count']
    subject_maze_trial_counts_per_session = steps_df_with_features.groupby(["subject", "maze_name", "maze_day"])['session_trial_count'].max()+1 

    for subject_maze_session in subject_maze_trial_counts_per_session.index:
        subject, maze_name, maze_day = subject_maze_session
        n_trials_per_session = subject_maze_trial_counts_per_session[(subject, maze_name)]
        sum_trials_up_to_this_session = n_trials_per_session.cumsum()
        past_trials_to_add_to_each_session = sum_trials_up_to_this_session - n_trials_per_session
        past_trials_to_add_to_this_session = past_trials_to_add_to_each_session[maze_day]

        steps_df_with_features.loc[(steps_df_with_features['subject']==subject)&(steps_df_with_features['maze_name']==maze_name)&(steps_df_with_features['maze_day']==maze_day), 'maze_trial_count'] += past_trials_to_add_to_this_session

    # we do the same for the step count
    steps_df_with_features.loc[:, 'maze_step_count'] = steps_df_with_features['session_step_count']
    subject_maze_step_counts_per_session = steps_df_with_features.groupby(["subject", "maze_name", "maze_day"])['session_step_count'].max()+1 

    for subject_maze_session in subject_maze_step_counts_per_session.index:
        subject, maze_name, maze_day = subject_maze_session
        n_steps_per_session = subject_maze_step_counts_per_session[(subject, maze_name)]
        sum_steps_up_to_this_session = n_steps_per_session.cumsum()
        past_steps_to_add_to_each_session = sum_steps_up_to_this_session - n_steps_per_session
        past_steps_to_add_to_this_session = past_steps_to_add_to_each_session[maze_day]

        steps_df_with_features.loc[(steps_df_with_features['subject']==subject)&(steps_df_with_features['maze_name']==maze_name)&(steps_df_with_features['maze_day']==maze_day), 'maze_step_count'] += past_steps_to_add_to_this_session

    return steps_df_with_features
