import copy
import numpy as np
import pandas as pd

from mazex.data.regionmapping import RegionMapping
from mazex.data.arduino import ArduinoDataFile
from mazex.data.bonsai import BonsaiData
from mazex import Session, Trial, Step

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

    aligner = rsync.Rsync_aligner(
        arduino_data.syncon_event_times.ravel().astype('int'),
        bonsai_data.syncon_event_times.ravel().astype('int')[0:arduino_data.syncon_event_times.size],
        chunk_size=5, plot=True)

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

def load_steps_from_single_session(session):

    data = []

    for i_t, t in enumerate(session.trials):
        for i_s, s in enumerate(t.steps[:-1]):

            # check if choice decreased geodesic distance in which case it was an optimal choice
            if session.maze.distances_dictionary[s.available_nodes[s.choice]][s.trial.goal] < session.maze.distances_dictionary[s.loc][s.trial.goal]:
                was_optimal = 1
            else:
                was_optimal = 0

            data.append([s.loc, s.trial.goal, s.available_nodes, s.available_nodes[s.choice], s.heading, was_optimal, s.trial.maze.name, i_s, i_t])

    steps_df = pd.DataFrame(data, columns=['loc', 'goal', 'available_nodes', 'choice', 'heading', 'choice_was_optimal', 'maze_name', 'step_#', 'trial_#'])

    return steps_df


def load_steps_from_several_sessions(sessions):

    # FIX: this is a bit of a hacky way to get an ordered nodes_list data
    # structure
    nodes_list = sessions[0].maze.nodes()

    data = []

    step_count = 0
    n_past_visits_loc = dict.fromkeys(nodes_list, 0)

    # n_past_visits_loc_goal = dict.fromkeys(nodes_list, dict.fromkeys(nodes_list, 0))

    n_past_visits_loc_goal = dict.fromkeys(nodes_list)
    
    for node in nodes_list:
        n_past_visits_loc_goal[node] = dict.fromkeys(nodes_list, 0)

    n_past_steps_goal = dict.fromkeys(nodes_list, 0)
    n_past_trials_goal = dict.fromkeys(nodes_list, 0)

    for i_ses, session in enumerate(sessions):

        for i_t, t in enumerate(session.trials):

            n_past_visits_loc_trial = dict.fromkeys(nodes_list, 0)

            for i_s, s in enumerate(t.steps[:-1]):

                # check if choice decreased geodesic distance in which case it was an optimal choice
                if session.maze.distances_dictionary[s.available_nodes[s.choice]][s.trial.goal] < session.maze.distances_dictionary[s.loc][s.trial.goal]:
                    was_optimal = 1
                else:
                    was_optimal = 0

                data.append([session.subject, session.maze.name, s.loc, s.trial.goal, s.available_nodes, s.available_nodes[s.choice], s.heading, was_optimal, i_s, i_t, i_ses, n_past_visits_loc_trial[s.loc], n_past_visits_loc[s.loc], n_past_visits_loc_goal[s.loc][s.trial.goal], n_past_steps_goal[s.trial.goal], n_past_trials_goal[s.trial.goal], step_count])

                # and now we add to the past_visits counters

                n_past_visits_loc[s.loc] += 1
                n_past_visits_loc_trial[s.loc] += 1
                n_past_visits_loc_goal[s.loc][s.trial.goal] += 1

                n_past_steps_goal[s.trial.goal] += 1

                step_count += 1

            n_past_trials_goal[s.trial.goal] += 1

    steps_df = pd.DataFrame(data, columns=['subject', 'maze_name', 'loc', 'goal', 'available_nodes', 'choice', 'heading', 'choice_was_optimal', 'step_n', 'trial_n', 'session_n', 'n_past_visits_loc_trial', 'n_past_visits_loc', 'n_past_visits_loc_goal', 'n_past_steps_goal','n_past_trials_goal', 'step_count'])

    return steps_df


def load_steps_multi(sessions):
    """Multisession Multisubject.
    """

    data = []

    for session in sessions:

        date_time = pd.to_datetime('{}-{}-{}'.format(*session.date))
        session_step_count = 0

        for i_t, t in enumerate(session.trials):

            for i_s, s in enumerate(t.steps[:-1]):

                data.append([date_time, session.experiment_day, session.maze_day, session.subject, session.maze.name, s.loc, s.trial.goal, s.available_nodes, s.available_nodes[s.choice], session.maze.optimal_choice_dic[s.loc][s.trial.goal], s.heading, i_s, session_step_count, i_t])

                session_step_count += 1

    steps_df = pd.DataFrame(data, columns=['date', 'experiment_day', 'maze_day', 'subject', 'maze_name', 'loc', 'goal', 'available_nodes', 'choice', 'optimal_choice', 'heading', 'trial_step_count', 'session_step_count', 'session_trial_count'])

    def choice_in_optimal(row):
        if row.choice in row.optimal_choice:
            return 1
        else:
            return 0

    steps_df['choice_was_optimal'] = steps_df.apply(choice_in_optimal, axis=1)

    return steps_df


def compute_features(steps_df):
    """Computes statistical features over steps

    Inputs:
        steps_df    :   DataFrame with steps rows as produced by load_steps_multi

        mazes_dic   :   dictionary containing maze objects for maze names appearing in steps_df column 'maze_name'
    """
    #all_maze_names = steps_df['maze_name'].unique
    #all_subject_names = steps_df['subject'].unique

    #assert all(maze_name in mazes_dic.keys() for maze_name in all_maze_names), "Not all maze_name instances in steps_df are in mazes_dic's keys"

    #loc_count = []
    #loc_goal_count = []
    #goal_count = []
    #trial_loc_count = []

    grouping = steps_df.groupby(['subject', 'maze_name'])

    features = ['loc_count', 'loc_goal_count', 'goal_count', 'trial_loc_count']

    # this is the augmented dataframe to which we will write the computed stats
    steps_df_with_features = steps_df.reindex(steps_df.columns.tolist() + features, axis=1)

    for group_name in grouping.groups.keys():
        df = grouping.get_group(group_name)

        step_iterator = df.iterrows()

        for i, step_row in step_iterator:

            # get a DataFrame with all rows up to current step row
            df_up_to_row = df.iloc[:i+1, :]

            # compute stats
            steps_df_with_features.loc[i, 'loc_count'] = len(df_up_to_row[df_up_to_row['loc']==step_row['loc']])
            steps_df_with_features.loc[i, 'loc_goal_count'] = len(df_up_to_row[(df_up_to_row['loc']==step_row['loc']) & (df_up_to_row['goal']==step_row['goal'])])
            steps_df_with_features.loc[i, 'goal_count'] = len(df_up_to_row[df_up_to_row['goal']==step_row['goal']])
            steps_df_with_features.loc[i, 'trial_loc_count'] = len(df_up_to_row[(df_up_to_row['loc']==step_row['loc']) & (df_up_to_row['maze_day']==step_row['maze_day']) & (df_up_to_row['session_trial_count']==step_row['session_trial_count'])])


            if '10II' in group_name:
                import ipdb; ipdb.set_trace()
#    # add features to steps_df
#    for maze_name in all_maze_names:
#        for subject in all_subject_names:
#            steps_maze_subject = steps_df[steps_df['maze_name']==maze_name & steps_df['subject'] ==subject]
#
#            # count number of times location has ever been visited in this maze topology

    return steps_df_with_features
