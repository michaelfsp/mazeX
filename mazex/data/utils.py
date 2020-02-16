
#-----------------------------Data loading methods----------------------------
"""Now we extract the regions corresponding to the different pokes on the
image using the annotated SVG file.
"""
region_mapping = mx.data.regionmapping.RegionMapping(background_file, background_PNG)
with open('./cache/time_shifts.pickle', 'rb') as f:
    time_shifts = pickle.load(f)

step_count = 0
trial_count=0
session_count = 0

discarded_trials_has_slow_step = 0
discarded_trials_neg_duration = 0
discarded_trials_not_consumed = 0
discarded_trials_alignment=0
discarded_steps_too_slow = 0
discarded_trials_empty = 0
shortened_trials = 0
n_shortenings = 0

inter_port_intervals = []

sessions = [] # all sessions separated by day an mouse
super_sessions = {} # for each mouse all trials together in chronological order
mega_session = mx.Session()
session_durations = {}
sessions_no_trials = 0

min_trial_duration = 1000 #ms
step_duration_threshold = 5500 #ms
jitter_tolerance = 2000 #ms

discard_whole_trials = False # If True, then trials with long steps will not be trimmed but entirely discarded
max_trial_duration = 100000

for mouse in mice:
    super_sessions[mouse] = mx.Session(subject = mouse)

def load_mouse(mouse, date_folders):
for d in experiment_days[:]:#sessions_by_day:
    date = str(d)+'122014'

    session_durations[d] = []

    cv_data_dir = os.path.join(data_folder, 'cv', date)
    for mouse in sessions_by_day[d]:

        arduino_file_path = os.path.join(data_folder, 'arduino', date, mouse+'.txt')
        arduino_data = mx.data.arduino.ArduinoDataFile(arduino_file_path)

        stored_coords_file_path = os.path.join(data_folder, 'cv', date, mouse+'.pickle')
        with open(stored_coords_file_path, 'rb') as f:
            t_cv, visited_ports = pickle.load(f)
        t_start_on_video = time_shifts[date][mouse][0]
        t_cv = np.array(t_cv)
        t_cv -= t_start_on_video #in the pickled files t_start_on_video has not been discounted, i.e., time is recording time
        
        # we only count after the first step
        
        inter_port_intervals.extend(np.diff(t_cv[1:]).tolist())

        goals = [region_mapping.address_to_coord_dic[poke_address] for poke_address in arduino_data.reward_events[:,1]]

        session = mx.Session(date=date, subject=mouse)

        #iterate over trials
        for i, tgoal in enumerate(zip(arduino_data.reward_event_times[1:], goals[1:])):
            #counters
            trial_count += 1
            shortened = False
                
            # we get the time of reward and the current goal but to reconstruct trajectory we need to look 'backward'
            t_arduino, goal = tgoal
            previous_goal = goals[i] #this does correspond to previous goal as we start iteration over second element of goals
            previous_goal_t = arduino_data.reward_event_times[i]
            i_start_approx = np.argmin(abs(t_cv - previous_goal_t)) #this is a first guess; we will then search for the exact time
            try:
                i_start = visited_ports[i_start_approx-2:i_start_approx+3].index(previous_goal) + i_start_approx -2
            except ValueError:
                discarded_trials_alignment += 1
                continue
            
            trial = mx.Trial(maze, goal, delay_period = arduino_data.delay_period)
            if visited_ports[i_start] != previous_goal:
                raise AlgorithmFail

            # timing from Arduino should be more accurate so we use it here as step time
            
            initial_step = mx.Step(loc=visited_ports[i_start], trial=trial, time=previous_goal_t + arduino_data.delay_period)
            trial.steps.append(initial_step)
            
            #iterate over steps

            for t_arrival_at_port, loc in zip(t_cv[i_start+1:], visited_ports[i_start+1:]):
                step_count += 1
                #TODO: reorganize these messy ifs 

                if (t_arrival_at_port - trial.steps[-1].time) < step_duration_threshold:
                    # Step duration less than the threshold
                    step = mx.Step(loc, trial, previous = trial.steps[-1], time=t_arrival_at_port)
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
                            initial_step = mx.Step(loc=loc, trial=trial, time=t_arrival_at_port)
                            trial.steps.append(initial_step)

                        else:
                            #the whole trial is discarded
                            discarded_trials_has_slow_step += 1
                            break

                    else:
                        #it's the first step so it still gets included as normal
                        step = mx.Step(loc, trial, previous = trial.steps[-1], time=t_arrival_at_port)
                        trial.steps.append(step)

                if loc == goal:
                    # the animal is at the reward port
                    # now we need to check that he actually poked and got the reward:
                    trial.steps[-1].time = copy.deepcopy(t_arduino)

                    if np.abs(t_arduino - t_arrival_at_port) < jitter_tolerance:
                        trial.steps[-1].time = t_arduino # for the last step we use the arduino time 'cause it is more accurate

                        if  min_trial_duration <= trial.duration <= max_trial_duration:
                            if trial.shortened:
                                shortened_trials += 1

                            if len(trial.steps) >= 3:
                                #discard empty trials where all previous steps to goal were discarded
                                session.trials.append(trial)
                                super_sessions[mouse].trials.append(trial)
                                mega_session.trials.append(trial)

                            else:
                                discarded_trials_empty += 1
                        else:
                            discarded_trials_neg_duration += 1

                    else:
                        #the animal didn't consume reward or there is another more exotic problem
                        discarded_trials_not_consumed += 1
                    break

#----------------------------------Useful bits--------------------------------
tosec = lambda h,m,s: 3600*h + 60*m + s

def get_trajectory(timed_centroid, maze, mapping, tolerance):
    tolerance = 30

    nearest_ports = []
    visited_ports = []
    t_axis = []
    last_visited_port = None
    time_visit = 0

    for t, x, y in timed_centroid:
        nearest_port = mapping.find_nearest_port((x,y))
        nearest_ports.append(nearest_port)

        #if mapping.portregions_dic[nearest_port].contains(Point(x,y)):
        if mapping.portregions_dic[nearest_port].distance(Point(x,y)) < tolerance:
            if not last_visited_port:
                visited_ports.append(nearest_port)
                last_visited_port = nearest_port
                t_axis.append(t)
                time_visit = t

            elif nearest_port != last_visited_port:
                if nearest_port in maze.neighbors(last_visited_port):
                    #check if it is connected to the last recognized port,i.e,
                    #  is not a tracking error
                    visited_ports.append(nearest_port)
                    last_visited_port = nearest_port
                    t_axis.append(t)
                    time_visit = t

    return t_axis, visited_ports


