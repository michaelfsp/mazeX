import os.path

import numpy as np
import pylab as pl

"""Here come the objects that unite Arduino and Bonsai data.
"""
class Experiment:
    def __init__(self, data_folder, subject_ids, dates, maze=None):
        self.subjects_ids = subjects_ids
        self.dates = dates
        self.maze = maze
        self.data_folder = data_folder

        """Here we should populate the objects that it contains"""


class Subject:
    def __init__(self, ID, data_folder, experiment_days):
        """This container should be independent of its container, experiment"""
        self.ID = ID
        self.data_folder = data_folder
        self.experiment_days = experiment_days

        self.raw = {}
        for d in experiment_days:
            file_path = os.path.join(data_folder, 'arduino', d, ID+'.txt')
            self.raw[d] = np.loadtxt(file_path)

        self.reward_events = {}
        for d in experiment_days:
            self.reward_events[d] = self.raw[d][(self.raw[d][:,1]<100)&(self.raw[d][:,2]==0)][:-1][:,:2]

    def ITI_dist(self, experiment_days=None, logscale=True, nbins=25):
        if experiment_days:
            experiment_days = experiment_days
        else:
            experiment_days = self.experiment_days

        reward_event_times = {}
        diff = []

        for d in experiment_days:
            reward_event_times[d] = self.reward_events[d][:,0]/1000 # convert to seconds
            diff.extend(list(np.diff(reward_event_times[d])))

        pl.figure()

        pl.title('ITIs ' + self.ID)
        if logscale:
            pl.hist(np.log(diff), bins=nbins)
            pl.xlabel('log(ITI)/s')
            pl.xlim((0,7))
        else:
            pl.hist(diff, bins=nbins)
            pl.xlabel('ITI/s')


class CombinedData:
    def __init__(self, fps):#here the stuff that initializes bonsai objects plus the
        #stuff that initalizes Arduino objects plus parameters like fps etc
        pass

    def score_tau(tau, fps, timed_centroid, candidate_indices, reward_events_xy):
        """Gives a score to the tested tau value.

        Parameters:
        -----------

        tau : int
            Value of the shift in ms

        fps : float
            Frames per second of the acquired video signal

        timed_centroid : numpy.array
            Numpy array containing time (ms), centroid X and Y coordinates

        reward_events : numpy.array
            Numpy array containing time of poke (ms) and poke port address

        mapping : RegionMapping object
            Contains the mapping of poke port addresses to their X,Y coordinates
            on the image/video.
        """
        # we find the indices on the timed_centroid array that correspond to the event
        #  times in the reward_events array

        tau_index_shift = round((tau/1000)*fps)
           
        re_indexed = copy.deepcopy(reward_events)
        re_indexed = re_indexed.astype('int')
        re_indexed[:,0] = candidate_indices + tau_index_shift

        centroid_xy_coord = timed_centroid[re_indexed[:,0], 1:3]
        
        return np.linalg.norm(centroid_xy_coord - reward_events_xy)

    def find_shift(timed_centroid, candidate_indices, tau_min=None,
            tau_max=None, tau_res=None, fps=30):
        """Calculate distance for every tau (shift).
        """
        reward_events_xy = np.zeros((reward_events.shape[0], 2))

        for i, re in enumerate(reward_events):
            poke_address = re[1]
            reward_events_xy[i] = poke_xy_dic[poke_address]

        candidate_indices = reward_events[:,0]/(1000/fps)
        # indexing requires these to be integers
        candidate_indices = candidate_indices.round().astype('int')
        candidate_indices -= candidate_indices[0] #so that when tau=0,
        #  candidate_indices[0] is also zero


        recording_length = timed_centroid[-1,0]

        tau_min = reward_events[0,0]
        tau_res = 33.3333 #ms
        tau_max = (recording_length - reward_events[-1,0]) - np.ceil(tau_res)

        total_steps = np.floor((tau_max-tau_min)/tau_res)# - delta_index


        score_vs_tau = np.zeros((total_steps, 2))
        n_steps = 0
        tau = tau_min
        while n_steps < total_steps:
            score_vs_tau[n_steps] = (tau, score_tau(tau, fps, timed_centroid,
                candidate_indices, reward_events_xy))

            if ((n_steps/total_steps)*100)%10 < 0.0001:
                print(int((n_steps/total_steps)*100), '%')

            n_steps += 1
            tau += tau_res

        """Test tau.
        """
        pl.figure()
        pl.plot(score_vs_tau[:,0], score_vs_tau[:,1])

        avg_dot_product = score_vs_tau.mean(axis=0)[1]
        abs_min_dot_product = np.min(score_vs_tau, axis=0)[1]
        mean_min_difference = abs_min_dot_product - avg_dot_product
        # we define the threshold as 90% distance to minimum from the mean
        #threshold = avg_dot_product - 0.9*(avg_dot_product - abs_min_dot_product)

        residual_norm = np.linalg.norm((avg_major_axis_length/2)*np.ones(
            reward_events_xy.shape[0]))
        threshold = abs_min_dot_product + residual_norm

        index_first_point_below_threshold = np.nonzero(score_vs_tau < threshold)[0][0]
        best_tau = score_vs_tau[:,0][index_first_point_below_threshold]

        adjusted_tau = best_tau - time_to_first_reward

        print('time of first reward: ', datetime.timedelta(seconds=best_tau/1000))
        print('data acquisition start: ', datetime.timedelta(seconds=adjusted_tau/1000))

        print('\nTimes of reward events\n')
        for r in reward_events[:,0]:
            print(datetime.timedelta(seconds=(adjusted_tau + r)/1000))
