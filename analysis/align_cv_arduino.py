import copy
import datetime
import numpy as np
import pylab as pl
from data import RegionMapping

#----------------------------------Useful bits--------------------------------
tosec = lambda h,m,s: 3600*h + 60*m + s


#------------------------------------ Parameters -----------------------------
fps = 30


#-------------------------------Parsing Arduino data--------------------------
"""For getting the reward times we get the times when the light went off,
which coincides with reward delivery, and discard the last event as that
is simply triggered by the task coming to an end.
"""
raw = np.loadtxt('/home/michael/data/122014/arduino/23122014/5III.txt')
reward_events = raw[(raw[:,1]<100)&(raw[:,2]==0)][:-1][:,:2]
time_to_first_reward = int(reward_events[0,0])


#---------------------------------Parsing CV data-----------------------------
data_dir = '/home/michael/data/122014/cv/'

meta = np.loadtxt(data_dir+'metadata2014-12-23T14_08_57.csv', skiprows=1)
tracked = np.loadtxt(data_dir+'tracking2014-12-23T14_08_57.csv', skiprows=1)
avg_major_axis_length = np.median(tracked[:,3])
centroid = tracked[:,0:2]

frame_number = meta[:,1] # Bonsai gets the frame_numbers from the camera.
time = (frame_number-frame_number[0])*1000/fps # the framerate was 30fps.
time = time[:, None] # this last silly indexing expands the dimensions
# of the vector so it can be stacked with the other matrix as follows:
timed_centroid = np.hstack((time, centroid))



"""Now we extract the regions corresponding to the different pokes on the
image using the annotated SVG file.
"""
mapping = RegionMapping('./mazex/analysis/background.svg')

poke_xy_dic = mapping.get_region_centroid_xy()

reward_events_xy = np.zeros((reward_events.shape[0], 2))

for i, re in enumerate(reward_events):
    poke_address = re[1]
    reward_events_xy[i] = poke_xy_dic[poke_address]
    

candidate_indices = reward_events[:,0]/(1000/fps)
# indexing requires these to be integers
candidate_indices = candidate_indices.round().astype('int')
candidate_indices -= candidate_indices[0] #so that when tau=0,
#  candidate_indices[0] is also zero

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
    

"""Calculate distance for every tau (shift).
"""
recording_length = timed_centroid[-1,0]

tau_min = reward_events[0,0]
tau_res = 33.3333 #ms
tau_max = (recording_length - reward_events[-1,0]) - np.ceil(tau_res)
delta_index = round((1000/fps)/tau_res) # by how much should the indexes be
# incremented to achieve delta. fps is the framerate

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

video_dir = '/home/michael/data/122014/video/' 


pl.show()
