import copy
import numpy as np
from data import RegionMapping

"""For getting the reward times we get the times when the light went off,
which coincides with reward delivery, and discard the last event as that
is simply triggered by the task coming to an end.
"""

raw = np.loadtxt('/home/michael/data/122014/arduino/23122014/4I.txt')
reward_events = raw[(raw[:,1]<100)&(raw[:,2]==0)][:-1][:,:2]



"""Now we extract the regions corresponding to the different ports on the
image using the annotated SVG file.
"""

mapping = RegionMapping('./mazex/analysis/background.svg')



"""
We extract the CV data.
"""

data_dir = '/home/michael/data/122014/cv/'

meta = np.loadtxt(data_dir+'metadata2014-12-23T14_08_57.csv', skiprows=1)
tracked = np.loadtxt(data_dir+'tracking2014-12-23T14_08_57.csv', skiprows=1)

centroid = tracked[:,0:2]

frame_number = meta[:,1] # Bonsai gets the frame_numbers from the camera.
time = (frame_number-frame_number[0])*1000/30 # the framerate was 30fps.
time = time[:, None] # this last silly indexing expands the dimensions
# of the vector so it can be stacked with the other matrix as follows:
timed_centroid = np.hstack((time, centroid))



"""Calculate distance for every tau (shift).
"""
recording_length = timed_centroid[-1,0]
trial_length = reward_events[-1,0]

tau_min = 0
tau_max = recording_length - trial_length
delta = 1000

distances_to_ports = np.zeros(reward_events.shape)

re_tau = copy.deepcopy(reward_events)

n_steps = 0
total_steps = np.floor(tau_max/delta)
"""
def RMS_distance(timed_centroid, reward_events, mapping):
    



while n_steps < total_steps:
    re_tau[:,0] += delta 
"""
tau = 0 #the shift parameter in ms to be optimized


"""Test tau.
"""

video_dir = '/home/michael/data/122014/video/' 
