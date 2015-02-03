import copy
import numpy as np
from data import RegionMapping

#------------------------------------ Parameters -----------------------------
fps = 30

"""For getting the reward times we get the times when the light went off,
which coincides with reward delivery, and discard the last event as that
is simply triggered by the task coming to an end.
"""

raw = np.loadtxt('/home/michael/data/122014/arduino/23122014/4IV.txt')
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
time = (frame_number-frame_number[0])*1000/fps # the framerate was 30fps.
time = time[:, None] # this last silly indexing expands the dimensions
# of the vector so it can be stacked with the other matrix as follows:
timed_centroid = np.hstack((time, centroid))



#def RMS(timed_centroi

def get_region_centroid_xy(mapping):
    dic = {}
    for portregion in mapping.portregions:
        # shapely has a strange way of returning the coordinates so we first
        #  store the information in a tuple
        centroid_xy_tuple = portregion.centroid.xy
        x = centroid_xy_tuple[0][0]
        y = centroid_xy_tuple[1][0]

        dic[portregion.poke_address] = (x,y)

    return dic

reward_events_xy = np.zeros((reward_events.shape[0], 2))

port_xy_dic = get_region_centroid_xy(mapping)

for i, re in enumerate(reward_events):
    reward_events_xy[i] = port_xy_dic[re[1]]
    
candidate_indices = np.zeros(reward_events.shape[0], dtype=int) #dtype has
# to be int or else numpy won't accept these as indexes

for i,r in enumerate(reward_events):
    candidate_indices[i] = np.argmin(abs(timed_centroid[:,0]-r[0]))

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
session_length = reward_events[-1,0] - reward_events[0,0]

tau_min = 0
tau_res = 33.3333 #ms
tau_max = (recording_length - session_length) - np.ceil(tau_res)
delta_index = round((1000/fps)/tau_res) # by how much should the indexes be
# incremented to achieve delta. fps is the framerate

total_steps = np.floor(tau_max/tau_res)# - delta_index

score_vs_tau = np.zeros((total_steps, 2))
n_steps = 0
tau = tau_min
while n_steps < total_steps:
    score_vs_tau[n_steps] = (tau, score_tau(tau, fps, timed_centroid,
        candidate_indices, reward_events_xy))
    n_steps += 1
    tau += tau_res
    if n_steps%1000 ==0:
        print(n_steps)

"""Test tau.
"""

video_dir = '/home/michael/data/122014/video/' 
