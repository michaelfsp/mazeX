import numpy as np

data_dir = '/home/michael/data/122014/cv/'

meta = np.loadtxt(data_dir+'metadata2014-12-23T14_08_57.csv', skiprows=1)
tracked = np.loadtxt(data_dir+'tracking2014-12-23T14_08_57.csv', skiprows=1)

centroid = tracked[:,0:2]

frame_number = meta[:,1] # Bonsai gets the frame_numbers from the camera.
time = (frame_number-frame_number[0])*1000/30 # the framerate was 30fps.
time = time[:, None] # this last silly indexing expands the dimensions
# of the vector so it can be stacked with the other matrix as follows:
timed_centroid = np.hstack((centroid, time))
