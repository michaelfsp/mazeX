import numpy as np
import xml.etree.ElementTree as ET

data_dir = '/home/michael/data/122014/cv/'
video_dir = '/home/michael/data/122014/video/'

meta = np.loadtxt(data_dir+'metadata2014-12-23T14_08_57.csv', skiprows=1)
tracked = np.loadtxt(data_dir+'tracking2014-12-23T14_08_57.csv', skiprows=1)
vid_markers = ET.parse(video_dir+'23122014.mlt')

centroid = tracked[:,0:2]

frame_number = meta[:,1] # Bonsai gets the frame_numbers from the camera.
time = (frame_number-frame_number[0])*1000/30 # the framerate was 30fps.
time = time[:, None] # this last silly indexing expands the dimensions
# of the vector so it can be stacked with the other matrix as follows:
timed_centroid = np.hstack((centroid, time))


"""
Code to extract the position of different video segments as manually annotated
on Openshot.
"""

root = vid_markers.getroot()
child = root.getchildren()[0]
multitrack = child.find('multitrack')
tracks = multitrack.getchildren()

start_stop = {}

for track in tracks:
    if track.get('id') == 'Background Track':
        pass
    elif track.get('id') == 'junk':
        pass
    else:
        track_id = track.get('id')
        start = track.find('producer').get('in')
        stop = track.find('producer').get('out')

        start_stop[track_id] = (int(start), int(stop))

tosec = lambda h,m,s: 3600*h + 60*m + s

epsilon_23 = 1.0201718872078982
