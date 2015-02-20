import bonsai

#------------------------------------ Parameters -----------------------------

fps = 30
background_file = 'background.svg'
data_dir = '/home/michael/data/122014/cv/'
metadata_file = data_dir+'metadata2014-12-23T14_08_57.csv'
centroid_file = data_dir+'tracking2014-12-23T14_08_57.csv'


bonsai_data = bonsai.BonsaiData(centroid_file, metadata_file, fps,
        background_file)

#---------------------------------Parsing CV data-----------------------------
avg_major_axis_length = np.median(tracked[:,3])


#-----------------------------------ITIs--------------------------------------
"""For getting the reward times we get the times when the light went off,
which coincides with reward delivery, and discard the last event as that
is simply triggered by the task coming to an end.
"""
mice = ['4I', '4II', '4III', '4IV', '5I', '5II', '5III', '5IV']
experiment_days = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


#raw = np.loadtxt('/home/michael/data/122014/arduino/23122014/5III.txt')
#reward_events = raw[(raw[:,1]<100)&(raw[:,2]==0)][:-1][:,:2]
#time_to_first_reward = int(reward_events[0,0])

for m in mice:
    diff = []
    for d in experiment_days[-3:]:
        raw = np.loadtxt('/home/michael/data/122014/arduino/'+str(d)+'122014/'+m+'.txt')
        reward_events = raw[(raw[:,1]<100)&(raw[:,2]==0)][:-1][:,:2]
        reward_event_times = reward_events[:,0]/1000 #convert to seconds
        diff.extend(list(np.diff(reward_event_times)))

    pl.figure()

    pl.title(m)
    for nbins in (100, 200):
        pl.hist(diff, bins=nbins)
    pl.xlim((0,60))

pl.show()

"""Code graveyard

centroid = tracked[:,0:2]

frame_number = meta[:,1] # Bonsai gets the frame_numbers from the camera.
time = (frame_number-frame_number[0])*1000/fps # the framerate was 30fps.
time = time[:, None] # this last silly indexing expands the dimensions
# of the vector so it can be stacked with the other matrix as follows:
"""
