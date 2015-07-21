import copy
import os

import numpy as np
import pylab as pl
from scipy.signal import butter, lfilter, filtfilt

from mpl_toolkits.mplot3d import Axes3D 

import mazex as mx

from shapely.geometry import Point

#------------------------------------ Parameters -----------------------------
fps = 30
background_file = 'figures/stills/background.svg'
background_PNG = 'figures/stills/background.bmp'
data_folder = '/home/michael/data/122014/'
experiment_days = [15, 16, 17, 18, 19, 20, 21, 22, 23]
mice = ['4I', '4II', '4III', '4IV', '5I', '5II', '5III', '5IV']

#---------------------------------Maze Configuration--------------------------
maze = mx.SquareMaze(6)
h_connections = [[(0,0), (1,0)], [(1,0), (2,0)], [(2,0), (3,0)], [(3,0), (4,0)], 
        [(3,1), (4,1)],
        [(0,2), (1,2)], [(2,2), (3,2)], [(3,2), (4,2)], [(4,2), (5,2)],
        [(1,3), (2,3)], [(3,3), (4,3)], 
        [(0,4), (1,4)], [(2,4), (3,4)], [(3,4), (4,4)], [(4,4), (5,4)],
        [(0,5), (1,5)], [(1,5), (2,5)], [(2,5), (3,5)], [(4,5), (5,5)]]

v_connections = [[(0,0), (0,1)], [(0,1), (0,2)], [(0,3), (0,4)], [(0,4), (0,5)],
        [(1,1), (1,2)], [(1,2), (1,3)], [(1,3), (1,4)],
        [(2,0), (2,1)], [(2,2), (2,3)], [(2,3), (2,4)], [(2,4), (2,5)],
        [(3,0), (3,1)], [(3,3), (3,4)],
        [(4,1), (4,2)], [(4,2), (4,3)], [(4,4), (4,5)],
        [(5,0), (5,1)], [(5,1), (5,2)], [(5,2), (5,3)], [(5,3), (5,4)]]

maze.add_edges_from(h_connections)
maze.add_edges_from(v_connections)

#----------------------------------Useful bits--------------------------------
tosec = lambda h,m,s: 3600*h + 60*m + s

#---------------------------------Parsing CV data-----------------------------
"""Now we extract the regions corresponding to the different pokes on the
image using the annotated SVG file.
"""
d = '20122014'
centroid_file = os.path.join(data_folder, 'cv', d, 'tracking.csv')
metadata_file = os.path.join(data_folder, 'cv', d, 'metadata.csv')

arduino_file = os.path.join(data_folder, 'arduino', d, '4I.txt')

#mapping = mx.data.bonsai.RegionMapping(background_file)
bonsai_data = mx.data.bonsai.BonsaiData(centroid_file, metadata_file, fps,
        background_file, background_PNG, maze=maze)

poke_xy_dic = bonsai_data.region_mapping.get_region_centroid_xy()

arduino_data = mx.data.arduino.ArduinoDataFile(arduino_file)

t_start = bonsai_data.find_shift(arduino_data)

# the time of last reward, not of end of session!
t_stop = arduino_data.reward_event_times[-1] 

xy_session =  bonsai_data.get_time_slice(t_start, t_stop)
xy_session[:,0] -= t_start

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

t, p = bonsai_data.get_trajectory(t_start, t_stop)

start_port = p[0]

magent = mx.OmniscientAgent(start_port, maze, heading = '^')

goals = [bonsai_data.region_mapping.address_to_coord_dic[poke_address] for poke_address in arduino_data.reward_events[:,1]]
mtrials = []
for goal in goals:
    initial_step = mx.SimStep(magent.loc)
    mtrial = mx.SimTrial(maze, goal, initial_step, magent)
    mtrial.solve()
    print('Solved trial %i' %len(mtrials))
    mtrials.append(mtrial)

msession = mx.Session()
msession.trials.extend(mtrials)
#pl.figure()
#mx.analysis.information.per_dist_psi(msession, plot=True, first_exit_only=True)

total_steps_omniscient_agent = 0
for t in msession.trials:
    total_steps_omniscient_agent += len(t.steps)

pl.show()
