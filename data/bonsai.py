"""Code to analyse data acquired from the setup.  """
import numpy as np
import pylab as pl

import xml.etree.ElementTree as ET
import shapely.geometry 
import re
import copy
import datetime

try:
    import prettyplotlib as ppl
except ImportError:
    print("Prettyplotlib not available. Falling back onto Matplotlib.")


class BonsaiData:
    def __init__(self, centroid_file, metadata_file, fps=30,
            background_file=None, background_PNG=None):
        """Container of data files created by CV in Bonsai.
        """
        self.centroid_file = centroid_file
        self.medatata_file = metadata_file
        self.fps = fps
        self.background_file = background_file
        self.background_PNG = background_PNG

        self.raw_array = np.loadtxt(self.centroid_file, skiprows=1)
        # it seems like we don't need this anymore
        #self.raw_array = np.where(raw_array==0, None, raw_array)
        self.xy = self.raw_array[:,0:2]
        self.major_axis_length = self.raw_array[:,3]

        self.raw_metadata = np.loadtxt(metadata_file, skiprows=1)
        # Bonsai gets the frame_numbers from the camera.
        self.frame_numbers = self.raw_metadata[:,1]
        self.time_axis = (self.frame_numbers-self.frame_numbers[0])*1000/self.fps 
        # this last silly indexing expands the dimensions
        #of the vector so it can be stacked with the other matrix as follows:
        self.time_axis = self.time_axis[:, None]

        #TODO: think of a way of not duplicating the data...
        self.timed_centroid = np.hstack((self.time_axis, self.xy))

        self.region_mapping = RegionMapping(self.background_file)
        self.poke_xy_dic = self.region_mapping.get_region_centroid_xy()

    def skipped_frames(self):
        """Show where and how many frames were skipped.
        #TODO
        """
    def plot(self, picture=True):
        pl.plot(self.xy[:,0], self.xy[:,1]) 

        if self.background_PNG:
            pl.imshow(pl.imread(self.background_PNG))

    def to_poke_ports(self, mapping):
        list_ports_transversed = []
        for xy in self.raw_array:
            # the following is a bit hacky but it works
            if np.any(xy == [None,None]):
                list_ports_transversed.append(None)
            else:
                #here determine to which port point is nearest
                list_ports_transversed.append(mapping.find_nearest_port(xy))

        return list_ports_transversed

    def score_tau(self, tau, candidate_indices, reward_events_xy):
        """Gives a score to the tested tau value.

        Parameters:
        -----------
        tau : int
            Value of the shift in ms

        reward_events : numpy.array
            Numpy array containing time of poke (ms) and poke port address

        mapping : RegionMapping object
            Contains the mapping of poke port addresses to their X,Y coordinates
            on the image/video.
        """
        # we find the indices on the timed_centroid array that correspond to the event
        #  times in the reward_events array

        tau_index_shift = round((tau/1000)*self.fps)
           
        re_indexed = copy.deepcopy(reward_events_xy)
        re_indexed = re_indexed.astype('int')
        re_indexed[:,0] = candidate_indices + tau_index_shift

        centroid_xy_coord = self.timed_centroid[re_indexed[:,0], 1:3]
        
        return np.linalg.norm(centroid_xy_coord - reward_events_xy)

    def find_shift(self, arduino_data_object, tau_min=None,
            tau_max=None, tau_res=None, plot=False, verbose=False):
        """Calculate distance for every tau (shift).
        """
        reward_events = arduino_data_object.reward_events
        reward_events_xy = np.zeros((reward_events.shape[0], 2))

        time_to_first_reward = reward_events[0,0]

        for i, re in enumerate(reward_events):
            poke_address = re[1]
            reward_events_xy[i] = self.poke_xy_dic[poke_address]

        candidate_indices = reward_events[:,0]/(1000/self.fps)
        # indexing requires these to be integers
        candidate_indices = candidate_indices.round().astype('int')
        candidate_indices -= candidate_indices[0] #so that when tau=0,
        #  candidate_indices[0] is also zero

        recording_length = self.timed_centroid[-1,0]

        tau_min = reward_events[0,0]
        tau_res = 33.3333 #ms
        tau_max = (recording_length - reward_events[-1,0]) - np.ceil(tau_res)

        total_steps = np.floor((tau_max-tau_min)/tau_res)# - delta_index

        score_vs_tau = np.zeros((total_steps, 2))
        n_steps = 0
        tau = tau_min
        while n_steps < total_steps:
            score_vs_tau[n_steps] = (tau, self.score_tau(tau, 
                candidate_indices, reward_events_xy))

            if ((n_steps/total_steps)*100)%10 < 0.0001:
                print(int((n_steps/total_steps)*100), '%')

            n_steps += 1
            tau += tau_res

        avg_dot_product = score_vs_tau.mean(axis=0)[1]
        abs_min_dot_product = np.min(score_vs_tau, axis=0)[1]
        mean_min_difference = abs_min_dot_product - avg_dot_product
        # we define the threshold as 90% distance to minimum from the mean
        #threshold = avg_dot_product - 0.9*(avg_dot_product - abs_min_dot_product)

        avg_major_axis_length = np.median(self.major_axis_length)

        residual_norm = np.linalg.norm((avg_major_axis_length/2)*np.ones(
            reward_events_xy.shape[0]))
        threshold = abs_min_dot_product + residual_norm

        if plot:
            pl.figure()
            pl.plot(score_vs_tau[:,0], score_vs_tau[:,1])
            xlim = pl.xlim()
            pl.hlines(threshold, *xlim)

        index_first_point_below_threshold = np.nonzero(score_vs_tau < threshold)[0][0]
        best_tau = score_vs_tau[:,0][index_first_point_below_threshold]

        adjusted_tau = best_tau - time_to_first_reward

        if verbose:
            print('time of first reward: ', datetime.timedelta(seconds=best_tau/1000))
            print('data acquisition start: ', datetime.timedelta(seconds=adjusted_tau/1000))

            print('\nTimes of reward events\n')
            for r in reward_events[:,0]:
                print(datetime.timedelta(seconds=(adjusted_tau + r)/1000))

        return adjusted_tau#this is when acquisition started

    def get_time_slice(self, t_start, t_stop):
        i_start = np.argmin(abs(self.time_axis - t_start))
        i_stop = np.argmin(abs(self.time_axis - t_stop))

        return copy.deepcopy(self.timed_centroid[i_start:i_stop])


class PortRegion(shapely.geometry.Polygon):
    def __init__(self, poke_coord, poke_address, shell=None, holes=None):
        super(PortRegion, self).__init__(shell, holes)
        self.poke_coord = poke_coord
        self.poke_address = poke_address


class RegionMapping:
    def __init__(self, svg_file):
        self.svg_file = svg_file
        self.tree = ET.parse(svg_file)
        self.root = self.tree.getroot()
        self.g = None #sometimes inkscape will place paths inside a node
        # called 'g' which groups other SVG elements in it such as the paths
        # of our polygons of interest

        self.portregions = []
        self.portregions_dic = {}

        # here we will detect the polygons and create PortRegions from them
        list_paths = self.root.findall('{http://www.w3.org/2000/svg}path')

        if len(list_paths) == 0 or self.root.find('{http://www.w3.org/2000/svg}g'):
            self.g = self.root.find('{http://www.w3.org/2000/svg}g')
            list_paths = self.g.findall('{http://www.w3.org/2000/svg}path')

        for x in list_paths: #each polygon is a path in SVG
            #gets a string that contains coordinates of vertices plus stuff
            #  we don't need
            coords_str = x.get('d')
            # this will extract only the coordinates and discard what we
            #  don't need
            coords_list = re.findall(r"[-+]?\d*\.\d+|\d+", coords_str)
            list_coord_tuples = []
            
            # because of the re processing above the x and y coordinates are
            #  serialized. here we group them again to retrieve the
            #  coordinates by point
            iter_coords_list = iter(coords_list)
            for xy_tuple in zip(iter_coords_list, iter_coords_list):
                list_coord_tuples.append((float(xy_tuple[0]), float(xy_tuple[1])))

            poke_coord = eval(x.get(
                '{http://www.inkscape.org/namespaces/inkscape}label'))
            poke_address = int(x.get('id'))
            region = PortRegion(poke_coord, poke_address, list_coord_tuples)

            self.portregions.append(region)
            self.portregions_dic[poke_coord] = region


    def find_nearest_port(self, xy_tuple):
        point = shapely.geometry.Point(xy_tuple)

        distances = []
        for port in self.portregions:
            distances.append(point.distance(port))

        nearest_port_index = np.argmin(distances)

        return self.portregions[nearest_port_index].poke_coord

    def get_region_centroid_xy(self):
        """Returns a dictionary with poke_address and (x,y) coordinates for of
        that poke's centroid.
        """
        dic = {}
        for portregion in self.portregions:
            # shapely has a strange way of returning the coordinates so we first
            #  store the information in a tuple
            centroid_xy_tuple = portregion.centroid.xy
            x = centroid_xy_tuple[0][0]
            y = centroid_xy_tuple[1][0]

            dic[portregion.poke_address] = (x,y)

        return dic
