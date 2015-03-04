"""Code to analyse data acquired from the setup.  """
import numpy as np
import pylab as pl

import xml.etree.ElementTree as ET
import shapely.geometry 
import re

try:
    import prettyplotlib as ppl
except ImportError:
    print("Prettyplotlib not available. Falling back onto Matplotlib.")


class BonsaiData:
    def __init__(self, centroid_file, metadata_file, fps=30,
            background_file=None):
        """Container of data files created by CV in Bonsai.
        """
        self.centroid_file = centroid_file
        self.medatata_file = metadata_file
        self.fps = fps
        self.background_file = background_file

        self.raw_array = np.loadtxt(self.centroid_file, skiprows=1)
        # it seems like we don't need this anymore
        #self.raw_array = np.where(raw_array==0, None, raw_array)
        self.xy = self.raw_array[:,0:2]
        self.major_axis_length = self.raw_array[:,3]

        self.raw_metadata = np.loadtxt(metadata_file, skiprows=1)
        # Bonsai gets the frame_numbers from the camera.
        self.frame_numbers = self.raw_metadata[:,1]
        self.time_axis = (frame_numbers-frame_numbers[0])*1000/fps 
        # this last silly indexing expands the dimensions
        #of the vector so it can be stacked with the other matrix as follows:
        self.time_axis = self.time_axis[:, None]

        #TODO: think of a way of not duplicating the data...
        self.timed_centroid = np.hstack((self.time_axis, self.xy))

    def skipped_frames(self):
        """Show where and how many frames were skipped.
        #TODO
        """
    def plot(self, picture=True):
        pl.plot(self.xy[:,0], self.xy[:,1]) 
        pl.imshow(pl.imread(self.background_file))

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

            self.portregions.append(PortRegion(poke_coord,
                poke_address, list_coord_tuples))

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
