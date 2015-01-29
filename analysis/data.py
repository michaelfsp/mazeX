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

class BonsaiFile:
    def __init__(self, filename, picture_filename=None):
        self.filename = filename
        array = np.loadtxt(self.filename, skiprows=1)
        self.array = np.where(array==0, None, array)
        
        if picture_filename:
            self.picture_filename = picture_filename

    def plot(self, picture=True):
       ppl.plot(self.array[:,0], self.array[:,1]) 
       pl.imshow(pl.imread(self.picture_filename))

    def to_poke_ports(self, mapping):
        list_ports_transversed = []
        for xy in self.array:
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




