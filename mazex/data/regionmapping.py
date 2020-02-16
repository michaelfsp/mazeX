import re

import numpy as np
import pylab as pl

import xml.etree.ElementTree as ET
import shapely.geometry 
from descartes.patch import PolygonPatch
from PIL import Image

class PortRegion(shapely.geometry.Polygon):
    def __init__(self, poke_coord, poke_address, shell=None, holes=None):
        super(PortRegion, self).__init__(shell, holes)
        self.poke_coord = poke_coord
        self.poke_address = poke_address

    def as_patch(self, zorder=2):
        return PolygonPatch(self, facecolor='r', edgecolor='k', alpha=0.5, zorder=zorder)


class RegionMapping:
    def __init__(self, svg_file, bmp_file=None):
        self.svg_file = svg_file
        self.bmp_file = bmp_file
        self.tree = ET.parse(svg_file)
        self.root = self.tree.getroot()
        self.g = None #sometimes inkscape will place paths inside a node
        # called 'g' which groups other SVG elements in it such as the paths
        # of our polygons of interest

        self.portregions = []
        self.portregions_dic = {}
        self.address_to_coord_dic = {}

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
            self.address_to_coord_dic[poke_address] = poke_coord

    def draw_regions(self, bmp_file=None):
        fig = pl.gcf()
        ax = fig.add_subplot(111)
        if not bmp_file:
            if self.bmp_file:
                bmp_file = self.bmp_file

        if bmp_file:
            img = Image.open(bmp_file)
            pl.imshow(img, cmap='gray', vmin=0, vmax=255)

        for r in self.portregions:
            ax.add_patch(r.as_patch())

        pl.xlim(0, 2048)
        pl.ylim(2048, 0)

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

