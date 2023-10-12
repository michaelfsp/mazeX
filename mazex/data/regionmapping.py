import re

import numpy as np
import pylab as pl

import xml.etree.ElementTree as ET

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import shapely.geometry

from descartes.patch import PolygonPatch
from PIL import Image

import scipy.optimize as optimize
import functools

import svgpath2mpl


# TODO: move these functions into the class

def general_lens_correction(xy, k, f, l=0.75, xy_c=(1023.5, 1023.5),
                            XY_c=(0.6, 0.6)):
    '''This function computes the correction for any type of lens. The fisheye
    factor k modifies the type of distortion / correction.
    see PTGui 11 fisheye in https://wiki.panotools.org/Fisheye_Projection

    xy are pixel coordinates on the image in px
    XY are real world coordinates in m

    xy_c are the coordinates of the center of the image in px
    XY_c are the coordinates of the center of the real world (i.e. the setup)

    l = 0.75 is default because the lens was exactly 0.75 m from the platforms

    k has to be > 1449 or else np.arctan would return an invalid number

    k should be between 0 and 1
    '''
    x, y = xy

    # we first center x and y
    x_c = x - xy_c[0]
    y_c = y - xy_c[1]

    # now we convert to polar coordinates in the image
    r = np.sqrt(x_c**2 + y_c**2)
    theta = np.arctan2(y_c, x_c)

    # R is the real world polar coordinate R
    if -1 <= k and k < 0:
        R = l * np.tan((-np.arcsin((k*r)/f)+np.pi)/k)
    elif 0 < k and k <= 1:
        R = l * np.tan(np.arctan((k*r)/f)/k)
    elif k == 0:
        R = l * np.tan(r/f)
    else:
        raise Exception('invalid k value. k must be a number between -1 and 1')

    # now we calculate the real world centered XY coordinates
    X = R * np.cos(theta)
    Y = R * np.sin(theta)

    # now we put it back into axes with the origin in the corner of the real
    # world, i.e. decenter again
    X += XY_c[0]
    Y += XY_c[1]
    return (X, Y)


##
# Discover correction coefficients for the lens
def loss_function(k, f, platform_centroid_px_coords_dic):
    '''This loss function is what we will minimize over r to find the correct
    radius of the imaginary sphere that undistorts the images.
    We will compute straightness of horizontal and vertical lines by taking
    the sum squared distance of each point in the line to the middle of it.
    We then take the sum accross all vertical and horizontal lines.
    '''
    # We know that the center of the outermost tower is at (15cm, 15cm).
    # since the towers are 10cm wide and the corridors 8cm long, then we know
    # the center coordinates for all towers in a row as 15 + n * 18
    # We are considering a square of 120cm * 120cm which is the size of the
    # maze
    #
    # The Y axis is "flipped", i.e., Y_(:,6) < Y_(:,5) < ... < Y_(:0) but the
    # X is not, i.e. X_(:,6) > X_(:,5) > ... > X_(:0) but the
    xy_coord_m = lambda i, j: (0.15 + i * 0.18, 1.2 - (0.15 + j * 0.18))

    # we initialize the Root Mean Square Error, which will be
    # RMSE = np.sqrt((1/36) * \Sum{Euclidian error**2})

    MSE = 0
    for i in range(6):
        # dim==0 is the x axis; dim==1 is the y axis
        for j in range(6):
            # we then get the y coord of the centroid of each port for rows and
            # the x coord of the centroid of each port for columns
            xy = platform_centroid_px_coords_dic[(i, j)]

            projected_xy = general_lens_correction(xy, k, f)

            MSE += np.sqrt((xy_coord_m(i, j)[0]-projected_xy[0])**2 + (
                xy_coord_m(i, j)[1]-projected_xy[1])**2)

    RMSE = np.sqrt(MSE/36)

    return RMSE


def find_optimal_params_correction(platform_centroid_px_coords, k_init=0.5,
                                   f_init=1500):
    partial_loss_function = functools.partial(
        loss_function, platform_centroid_px_coords_dic=platform_centroid_px_coords)
    f = lambda kf: partial_loss_function(kf[0], kf[1])

    initial_guess = np.array([[k_init], [f_init]])
    result = optimize.minimize(f, initial_guess, bounds=((0,0.999),(0, None)))
    k_optimal, f_optimal = result.x

    return k_optimal, f_optimal


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

        global_translate = (0, 0)

        if len(list_paths) == 0 or self.root.find('{http://www.w3.org/2000/svg}g'):
            self.g = self.root.find('{http://www.w3.org/2000/svg}g')
            list_paths = self.g.findall('{http://www.w3.org/2000/svg}path')
            global_translate_str = self.g.get('transform')
            global_translate = eval(global_translate_str.lstrip('translate'))

        for x in list_paths: #each polygon is a path in SVG
            #gets a string that contains coordinates of vertices plus stuff
            #  we don't need
            coords_str = x.get('d')
            ## this will extract only the coordinates and discard what we
            ##  don't need
            #coords_list = re.findall(r"[-+]?\d*\.\d+|\d+", coords_str)
            #list_coord_tuples = []

            ## because of the re processing above the x and y coordinates are
            ##  serialized. here we group them again to retrieve the
            ##  coordinates by point
            #iter_coords_list = iter(coords_list)
            #for xy_tuple in zip(iter_coords_list, iter_coords_list):
            #    list_coord_tuples.append((float(xy_tuple[0]), float(xy_tuple[1])))

            #list_coord_tuples_translated = []
            #for t in list_coord_tuples:
            #    list_coord_tuples_translated.append((t[0]+global_translate[0], t[1]+global_translate[1]))

            #import ipdb; ipdb.set_trace()

            parsed_path = svgpath2mpl.parse_path(coords_str)
            path_coords = parsed_path.to_polygons()[0]
            path_coords += np.array([global_translate[0], global_translate[1]])
            list_coord_tuples_translated = path_coords.tolist()

            poke_coord = eval(x.get(
                '{http://www.inkscape.org/namespaces/inkscape}label'))
            poke_address = int(x.get('id'))
            region = PortRegion(poke_coord, poke_address, list_coord_tuples_translated)

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

    def get_region_centroid_xy(self, loc_coords=False):
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

            if loc_coords:
                dic[self.address_to_coord_dic[portregion.poke_address]] = (x,y)
            else:
                dic[portregion.poke_address] = (x,y)

        return dic

