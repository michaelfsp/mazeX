"""Code to analyse data acquired from the setup.  """
import numpy as np
import pylab as pl
import xml.etree.ElementTree as ET
import shapely.geometry 

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

class PortRegion(shapely.geometry.Polygon):
    def __init__(self, poke_coord, shell=None, holes=None):
        super(PortRegion, self).__init__(shell, holes)
        self.poke_coord = poke_coord

class Mapping:
    def __init__(self, svg_file):
        self.svg_file = svg_file
