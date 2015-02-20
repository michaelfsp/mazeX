import copy
import datetime
import numpy as np
import pylab as pl
from data import RegionMapping

#----------------------------------Useful bits--------------------------------
tosec = lambda h,m,s: 3600*h + 60*m + s


"""Now we extract the regions corresponding to the different pokes on the
image using the annotated SVG file.
"""
mapping = RegionMapping('./mazex/analysis/background.svg')

poke_xy_dic = mapping.get_region_centroid_xy()



