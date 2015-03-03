import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D 
import mazex as mx

data_folder = '/home/michael/data/122014/'
experiment_days = [15, 16, 17, 18, 19, 20, 21, 22, 23]
experiment_dates = []
for d in experiment_days:
    experiment_dates.append(str(d)+'122014')

mice = ['4I', '4II', '4III', '4IV', '5I', '5II', '5III', '5IV']

#s1 = mx.analysis.datamodel.Subject('4I', data_folder, experiment_dates)

#s1.ITI_dist()

s = {} #subjects
hist = {} #histogram results
"""
for m in mice:
    s[m] = mx.analysis.datamodel.Subject(m, data_folder, experiment_dates)
    pl.figure()
    hist[m] = s[m].ITI_dist(nbins=20)
"""

fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
z = 1
colors = pl.cm.rainbow(np.linspace(0,1,len(mice)))
for m,c in zip(mice, colors):
    s[m] = mx.analysis.datamodel.Subject(m, data_folder, experiment_dates)
    hist[m] = s[m].ITI_dist(nbins=20)
    ax.bar(hist[m][1][:-1], hist[m][0]/max(hist[m][0]), zs=z, zdir='y', width=0.2, color=c,
        alpha=1)
    z+=2

#TODO: do this plot just stacking them over vertically

pl.show()
