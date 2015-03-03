import pylab as pl
import mazex as mx

data_folder = '/home/michael/data/122014/'
experiment_days = [15, 16, 17, 18, 19, 20, 21, 22, 23]
experiment_dates = []
for d in experiment_days:
    experiment_dates.append(str(d)+'122014')

mice = ['4I', '4II', '4III', '4IV', '5I', '5II', '5III', '5IV']

#s1 = mx.analysis.datamodel.Subject('4I', data_folder, experiment_dates)

#s1.ITI_dist()

s = {}

for m in mice:
    s[m] = mx.analysis.datamodel.Subject(m, data_folder, experiment_dates)
    s[m].ITI_dist(nbins=20)

pl.show()
