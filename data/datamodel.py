import os.path

import numpy as np
import pylab as pl

from mazex.data.arduino import ArduinoDataFile

"""Here come the objects that unite Arduino and Bonsai data.
"""
class Experiment:
    def __init__(self, data_folder, subject_ids, dates, maze=None):
        self.subjects_ids = subjects_ids
        self.dates = dates
        self.maze = maze
        self.data_folder = data_folder

        """Here we should populate the objects that it contains"""

class Subject:
    def __init__(self, ID, data_folder, experiment_days):
        """This container should be independent of its container, experiment"""
        self.ID = ID
        self.data_folder = data_folder
        self.experiment_days = experiment_days

        self.raw_arduino = {}
        for d in experiment_days:
            arduino_file_path = os.path.join(data_folder, 'arduino', d, ID+'.txt')
            self.raw_arduino[d] = ArduinoDataFile(arduino_file_path)

        self.raw_coordinates = {}
        #for d in experiment_days:
        #   extract only the relevant coordinates to here


    def ITI_dist(self, experiment_days=None, logscale=True, nbins=25):
        if experiment_days:
            experiment_days = experiment_days
        else:
            experiment_days = self.experiment_days

        diff = []
        for d in experiment_days:
            diff.extend(list(self.raw_arduino[d].ITIs/1000)) #/1000 to convert to seconds

        pl.title('ITIs ' + self.ID)
        if logscale:
            pl.xlabel('log(ITI)/s')
            pl.xlim((0,7))
            return pl.hist(np.log(diff), bins=nbins)

        else:
            pl.xlabel('ITI/s')
            return pl.hist(diff, bins=nbins)


