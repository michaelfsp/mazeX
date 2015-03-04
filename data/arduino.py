import numpy as np
import pylab as pl

class ArduinoDataFile:
    def __init__(self, arduino_file):
        self.arduino_file = arduino_file

        self.raw = np.loadtxt(self.arduino_file)
        self.reward_events = self.raw[(self.raw[:,1]<100)&(
            self.raw[:,2]==0)][:-1][:,:2]
        self.reward_event_times = self.reward_events[:,0]

    @property
    def ITIs(self):
        return np.diff(self.reward_event_times)

