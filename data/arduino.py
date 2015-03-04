import numpy as np
import pylab as pl

class ArduinoDataFile:
    def __init__(self, arduino_file):
        self.arduino_file = arduino_file

        self.raw = np.loadtxt(self.arduino_file)
        self.reward_events = self.raw[(self.raw[:,1]<100)&(
            self.raw[:,2]==0)][:,:2]
        self.reward_event_times = self.reward_events[:,0]

        self.lightson_events = self.raw[(self.raw[:,1]<100)&(
            self.raw[:,2]==1)][:,:2]
        self.lightson_event_times = self.lightson_events[:,0]


    @property
    def ITIs(self):
        return np.diff(self.reward_event_times)
    
    @property
    def delay_period(self):
        # we cannot use all events because sometimes the task finishes within
        #   the delay period
        if len(self.lightson_event_times[1:]) < len(self.reward_event_times):
            delays = self.lightson_event_times[1:] - self.reward_event_times[:-1]
        elif len(self.lightson_event_times[1:]) > len(self.reward_event_times):
            delays = self.lightson_event_times[1:-1] - self.reward_event_times
        else:
            delays = self.lightson_event_times[1:] - self.reward_event_times
        return delays.mean()
