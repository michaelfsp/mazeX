import numpy as np
import pylab as pl
import prettyplotlib as ppl

class ArduinoData:
    def __init__(self, arduino_file, mouse_id, date):
        self.arduino_file = arduino_file
        self.mouse_id = mouse_id
        self.date = date

        self.raw = np.loadtxt(self.arduino_file)
        self.reward_events = self.raw[(self.raw[:,1]<100)&(
            self.raw[:,2]==0)][:-1][:,:2]
        self.reward_event_times = reward_events[m][d][:,0]

    def ITIs(self):
        return np.diff(self.reward_event_times)




#-------------------------------Parsing Arduino data--------------------------

