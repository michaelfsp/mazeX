import numpy as np
import pandas as pd
import h5py
from . import rsync

class InscopixCNMFEData:
    def __init__(self, gpio_file_paths, cnmfe_file_path, fr=30.0, threshold=62000):
        self.gpio_file_paths = gpio_file_paths
        self.cnmfe_file_path = cnmfe_file_path
        self.timestamps = []
        self.on_timestamps = []
        for f in gpio_file_paths:
            inscopix_pulses = h5py.File(f)
            #frames.extend(list(inscopix_pulses['frame']))
            sync_channel = np.max(inscopix_pulses['anc_data'][:,0:8],axis=1)
            timestamps = np.array(inscopix_pulses['timestamp'])
            self.timestamps.append(timestamps)

            sync_channel_digital = np.where(sync_channel>threshold, 1, 0)
            change_time = sync_channel_digital[1:] - sync_channel_digital[:-1]
            on_events = np.where(change_time == 1, True, False) # it has len(sync_channel) - 1

            self.on_timestamps.append(timestamps[on_events])

        # Load CNMFE results
        cnmfe = h5py.File(cnmfe_file_path)
        self.S_concatenated = np.array(cnmfe['estimates']['S']).T
        self.S_chunks = []

        start_idx = 0
        for ts in self.timestamps:
            chunk_len = len(ts)
            self.S_chunks.append(self.S_concatenated[start_idx:chunk_len, :])
            start_idx += chunk_len

#class CombinedData:
#    def __init__(self, inscopix_cnmfe_data_object, bonsai_data_objects)
#        
