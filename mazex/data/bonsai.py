"""Code to analyse data acquired from the setup.  """
import numpy as np
import pylab as pl

import xml.etree.ElementTree as ET
import shapely.geometry
from descartes.patch import PolygonPatch
from PIL import Image

import re
import copy
import datetime
import os
import h5py

from mazex.exceptions import InvalidMaze
from mazex.data.regionmapping import PortRegion, RegionMapping


def converttime(time):
    """Convert pointgrey timestamps to seconds.
    Code written by Gonçalo Lopes.
    https://groups.google.com/d/topic/bonsai-users/WD6mV94KAQs/discussion
    """
    # offset = time & 0xFFF
    cycle1 = (time >> 12) & 0x1FFF
    cycle2 = (time >> 25) & 0x7F
    seconds = cycle2 + cycle1 / 8000.0
    return seconds


def uncycle(time):
    """Uncycle pointgrey timestamps that have been converted to seconds using function converttime.
    Code written by Gonçalo Lopes.
    https://groups.google.com/d/topic/bonsai-users/WD6mV94KAQs/discussion
    """
    cycles = np.insert(np.diff(time) < 0, 0, False)
    cycleindex = np.cumsum(cycles)
    return time + cycleindex * 128


class BonsaiData:
    def __init__(
        self,
        centroid_file,
        metadata_file,
        metadata_delimiter=" ",
        fps=30,
        background_file=None,
        background_PNG=None,
        maze=None,
        frame_number_column=1,
        timestamps_column=0,
        sync_column=None,
        hardware_timestamps=False,
    ):
        """Container of data files created by CV in Bonsai.

        The default values were chosen such as to make this Class backwards
        compatible with the old data format used until 2019. For 2019
        dataformat use frame_number_column=0, timestamps_column=1,
        sync_column=2, hardware_timestamps=True

        Parameters
        ----------
        frame_number_column : int
            Integer value with number (zero-indexed) of column in metadat file
            where the frame numbers are to be found
        sync_column : int
            Integer value with number (zero-indexed) of column in metadat file
            where the sync signal is to be found
        hardware_timestamps : bool
            Whether to use the timestamps produced by the PointGrey camera.
            Alternatively time will be calculated assuming the given fps value
            can be trusted.
        """
        self.centroid_file = centroid_file
        self.medatata_file = metadata_file
        self.fps = fps
        self.background_file = background_file
        self.background_PNG = background_PNG
        self.maze = maze

        centroid_fn, centroid_fext = os.path.splitext(self.centroid_file)
        if centroid_fext == ".h5":
            f = h5py.File(self.centroid_file, "r")
            all_coords = f["df_with_missing/table"]["values_block_0"]
            f.close()

            # FIXME: implement filtering by confidence
            coords_certain = all_coords
            leftear = coords_certain[:, 0:2]
            rightear = coords_certain[:, 3:5]
            nose = coords_certain[:, 6:8]
            tailbase = coords_certain[:, 9:11]

            head_x = np.nanmedian((leftear[:, 0], rightear[:, 0], nose[:, 0]), axis=0)
            head_y = np.nanmedian((leftear[:, 1], rightear[:, 1], nose[:, 1]), axis=0)
            head = np.hstack((head_x[:, np.newaxis], head_y[:, np.newaxis]))

            self.raw_array = all_coords
            self.xy = head

        else:
            self.raw_array = np.loadtxt(self.centroid_file, skiprows=1)
            self.xy = self.raw_array[:, 0:2]
            self.major_axis_length = self.raw_array[:, 3]

        # it seems like we don't need this anymore
        # self.raw_array = np.where(raw_array==0, None, raw_array)
        self.raw_metadata = np.loadtxt(
            metadata_file, skiprows=1, delimiter=metadata_delimiter
        )
        # Bonsai gets the frame_numbers from the camera.
        self.frame_numbers = self.raw_metadata[:, frame_number_column]

        if hardware_timestamps:
            self.raw_timestamps = self.raw_metadata[:, timestamps_column]
            self.timestamps = uncycle(converttime(self.raw_timestamps.astype("int")))
            # convert from s to ms
            self.time_axis = self.timestamps * 1000
            self.time_axis = self.time_axis[:, np.newaxis]

        else:
            self.time_axis = (
                (self.frame_numbers - self.frame_numbers[0]) * 1000 / self.fps
            )
            # this last silly indexing expands the dimensions
            # of the vector so it can be stacked with the other matrix as follows:
            self.time_axis = self.time_axis[:, None]

        self.t_start = float(self.time_axis[0])
        self.t_stop = float(self.time_axis[-1])

        # TODO: think of a way of not duplicating the data...
        len_timed_centroid = min(len(self.time_axis), len(self.xy))
        self.timed_centroid = np.hstack(
            (self.time_axis[0:len_timed_centroid], self.xy[0:len_timed_centroid])
        )
        sample_gap = abs(max(len(self.time_axis), len(self.xy)) - len_timed_centroid)
        # FIXME: inspect what's really going on with size mismatches
        if sample_gap > 2:
            raise Exception(
                "there's a mismatch of {} time points between time_axis and xy".format(
                    sample_gap
                )
            )

        self.region_mapping = RegionMapping(self.background_file)
        self.poke_xy_dic = self.region_mapping.get_region_centroid_xy()

        # if sync_column or nan_dropped_frames:
        if sync_column:
            self.dropped_frames = self.skipped_frames()

        if sync_column:
            self.sync_signal = self.raw_metadata[:, sync_column]
            i_sync_sign_change = (
                np.where(self.sync_signal[:-1] != self.sync_signal[1:])[0] + 1
            )
            i_sync_signal_on = np.where(self.sync_signal == 1)[0]
            # now we find when there was a sign change on the sync signal and
            # the change was from 0 to 1
            i_syncon = i_sync_sign_change[np.in1d(i_sync_sign_change, i_sync_signal_on)]
            # i_syncon is the index in the original CSV data file but when
            # there are missed frames we need to correct these indices so that
            # they can match our generated self.time_axis
            if not hardware_timestamps:
                for i, ind in enumerate(i_syncon):
                    earlier_missed_frame_count = np.count_nonzero(
                        self.dropped_frames < ind
                    )
                    i_syncon[i] += earlier_missed_frame_count

            self.syncon_event_times = self.time_axis[i_syncon.astype("int")]
        # import ipdb; ipdb.set_trace()

        # if nan_dropped_frames:
        #    self.xy = self.xy.astype('float')
        #    self.xy = np.insert(
        #        self.xy, self.dropped_frames, np.array([np.nan, np.nan]), 0)

    def find_shift(
        self,
        arduino_data_object,
        tau_min=None,
        tau_max=None,
        tau_res=None,
        plot=False,
        verbose=False,
    ):
        """Calculate distance for every tau (shift)."""
        reward_events = arduino_data_object.reward_events
        reward_events_xy = np.zeros((reward_events.shape[0], 2))

        time_to_first_reward = reward_events[0, 0]

        for i, re in enumerate(reward_events):
            poke_address = re[1]
            reward_events_xy[i] = self.poke_xy_dic[poke_address]

        candidate_indices = reward_events[:, 0] / (1000 / self.fps)
        # indexing requires these to be integers
        candidate_indices = candidate_indices.round().astype("int")
        candidate_indices -= candidate_indices[0]  # so that when tau=0,
        #  candidate_indices[0] is also zero

        if not tau_res:
            tau_res = 33.3333  # ms
        if not tau_min:
            tau_min = reward_events[0, 0]
        if not tau_max:
            recording_length = self.timed_centroid[-1, 0]
            tau_max = (recording_length - reward_events[-1, 0]) - np.ceil(tau_res)
        total_steps = np.floor((tau_max - tau_min) / tau_res)  # - delta_index

        score_vs_tau = np.zeros((total_steps, 2))
        n_steps = 0
        tau = tau_min
        while n_steps < total_steps:
            score_vs_tau[n_steps] = (
                tau,
                self.score_tau(tau, candidate_indices, reward_events_xy),
            )

            if ((n_steps / total_steps) * 100) % 10 < 0.0001:
                print(int((n_steps / total_steps) * 100), "%")

            n_steps += 1
            tau += tau_res

        avg_dot_product = score_vs_tau.mean(axis=0)[1]
        abs_min_dot_product = np.min(score_vs_tau, axis=0)[1]
        mean_min_difference = abs_min_dot_product - avg_dot_product
        # we define the threshold as 90% distance to minimum from the mean
        # threshold = avg_dot_product - 0.9*(avg_dot_product - abs_min_dot_product)

        avg_major_axis_length = np.median(self.major_axis_length)

        residual_norm = np.linalg.norm(
            (avg_major_axis_length / 2) * np.ones(reward_events_xy.shape[0])
        )
        threshold = abs_min_dot_product + residual_norm

        if plot:
            pl.figure()
            pl.plot(score_vs_tau[:, 0], score_vs_tau[:, 1])
            xlim = pl.xlim()
            pl.hlines(threshold, *xlim)

        index_first_point_below_threshold = np.nonzero(score_vs_tau < threshold)[0][0]
        best_tau = score_vs_tau[:, 0][index_first_point_below_threshold]

        adjusted_tau = best_tau - time_to_first_reward

        if verbose:
            print("time of first reward: ", datetime.timedelta(seconds=best_tau / 1000))
            print(
                "data acquisition start: ",
                datetime.timedelta(seconds=adjusted_tau / 1000),
            )

            print("\nTimes of reward events\n")
            for r in reward_events[:, 0]:
                print(datetime.timedelta(seconds=(adjusted_tau + r) / 1000))

        return adjusted_tau  # this is when acquisition started

    def get_time_slice(self, t_start, t_stop):
        i_start = np.argmin(abs(self.time_axis - t_start))
        i_stop = np.argmin(abs(self.time_axis - t_stop))

        return copy.deepcopy(self.timed_centroid[i_start:i_stop])

    def get_trajectory(self, t_start=None, t_stop=None, maze=None, tolerance=30):
        if not maze:
            maze = self.maze
        if not maze:
            raise InvalidMaze
        if not t_start:
            t_start = self.t_start
        if not t_stop:
            t_stop = self.t_stop

        nearest_ports = []
        visited_ports = []
        t_axis = []
        last_visited_port = None
        time_visit = 0

        timed_centroid = self.get_time_slice(t_start, t_stop)

        for t, x, y in timed_centroid:
            # import ipdb; ipdb.set_trace()
            nearest_port = self.region_mapping.find_nearest_port((x, y))
            nearest_ports.append(nearest_port)

            # if self.region_mapping.portregions_dic[nearest_port].contains(shapely.geometry.Point(x,y)):
            if (
                self.region_mapping.portregions_dic[nearest_port].distance(
                    shapely.geometry.Point(x, y)
                )
                < tolerance
            ):
                if not last_visited_port:
                    visited_ports.append(nearest_port)
                    last_visited_port = nearest_port
                    t_axis.append(t)
                    time_visit = t

                elif nearest_port != last_visited_port:
                    if nearest_port in maze.neighbors(last_visited_port):
                        # check if it is connected to the last recognized port,i.e,
                        #  is not a tracking error
                        visited_ports.append(nearest_port)
                        last_visited_port = nearest_port
                        t_axis.append(t)
                        time_visit = t

        return t_axis, visited_ports

    def plot(self, t_start=None, t_stop=None, picture=True, color_map=None):
        img = Image.open(self.background_PNG)

        if self.background_PNG:
            pl.imshow(img, cmap="gray", vmin=0, vmax=255)

        if t_start is None and t_stop is None:
            pl.plot(self.xy[:, 0], self.xy[:, 1], linewidth=0, marker="o")

        else:
            xy_slice = self.get_time_slice(t_start, t_stop)[:, 1:3]
            if not color_map:
                pl.plot(xy_slice[:, 0], xy_slice[:, 1], linewidth=0, marker="o")
            else:
                npoints = len(xy_slice)
                # color_map='winter'

                cm = pl.get_cmap(color_map)
                fig = pl.gcf()
                ax1 = fig.add_subplot(111)
                ax1.set_color_cycle(
                    [cm(1.0 * i / (npoints - 1)) for i in range(npoints - 1)]
                )

                for i in range(npoints - 1):
                    ax1.plot(xy_slice[i : i + 2, 0], xy_slice[i : i + 2, 1])

                pl.title(
                    "black=%i s, white=%i s"
                    % (np.round(t_start / 1000), np.round(t_stop / 1000))
                )

        pl.ylim(2048, 0)
        pl.xlim(0, 2048)

    def score_tau(self, tau, candidate_indices, reward_events_xy):
        """Gives a score to the tested tau value.

        Parameters:
        -----------
        tau : int
            Value of the shift in ms

        reward_events : numpy.array
            Numpy array containing time of poke (ms) and poke port address

        mapping : RegionMapping object
            Contains the mapping of poke port addresses to their X,Y coordinates
            on the image/video.
        """
        # we find the indices on the timed_centroid array that correspond to the event
        #  times in the reward_events array

        tau_index_shift = round((tau / 1000) * self.fps)

        re_indexed = copy.deepcopy(reward_events_xy)
        re_indexed = re_indexed.astype("int")
        re_indexed[:, 0] = candidate_indices + tau_index_shift

        centroid_xy_coord = self.timed_centroid[re_indexed[:, 0], 1:3]

        return np.linalg.norm(centroid_xy_coord - reward_events_xy)

    def skipped_frames(self):
        """Show where and how many frames were skipped.
        #TODO
        """
        index_last_contiguous_frame = np.where(
            self.frame_numbers[:-1] != self.frame_numbers[1:] + 1
        )[0]

        index_missing_frames = []
        for i in index_last_contiguous_frame:
            frame_number_i = self.frame_numbers[i]
            frame_number_i_prime = self.frame_numbers[i + 1]
            n_frames_missing = frame_number_i_prime - frame_number_i - 1
            for x in range(int(n_frames_missing)):
                index_missing_frames.append(index_last_contiguous_frame + x + 1)
        return index_missing_frames
