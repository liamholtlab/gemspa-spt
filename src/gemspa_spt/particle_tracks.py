"""
This file contains functionality for analysis of single particle tracking data  in an implementation of the
ParticleTracks class.

Internally, track data is stored as a numpy array in the napari format for the 'tracks' layer.
See: https://napari.org/stable/howtos/layers/tracks.html#tracks-data

The class can be initialized with a numpy array of track data.  It also has a function to read in tracking data from a
file.  Mosaic, Trackmate and trackpy file formats are supported.

"""

import pandas as pd
import numpy as np
from skimage import draw


class ParticleTracks:

    file_formats = ['mosaic', 'trackmate', 'trackpy', 'napari']
    file_columns = {'mosaic':    ['trajectory', 'frame', 'z', 'y', 'x'],
                    'trackmate': ['track_id', 'frame', 'position_z', 'position_y', 'position_x'],
                    'trackpy':   ['particle', 'frame', 'z', 'y', 'x'],
                    'napari':    ['track_id', 't', 'z', 'y', 'x']}
    skip_rows = {'mosaic': None, 'trackmate': [1, 2, 3], 'trackpy': None, 'napari': None}

    def __init__(self, tracks=None):
        """
            Initialize class instance.

            Parameters
            ----------
            tracks : numpy 2d array
            the track data with columns in this order: track_id, t, z, y, x

            Returns
            -------
            None
        """

        if tracks is not None:
            if isinstance(tracks, np.ndarray) and (len(tracks.shape) == 2) and (tracks.shape[1] >= 5):
                # format looks correct
                self.tracks = tracks

                # check if it is 2d or 3d data, and set self.dim
                self._set_dim()

                # self.track_df is the original input track data, as a pandas data frame
                # (as input by read_track_df or read_track_file)
                # here, it is the same as self.tracks but with the column names added
                self.track_df = pd.DataFrame(self.tracks, columns=self.file_columns['napari'])
            else:
                raise Exception(f"Error initializing track data: invalid format.")
        else:
            self.tracks = None
            self.track_df = None
            self.dim = None

        self.track_lengths = None
        self.mean_track_intensities = None
        self.track_intensities = None
        self.msds = None
        self.fit_results = {'linear': None, 'log': None}

        self.microns_per_pixel = 0.11
        self.time_lag_sec = 0.010
        self.max_lagtime_fit = 11
        self.error_term_fit = True

    def _set_dim(self):
        if self.tracks is not None:
            if np.unique(self.tracks[:, 2]).shape[0] == 1:
                self.dim = 2
            else:
                self.dim = 3

    def read_track_df(self, df, data_format='mosaic'):
        """
            Read track data from a pandas data frame.

            Reads track data formatted in mosaic, trackmate or trackpy, format from a pandas data frame.  Track data
            will be extracted and stored as a numpy array in the class variable, 'tracks'.  Original data frame will be
            stored in self.track_df

            Parameters
            ----------
            df : pandas.DataFrame
                the track data as a data frame
            data_format : str
                String indicating type of formatting: 'mosaic', 'trackpy' or 'trackmate'

            An Exception is raised if the formatting is invalid or file_format is not one of:
            'mosiac', 'trackmate', 'trackpy'

            Returns
            -------
            (n, 5) ndarray: (n=number of tracks), column order is [track_id, t, z, y, x]
        """

        data_format = data_format.lower()
        if data_format not in ParticleTracks.file_formats:
            raise Exception(f"Error in reading tracks data frame: file format '{data_format}' is not recognized.")

        if not isinstance(df, pd.DataFrame):
            raise Exception(f"Error in reading tracks data frame: input parameter df is not a pandas data frame")

        for col in ParticleTracks.file_columns[data_format]:
            if col != ParticleTracks.file_columns[data_format][2] and col not in df.columns:
                raise Exception(
                    f"Error in reading tracks data frame: required column {col} is missing for format {data_format}")

        self.track_df = df.copy()
        if ParticleTracks.file_columns[data_format][2] not in df.columns:
            df[ParticleTracks.file_columns[data_format][2]] = 0

        self.tracks = df[ParticleTracks.file_columns[data_format]]
        self._set_dim()
        return self.tracks

    def read_track_file(self, path, data_format='mosaic', sep=','):
        """
            Read track data from a file.

            Reads the track data from Mosaic, Trackmate or trackpy format files.  Track data will be extracted and
            stored as a numpy array in the class variable, 'tracks'.

            Parameters
            ----------
            path : str
                Full path to the track file.
            data_format : str
                String indicating type of file: 'mosaic', 'trackpy' or 'trackmate'
            sep : str
                String indicating field separator for file, e.g. ',' or '\t'

            An Exception is raised if the file format is invalid or file_format is not one of:
            'mosiac', 'trackmate', 'trackpy'

            Returns
            -------
            (n, 5) ndarray: (n=number of tracks), column order is [track_id, t, z, y, x]
        """

        if data_format not in ParticleTracks.file_formats:
            raise Exception(f"Error in reading {path}: file format '{data_format}' is not recognized.")

        df = pd.read_csv(path, sep=sep, header=0, skiprows=ParticleTracks.skip_rows[data_format])
        return self.read_track_df(df, data_format=data_format)

    def write_track_file(self, path):
        """
            Write track data to a file.

            Writes track data as a tab-delimited text file with columns: 'track_id', 't', 'z', 'y', 'x'

            Parameters
            ----------
            path : str
                Full path to the track file.

            Returns
            -------
            None
        """

        df = pd.DataFrame(self.tracks[:, 5], columns=['track_id', 't', 'z', 'y', 'x'])
        df.to_csv(path, sep='\t')

    def find_track_lengths(self):
        """
            Compute track lengths and fills the class variable, 'track_lengths'

            Parameters
            ----------

            Returns
            -------
            (n, 2) ndarray: (n=number of tracks), column order is [track_id, track-length]
        """
        if self.tracks is not None:
            self.track_lengths = np.stack(np.unique(self.tracks[:, 0], return_counts=True), axis=1)
        else:
            raise Exception(f"Error find_track_lengths: track data is empty.")

        return self.track_lengths

    def find_track_intensities(self, movie, radius=3):
        """
            Compute mean and sum of track intensities.  (1) Fills the class variable, 'track_intensities' with mean and
            sum intensity of a disk with given radius at each track position; it is a Nx3 numpy array, column positions
            are: [track_id, mean_intensity, sum_intensity].  (2) Fills the class variable, 'mean_track_intensities' with
            the mean track intensity for each track; it is a Nx3 numpy array, column positions are:
            [track_id, mean_intensity, stdev_intensity]

            Parameters
            ----------
            movie : (t, N, M) ndarray
                Movie to extract intensity information.  first dimension corresponds to time (frames) of movie

            radius : int
                Disk radius around each point for intensity calculation

            Returns
            -------
            (n, 2) ndarray: (n=number of tracks), columns are [track_id, mean_intensity, stdev_intensity]
        """

        if self.tracks is None:
            raise Exception(f"Error find_track_intensities: track data is empty.")

        if (type(np.array([])) != type(movie)) or (len(movie.shape) != 3):
            raise Exception(f"Error find_track_intensities: movie is not of valid format.")

        if self.tracks[:, 1].max() >= movie.shape[0]:
            raise Exception("Error find_track_intensities: track frames exceed size of image first dimension.")

        track_ids = np.unique(self.tracks[:, 0])
        self.mean_track_intensities = np.zeros(shape=(track_ids.size, 3), )
        self.track_intensities = np.zeros(shape=(self.tracks.shape[0], 3), )

        index = 0
        for i, track_id in enumerate(track_ids):
            cur_track = self.tracks[self.tracks[:, 0] == track_id]
            for j in range(cur_track.shape[0]):
                rr, cc = draw.disk((int(cur_track[j][3]),
                                    int(cur_track[j][4])),
                                   radius,
                                   shape=movie.shape[1:])

                self.track_intensities[index][0] = track_id
                self.track_intensities[index][1] = np.mean(movie[int(cur_track[j][1])][rr, cc])
                self.track_intensities[index][2] = np.sum(movie[int(cur_track[j][1])][rr, cc])
                index += 1

            self.mean_track_intensities[i][0] = track_id
            self.mean_track_intensities[i][1] = np.mean(self.track_intensities[index-cur_track.shape[0]:index, 1])
            self.mean_track_intensities[i][2] = np.std(self.track_intensities[index-cur_track.shape[0]:index, 1])

        return self.mean_track_intensities

    def msd_and_fitting(self, track_id, fft=True):
        pass

    def find_msds(self, track_id=None, fft=True):

        if self.tracks is None:
            raise Exception(f"Error in find_msds: track data is empty.")

        # init msd array if not already done so
        if self.msds is None:
            self.msds = np.zeros(shape=(self.tracks.shape[0], 5))
            self.msds[:, 0] = self.tracks[:, 0]

        # 2d or 3d data?
        if self.dim == 3:
            pos_cols = [2, 3, 4]
        else:
            pos_cols = [3, 4]

        if track_id is None:
            pass
            # BATCH MSD ALL TRACKS
        else:
            if track_id not in np.unique(self.tracks[:, 0]):
                raise Exception(f"Error in find_msds: track_id not found.")

            traj = self.tracks[self.tracks[:, 0] == track_id]
            r = traj[:, pos_cols] * self.microns_per_pixel
            t = (traj[:, 1] - traj[0, 1]) * self.time_lag_sec

            if fft:
                #     MSD calculation using Fourier transform (faster)
                #
                #     The original Python implementation comes from a SO answer :
                #     http://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft#34222273.
                #     The algorithm is described in this paper : http://dx.doi.org/10.1051/sfn/201112010.
                #
                #     Vectorized implementation comes from "trackpy" package code :
                #     https://github.com/soft-matter/trackpy/blob/master/trackpy/motion.py
                #

                N = len(r)

                # S1:
                D = r**2
                D_sum = D[:N-1] + D[:-N:-1]
                S1 = 2*D.sum(axis=0)-np.cumsum(D_sum, axis=0)

                # S2 (auto correlation): find PSD, which is the fourier transform of the auto correlation
                F = np.fft.fft(r, n=2*N, axis=0)  # 2*N because of zero-padding
                PSD = F*F.conjugate()
                S2 = np.fft.ifft(PSD, axis=0)[1:N].real

                # MSD = S1 - 2*S2
                squared_disp = S1 - 2 * S2
                squared_disp /= (N - np.arange(1, N)[:, np.newaxis])  # divide by (N-m)

                # Insert into self.msds
                self.msds[self.msds[:, 0] == track_id, 1] = t
                self.msds[(self.msds[:, 0] == track_id) & (self.msds[:, 1] > 0), 2:2+self.dim] = squared_disp[:, 0:self.dim]
                self.msds[(self.msds[:, 0] == track_id) & (self.msds[:, 1] > 0), 2+self.dim] = squared_disp.sum(axis=1)

            else:
                # straight-forward MSD calculation
                shifts = np.arange(len(r))
                msds = np.zeros(shape=(shifts.size, 4))

                for i, shift in enumerate(shifts):
                    diffs = r[:-shift if shift else None] - r[shift:]
                    sqdist = np.square(diffs)

                    msds[i, 0] = t[i]
                    msds[i, 1] = sqdist[:, 0].mean()
                    msds[i, 2] = sqdist[:, 1].mean()
                    msds[i, 3] = sqdist.sum(axis=1).mean()

                # Insert into self.msds
                self.msds[self.msds[:, 0] == track_id, 1:] = msds

        return self.msds[self.msds[:, 0] == track_id]

    def fit_msd(self, track_id=None, msds=None, scale='linear'):

        scale = scale.lower()
        if scale not in ['linear', 'log']:
            raise Exception(f"Error in fit_msd: scale must be one of 'linear' or 'log'")

        if self.msds is None:
            raise Exception(f"Error in fit_msd: msd data is empty.")

        # init fit array if not already done so
        if self.fit_results[scale] is None:
            track_ids = np.unique(self.msds[:,0])
            self.fit_results[scale] = np.zeros(shape=(track_ids.size, 5))
            self.fit_results[scale][:, 0] = track_ids

        if track_id is None:
            pass
            # BATCH MSD ALL TRACKS
        else:
            if track_id not in np.unique(self.msds[:, 0]):
                raise Exception(f"Error in fit_msd: track_id not found.")

            # Check find_msds has been run for this track (or all tracks)
            traj = self.msds[self.msds[:, 0] == track_id]
            if traj[:, 1:].sum() == 0:
                raise Exception(f"Error in fit_msd: no msd data (find_msds must be run first for this track).")

            # Fit linear or loglog scale
            if scale == 'linear':  # D, Err, r_sq
                pass
            elif scale == 'log':  # K, alpha, r_sq
                pass

    # TODO: make one function to find msd and do fitting for linear and log scale if given a track ID
    #  (data is not saved and this is a quick calculation)
    #
    # TODO: make 2 other functions, one to find msd for all tracks and the 2nd to do fitting for linear or log scale
    #  (as chosen by user) run these to batch over all tracks





