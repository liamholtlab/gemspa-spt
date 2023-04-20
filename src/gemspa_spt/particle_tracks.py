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
from scipy.optimize import curve_fit


class ParticleTracks:

    file_formats = ['mosaic', 'trackmate', 'trackpy', 'napari']
    file_columns = {'mosaic':    ['trajectory', 'frame', 'z', 'y', 'x'],
                    'trackmate': ['track_id', 'frame', 'position_z', 'position_y', 'position_x'],
                    'trackpy':   ['particle', 'frame', 'z', 'y', 'x'],
                    'napari':    ['track_id', 't', 'z', 'y', 'x']}
    skip_rows = {'mosaic': None, 'trackmate': [1, 2, 3], 'trackpy': None, 'napari': None}

    def __init__(self, tracks=None, tracks_df=None, path=None, data_format='mosaic', sep=','):
        """
            Initialize class instance.
            The track data will be initialized from one of: tracks, tracks_df or path

            Parameters
            ----------
            tracks : numpy 2d array
                the track data with columns in this order: track_id, t, z, y, x

            tracks_df : pandas.DataFrame
                the track data as a data frame; data_format indicates the formatting

            path : str
                full path to the track file; data_format indicates the formatting, sep indicates file separator

            data_format : str
                String indicating type of formatting: 'mosaic', 'trackpy' or 'trackmate',  An Exception is raised if
                the formatting is invalid or file_format is not one of: 'mosiac', 'trackmate', 'trackpy'
                Used only if reading from a pandas data frame or from path

            sep : str
                String indicating field separator for file, e.g. ',' or '\t'
                Used only if reading from path

            Returns
            -------
            None
        """

        if tracks is not None:
            if isinstance(tracks, np.ndarray) and (len(tracks.shape) == 2) and (tracks.shape[1] >= 5):
                self.tracks = tracks

                # self.track_df is the original input track data, as a pandas data frame
                # here, it is the same as self.tracks but with the column names added
                self.tracks_df = pd.DataFrame(self.tracks, columns=self.file_columns['napari'])
            else:
                raise Exception(f"Error initializing track data: invalid format.")
        else:
            data_format = data_format.lower()
            if data_format not in ParticleTracks.file_formats:
                raise Exception(f"Error in initializing track data: data format '{data_format}' is not recognized.")

            if tracks_df is None and path is not None:
                tracks_df = pd.read_csv(path, sep=sep, header=0, skiprows=ParticleTracks.skip_rows[data_format])

            if tracks_df is not None:
                self._read_track_df(tracks_df, data_format=data_format)
            else:
                raise Exception(f"Error: no track data.")

        self._init_track_data()
        self.mean_track_intensities = None
        self.track_intensities = None
        self.msds = None
        self.ensemble_avg = None
        self.fit_results = {'linear': None, 'log': None}

        self.microns_per_pixel = 0.11
        self.time_lag_sec = 0.010

    def _init_track_data(self):
        self._set_dim()
        self.track_ids = np.unique(self.tracks[:, 0])
        self._set_track_lengths()

    def _set_dim(self):
        if self.tracks is not None:
            if np.unique(self.tracks[:, 2]).shape[0] == 1:
                self.dim = 2
            else:
                self.dim = 3

    def _read_track_df(self, df, data_format='mosaic'):
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
        """

        if not isinstance(df, pd.DataFrame):
            raise Exception(f"Error in reading tracks data frame: input parameter df is not a pandas data frame")

        for col in ParticleTracks.file_columns[data_format]:
            if col != ParticleTracks.file_columns[data_format][2] and col not in df.columns:
                raise Exception(
                    f"Error in reading tracks data frame: required column {col} is missing for format {data_format}")

        self.tracks_df = df.copy()
        if ParticleTracks.file_columns[data_format][2] not in df.columns:
            df[ParticleTracks.file_columns[data_format][2]] = 0

        self.tracks = df[ParticleTracks.file_columns[data_format]]

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

    def _set_track_lengths(self):
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

        self.mean_track_intensities = np.zeros(shape=(len(self.track_ids), 3), )
        self.track_intensities = np.zeros(shape=(self.tracks.shape[0], 3), )

        index = 0
        for i, track_id in enumerate(self.track_ids):
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

    def msd(self, track_id, fft=True):
        if self.tracks is None:
            raise Exception(f"Error in msd_and_fitting: track data is empty.")

        if track_id not in self.track_ids:
            raise Exception(f"Error in msd_and_fitting: track_id not found.")

        traj = self.tracks[self.tracks[:, 0] == track_id]
        r = traj[:, [2, 3, 4]] * self.microns_per_pixel
        t = (traj[:, 1] - traj[0, 1]) * self.time_lag_sec

        msds = np.zeros(shape=(r.shape[0], 5))
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
            D = r ** 2
            D_sum = D[:N - 1] + D[:-N:-1]
            S1 = 2 * D.sum(axis=0) - np.cumsum(D_sum, axis=0)

            # S2 (auto correlation): find PSD, which is the fourier transform of the auto correlation
            F = np.fft.fft(r, n=2 * N, axis=0)  # 2*N because of zero-padding
            PSD = F * F.conjugate()
            S2 = np.fft.ifft(PSD, axis=0)[1:N].real

            # MSD = S1 - 2*S2
            squared_disp = S1 - 2 * S2
            squared_disp /= (N - np.arange(1, N)[:, np.newaxis])  # divide by (N-m)

            # msds array to return from function
            msds[1:, 0] = t[1:]
            msds[1:, 1:4] = squared_disp[:, 0:3]
            msds[1:, 4] = squared_disp.sum(axis=1)
        else:
            # straight-forward MSD calculation
            shifts = np.arange(1, r.shape[0])
            for i, shift in enumerate(shifts):
                diffs = r[:-shift if shift else None] - r[shift:]
                sqdist = np.square(diffs)

                msds[i+1, 0] = t[i+1]
                msds[i+1, 1:4] = sqdist[:, 0:3].mean(axis=0)
                msds[i+1, 4] = sqdist.sum(axis=1).mean(axis=0)
        return msds

    def fit_msd_linear(self, t, msd, dim, max_lagtime=10, err=True):
        t = t[:max_lagtime]
        msd = msd[:max_lagtime]
        if err:
            def linear_fn(x, a, c):
                return a * x + c
            linear_fn_v = np.vectorize(linear_fn)
            popt, pcov = curve_fit(linear_fn, t, msd, p0=[2*dim*0.2, 0])
            residuals = msd - linear_fn_v(t, popt[0], popt[1])
        else:
            def linear_fn(x, a):
                return a * x
            linear_fn_v = np.vectorize(linear_fn)
            popt, pcov = curve_fit(linear_fn, t, msd, p0=[2*dim*0.2])
            residuals = msd - linear_fn_v(t, popt[0])

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((msd - np.mean(msd)) ** 2)
        r_squared = max(0, 1-(ss_res/ss_tot))

        D = popt[0]/(2*dim)
        if err:
            E = popt[1]
        else:
            E = 0

        return D, E, r_squared

    def fit_msd_loglog(self, t, msd, dim, max_lagtime=10):
        t = t[:max_lagtime]
        msd = msd[:max_lagtime]
        def linear_fn(x, m, b):
            return m * x + b
        linear_fn_v = np.vectorize(linear_fn)

        popt, pcov = curve_fit(linear_fn, np.log(t), np.nan_to_num(np.log(msd)), p0=[1, np.log(2*dim*0.2)])
        residuals = np.nan_to_num(np.log(msd)) - linear_fn_v(np.log(t), popt[0], popt[1])
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((np.nan_to_num(np.log(msd))-np.mean(np.nan_to_num(np.log(msd))))**2)
        r_squared = max(0, 1-(ss_res/ss_tot))

        alpha = popt[0]
        K = np.exp(popt[1])/(2*dim)

        return K, alpha, r_squared

    def msd_all_tracks(self, fft=True):

        if self.tracks is None:
            raise Exception(f"Error in find_msds: track data is empty.")

        # init msd array
        self.msds = np.zeros(shape=(self.tracks.shape[0], 6))
        self.msds[:, 0] = self.tracks[:, 0]

        # MSD ALL TRACKS
        for track_id in self.track_ids:
            self.msds[self.msds[:, 0] == track_id, 1:] = self.msd(track_id, fft)

        return self.msds

    def ensemble_avg_msd(self, min_len=11):
        if self.tracks is None:
            raise Exception(f"Error in ensemble_avg_msd: track data is empty.")

        if self.msds is None:
            raise Exception(f"Error in ensemble_avg_msd: msd data is empty.")

        if min_len > 1:
            # filter on track length
            track_ids = self.track_lengths[self.track_lengths[:, 1] >= min_len, 0]
            msds = self.msds[np.isin(self.msds[:, 0], track_ids)]
        else:
            track_ids = self.track_ids
            msds = self.msds

        if len(track_ids) > 0:
            all_tlags = np.unique(msds[:, 1])
            self.ensemble_avg_n = np.zeros(shape=len(all_tlags), dtype=int)
            self.ensemble_avg = np.zeros(shape=(len(all_tlags), 5))
            for i, tlag in enumerate(all_tlags):
                tlag_msds = msds[msds[:, 1] == tlag, 1:]
                self.ensemble_avg_n[i] = len(tlag_msds)
                self.ensemble_avg[i] = np.mean(tlag_msds, axis=0)

        return self.ensemble_avg, self.ensemble_avg_n


    #def fit_msd_linear_all_tracks(self, max_lagtime=10, err=True):
    #
    #     scale = scale.lower()
    #     if scale not in ['linear', 'log']:
    #         raise Exception(f"Error in fit_msd: scale must be one of 'linear' or 'log'")
    #
    #     if self.msds is None:
    #         raise Exception(f"Error in fit_msd: msd data is empty.")
    #
    #     # init fit array if not already done so
    #     if self.fit_results[scale] is None:
    #         self.fit_results[scale] = np.zeros(shape=(len(self.track_ids), 5))
    #         self.fit_results[scale][:, 0] = self.track_ids
    #
    #     if track_id is None:
    #         pass
    #         # BATCH MSD ALL TRACKS
    #     else:
    #         if track_id not in self.track_ids:
    #             raise Exception(f"Error in fit_msd: track_id not found.")
    #
    #         # Check find_msds has been run for this track (or all tracks)
    #         traj = self.msds[self.msds[:, 0] == track_id]
    #         if traj[:, 1:].sum() == 0:
    #             raise Exception(f"Error in fit_msd: no msd data (find_msds must be run first for this track).")
    #
    #         # Fit linear or loglog scale
    #         if scale == 'linear':  # D, Err, r_sq
    #             pass
    #         elif scale == 'log':  # K, alpha, r_sq
    #             pass






