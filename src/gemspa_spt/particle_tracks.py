"""
This file contains functionality for analysis of single particle tracking data  in an implementation of the
ParticleTracks class.

Internally, track data is stored as a numpy array in the napari format for the 'tracks' layer.
See: https://napari.org/stable/howtos/layers/tracks.html#tracks-data

The class can be initialized with a numpy array of track data.  It also has a function to read in tracking data from a
file.  Mosaic, Trackmate, gemspa and trackpy file formats are supported.

"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool
from numpy import linalg as LA


class ParticleTracks:

    file_columns = {'mosaic':    ['trajectory', 'frame', 'z', 'y', 'x'],
                    'trackmate': ['track_id', 'frame', 'position_z', 'position_y', 'position_x'],
                    'trackpy':   ['particle', 'frame', 'z', 'y', 'x'],
                    'gemspa':    ['track_id', 'frame', 'z', 'y', 'x']}
    skip_rows = {'mosaic': None, 'trackmate': [1, 2, 3], 'trackpy': None, 'gemspa': None}
    dim_columns = {'z': 2, 'y': 3, 'x': 4, 'sum': 5}
    column_dims = {2: 'z', 3: 'y', 4: 'x', 5: 'sum'}

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
                String indicating type of formatting.  An Exception is raised if the formatting is invalid or
                file_format is not one of: 'mosiac', 'trackmate', 'gemspa', 'trackpy'
                Used only if reading from a pandas data frame or from path

            sep : str
                String indicating field separator for file, e.g. ',' or '\t'
                Used only if reading from path

            Returns
            -------
            None
        """
        self.mean_track_intensities = None
        self.track_intensities = None
        self.msds = None
        self.ensemble_avg = None
        self.linear_fit_results = None
        self.loglog_fit_results = None
        self.radius_of_gyration = None
        self.step_sizes = None

        self.track_ids = None
        self.dimension = 0
        self.track_lengths = None

        self.microns_per_pixel = 0.11
        self.time_lag_sec = 0.010

        if tracks is not None:
            if isinstance(tracks, np.ndarray) and (len(tracks.shape) == 2) and (tracks.shape[1] >= 5):
                self.tracks = tracks

                # self.track_df is the original input track data, as a pandas data frame
                # here, it is the same as self.tracks but with the column names added
                self.tracks_df = pd.DataFrame(self.tracks, columns=self.file_columns['gemspa'])
            else:
                raise Exception(f"Error initializing track data: invalid format.")
        else:
            data_format = data_format.lower()
            if data_format not in ParticleTracks.file_columns.keys():
                raise Exception(f"Error in initializing track data: data format '{data_format}' is not recognized.")

            if tracks_df is None and path is not None:
                tracks_df = pd.read_csv(path, sep=sep, header=0, skiprows=ParticleTracks.skip_rows[data_format])

            if tracks_df is not None:
                self._read_track_df(tracks_df, data_format=data_format)
                self.tracks_df = tracks_df
            else:
                raise Exception(f"Error: no track data.")

        self._init_track_data()

    def _init_track_data(self):
        self._set_dim()
        self.track_ids = np.unique(self.tracks[:, 0])
        self._set_track_lengths()

    def _set_dim(self):
        self.dimension = 0
        if self.tracks is not None:
            for dim in ParticleTracks.dim_columns.keys():
                if dim != 'sum' and np.unique(self.tracks[:, ParticleTracks.dim_columns[dim]]).shape[0] > 1:
                    self.dimension += 1

    def _read_track_df(self, df, data_format='mosaic'):
        """
            Read track data from a pandas data frame.

            Reads track data formatted in mosaic, trackmate, gemspa or trackpy format from a pandas data frame.  Track
            data will be extracted and stored as a numpy array in the class variable, 'tracks'

            Parameters
            ----------
            df : pandas.DataFrame
                the track data as a data frame
            data_format : str
                String indicating type of formatting: 'mosaic', 'gemspa', 'trackpy' or 'trackmate'
        """

        if not isinstance(df, pd.DataFrame):
            raise Exception(f"Error in reading tracks data frame: input parameter df is not a pandas data frame")

        df.columns = df.columns.str.lower()

        for col in ParticleTracks.file_columns[data_format]:
            if col != ParticleTracks.file_columns[data_format][2] and col not in df.columns:
                raise Exception(
                    f"Error in reading tracks data frame: required column {col} is missing for format {data_format}")

        if ParticleTracks.file_columns[data_format][2] not in df.columns:
            df[ParticleTracks.file_columns[data_format][2]] = 0

        self.tracks = df[ParticleTracks.file_columns[data_format]].to_numpy()

    def write_track_file(self, path, sep=','):
        """
            Write track data to a file.

            Writes track data as a delimited text file with columns: 'track_id', 'frame', 'z', 'y', 'x'
            This is a 'gemspa' formatted tracks file

            Parameters
            ----------
            path : str
                Full path to the track file.

            sep : file separator for delimited table data

            Returns
            -------
            None
        """

        df = pd.DataFrame(self.tracks[:, :5], columns=self.file_columns['gemspa'])
        df.to_csv(path, sep=sep)

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

    def step_size(self, track_id):
        """
                Calculates the step sizes in a track given its ID.

                The method retrieves the track based on the provided ID, scales the
                coordinates according to microns per pixel, and calculates the time lag.
                It then computes the step sizes, with the first column being time,
                next three being differences in x, y, z coordinates, and the last column
                being the Euclidean distance.

                Parameters:
                track_id : int
                    Unique identifier of a track in the track_ids list.

                Raises:
                Exception:
                    If track data is empty or if the provided track_id is not found.

                Returns:
                step_sizes : numpy.ndarray
                    A 2D array with columns representing time, differences in x, y, z
                    coordinates and the Euclidean distance respectively.
                """
        if self.tracks is None:
            raise Exception(f"Error in step_size: track data is empty.")

        if track_id not in self.track_ids:
            raise Exception(f"Error in step_size: track_id not found.")

        traj = self.tracks[self.tracks[:, 0] == track_id]
        r = traj[:, [2, 3, 4]] * self.microns_per_pixel
        t = (traj[:, 1] - traj[0, 1]) * self.time_lag_sec

        step_sizes = np.zeros(shape=(r.shape[0], 5))

        diffs_1d = np.abs(r[:-1] - r[1:])

        step_sizes[:, 0] = t
        step_sizes[1:, 1:4] = diffs_1d
        step_sizes[1:, 4] = np.sqrt(np.square(diffs_1d).sum(axis=1))

        return step_sizes

    def msd(self, track_id, fft=True):
        """
         Calculates the Mean Square Displacement (MSD) for a given track ID.

         This method first retrieves the track based on the provided ID and scales
         the coordinates and time appropriately. It then calculates the MSD either
         using a Fast Fourier Transform (FFT) algorithm, or a straight-forward
         method depending on the fft parameter.

         Parameters:
         track_id : int
             Unique identifier of a track in the track_ids list.
         fft : bool, optional
             If True, MSD calculation uses FFT for speed (default is True).

         Raises:
         Exception:
             If track data is empty or if the provided track_id is not found.

         Returns:
         msds : numpy.ndarray
             A 2D array with columns representing time, mean square displacements
             in x, y, z coordinates, and the overall mean square displacement respectively.
         """

        if self.tracks is None:
            raise Exception(f"Error in msd: track data is empty.")

        if track_id not in self.track_ids:
            raise Exception(f"Error in msd: track_id not found.")

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
        """
        Fits the Mean Square Displacement (MSD) to a linear function.

        This method takes time and MSD data and fits them to a linear function
        to find the diffusion coefficient and optionally, the error. It uses
        least-squares optimization for fitting the data. The goodness of fit
        is represented using the R-squared statistic.

        Parameters:
        t : numpy.ndarray
            1D array containing time data.
        msd : numpy.ndarray
            1D array containing Mean Square Displacement data.
        dim : int
            The dimension of the displacement (2 for 2D, 3 for 3D)
        max_lagtime : int, optional
            Maximum lag time to consider for the fit (default is 10).
        err : bool, optional
            If True, includes error in the linear model (default is True).

        Returns:
        D : float
            Estimated diffusion coefficient.
        E : float
            Estimated error (or zero if err is False).
        r_squared : float
            R-squared statistic representing the goodness of fit.

        Note:
        Uses scipy's curve_fit for the least-squares minimization.
        """
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
        """
        Fits the logarithm of the Mean Square Displacement (MSD) to a linear function.

        This method takes logarithm of time and MSD data and fits them to a linear function
        to find the anomalous exponent and the generalized diffusion coefficient. It uses
        least-squares optimization for fitting the data. The goodness of fit
        is represented using the R-squared statistic.

        Parameters:
        t : numpy.ndarray
            1D array containing time data.
        msd : numpy.ndarray
            1D array containing Mean Square Displacement data.
        dim : int
            The dimension of the displacement (2 for 2D, 3 for 3D)
        max_lagtime : int, optional
            Maximum lag time to consider for the fit (default is 10).

        Returns:
        K : float
            Estimated generalized diffusion coefficient.
        alpha : float
            Estimated anomalous exponent.
        r_squared : float
            R-squared statistic representing the goodness of fit.

        Note:
        Uses scipy's curve_fit for the least-squares minimization.
        """
        t = t[:max_lagtime]
        msd = msd[:max_lagtime]
        def linear_fn(x, m, b):
            return m * x + b
        linear_fn_v = np.vectorize(linear_fn)

        log_msd = np.nan_to_num(np.log(msd), nan=0.0, neginf=0.0)
        log_t = np.nan_to_num(np.log(t), nan=0.0, neginf=0.0)
        p0_K = np.nan_to_num(np.log(2*dim*0.2), nan=0.0, neginf=0.0)
        p0_a = 1

        popt, pcov = curve_fit(linear_fn, log_t, log_msd, p0=[p0_a, p0_K])
        residuals = log_msd - linear_fn_v(log_t, popt[0], popt[1])
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_msd-np.mean(log_msd))**2)
        r_squared = max(0, 1-(ss_res/ss_tot))

        alpha = popt[0]
        K = np.exp(popt[1])/(2*dim)

        return K, alpha, r_squared

    def step_size_all_tracks(self):
        if self.tracks is None:
            raise Exception(f"Error in step_size_all_tracks: track data is empty.")

        # init step size array
        self.step_sizes = np.zeros(shape=(self.tracks.shape[0], 6))
        self.step_sizes[:, 0] = self.tracks[:, 0]

        # ALL TRACKS
        for track_id in self.track_ids:
            self.step_sizes[self.step_sizes[:, 0] == track_id, 1:] = self.step_size(track_id)

        return self.step_sizes

    def r_of_g(self, track_id, full=True):
        # https://doi.org/10.1039/c0cp01805h
        # Calculate the radius of gyration for the track
        # if full is set to True, r-of-g will be returned for each step
        # if full is set to False, only return the r-of-g for the entire track
        if self.tracks is None:
            raise Exception(f"Error in radius_of_gyration: track data is empty.")

        if track_id not in self.track_ids:
            raise Exception(f"Error in radius_of_gyration: track_id not found.")

        traj = self.tracks[self.tracks[:, 0] == track_id]

        if full:
            r_of_g = np.zeros(shape=(traj.shape[0], 2))
            r_of_g[:, 0] = (traj[:, 1] - traj[0, 1]) * self.time_lag_sec

            for n in range(2, len(traj) + 1):
                # np.cov divides by (n-1) but I want to divide by n
                T = np.cov(traj[:n, 2:], rowvar=False) * (n - 1) / n

                # eigenvalues (w) are squared radii of gyration
                w, v = LA.eig(T)

                r_of_g[n - 1, 1] = np.sqrt(np.sum(w)) * self.microns_per_pixel

            return r_of_g

        else:
            n = len(traj)

            # np.cov divides by (n-1) but I want to divide by n
            T = np.cov(traj[:, 2:], rowvar=False) * (n - 1) / n

            # eigenvalues (w) are squared radii of gyration
            w, v = LA.eig(T)

            return np.sqrt(np.sum(w)) * self.microns_per_pixel

    def r_of_g_all_tracks(self, full=True):
        # Calculate the radius of gyration for all tracks
        # if full is set to True, r-of-g will be returned for each step
        # if full is set to False, only return the r-of-g for the entire track
        if self.tracks is None:
            raise Exception(f"Error in radius_of_gyration_all_tracks: track data is empty.")

        if full:
            # init array
            self.radius_of_gyration = np.zeros(shape=(self.tracks.shape[0], 3))
            self.radius_of_gyration[:, 0] = self.tracks[:, 0]

            # R of G ALL TRACKS
            for track_id in self.track_ids:
                self.radius_of_gyration[self.radius_of_gyration[:, 0] == track_id, 1:] = self.r_of_g(track_id)

        else:
            # init array
            self.radius_of_gyration = np.zeros(shape=(self.track_ids.shape[0], 2))
            self.radius_of_gyration[:, 0] = self.track_ids

            # R of G ALL TRACKS
            for i, track_id in enumerate(self.track_ids):
                self.radius_of_gyration[i, 0] = track_id
                self.radius_of_gyration[i, 1] = self.r_of_g(track_id, full)

        return self.radius_of_gyration

    def msd_all_tracks(self, fft=True):
        if self.tracks is None:
            raise Exception(f"Error in msd_all_tracks: track data is empty.")

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

    def _fit_msd_all_tracks(self, params):
        track_ids, msds, linear_fit, dim, max_lagtime, err = params

        if dim == 'sum':
            d = self.dimension
        else:
            d = 1

        fit_results = np.zeros(shape=(len(track_ids), 5))
        for i, track_id in enumerate(track_ids):
            msd = msds[msds[:, 0] == track_id]
            if linear_fit:
                D, E, r_squared = self.fit_msd_linear(msd[1:, 1], msd[1:, 2], d, max_lagtime=max_lagtime, err=err)
                fit_results[i] = np.asarray([track_id, ParticleTracks.dim_columns[dim], D, E, r_squared])
            else:
                K, alpha, r_squared = self.fit_msd_loglog(msd[1:, 1], msd[1:, 2], d, max_lagtime=max_lagtime)
                fit_results[i] = np.asarray([track_id, ParticleTracks.dim_columns[dim], K, alpha, r_squared])

        return fit_results

    def fit_msd_all_tracks(self, linear_fit=True, min_len=11, max_lagtime=10, err=True, all_dims=False):
        """
        Fits the Mean Square Displacement (MSD) for all tracks either to a linear function or a log-log function.

        This method takes a set of tracks and fits them to a linear or log-log model to find the
        diffusion coefficients and/or the anomalous exponent. Tracks that don't meet the minimum
        length are ignored. The results are stored in the instance variables linear_fit_results or
        loglog_fit_results, depending on the selected model.

        Parameters:
        linear_fit : bool, optional
            If True, uses a linear model; if False, uses a log-log model (default is True).
        min_len : int, optional
            Minimum length of tracks to be considered for the fit (default is 11).
        max_lagtime : int, optional
            Maximum lag time to consider for the fit (default is 10).
        err : bool, optional
            If True, includes error in the linear model (ignored for log-log model) (default is True).
        all_dims : bool, optional
            If True, fits the model separately for each dimension; if False, uses the sum of all dimensions
            (default is False).

        Raises:
        Exception:
            If MSD data is empty.

        Returns:
        numpy.ndarray: results
            An array containing the fit results for each track. Each row represents one track and contains:
            - Track ID
            - Dimension ('sum if all_dims == False or 'x', 'y', 'z', and 'sum' if all_dims == True)
            - Diffusion coefficient (linear) or Generalized Diffusion coefficient (log-log)
            - Error (linear) or anomalous exponent (log-log)
            - R-squared statistic representing the goodness of fit

        Note:
        Uses scipy's curve_fit for the least-squares minimization and multiprocessing for efficiency.
        """
        if self.msds is None:
            raise Exception(f"Error in fit_msd_all_tracks: msd data is empty.")

        # subset track_ids that reach min_len
        track_ids = self.track_lengths[self.track_lengths[:, 1] >= min_len, 0]

        # fit for each dimension, then fill array - use multiprocessing
        msds_list = []
        dim_list = []
        if all_dims:
            for dim in ParticleTracks.dim_columns.keys():
                msds = self.msds[:, [0, 1, ParticleTracks.dim_columns[dim]]]
                if len(np.unique(msds[:, 2])) > 1:
                    msds_list.append(msds)
                    dim_list.append(dim)
        else:
            msds_list.append(self.msds[:, [0, 1, ParticleTracks.dim_columns['sum']]])
            dim_list.append('sum')

        group_size = len(msds_list)
        params_arr = list(zip([track_ids] * group_size,
                              msds_list,
                              [linear_fit] * group_size,
                              dim_list,
                              [max_lagtime] * group_size,
                              [err] * group_size))
        with Pool(group_size) as p:
            results_list = p.map(self._fit_msd_all_tracks, params_arr)

            # fill class variable and sort by track id
            results_list = np.concatenate(results_list, axis=0)
            if all_dims:
                results_list = results_list[np.lexsort((results_list[:, 1], results_list[:, 0]))]

        if linear_fit:
            self.linear_fit_results = results_list
            return self.linear_fit_results
        else:
            self.loglog_fit_results = results_list
            return self.loglog_fit_results

    ####################################################################################################################
    # def find_track_intensities(self, movie, radius=3):
    #     """
    #         Compute mean and sum of track intensities.  (1) Fills the class variable, 'track_intensities' with mean and
    #         sum intensity of a disk with given radius at each track position; it is a Nx3 numpy array, column positions
    #         are: [track_id, mean_intensity, sum_intensity].  (2) Fills the class variable, 'mean_track_intensities' with
    #         the mean track intensity for each track; it is a Nx3 numpy array, column positions are:
    #         [track_id, mean_intensity, stdev_intensity]
    #
    #         Parameters
    #         ----------
    #         movie : (t, N, M) ndarray
    #             Movie to extract intensity information.  first dimension corresponds to time (frames) of movie
    #
    #         radius : int
    #             Disk radius around each point for intensity calculation
    #
    #         Returns
    #         -------
    #         (n, 2) ndarray: (n=number of tracks), columns are [track_id, mean_intensity, stdev_intensity]
    #     """
    #
    #     if self.tracks is None:
    #         raise Exception(f"Error find_track_intensities: track data is empty.")
    #
    #     if (type(np.array([])) != type(movie)) or (len(movie.shape) != 3):
    #         raise Exception(f"Error find_track_intensities: movie is not of valid format.")
    #
    #     if self.tracks[:, 1].max() >= movie.shape[0]:
    #         raise Exception("Error find_track_intensities: track frames exceed size of image first dimension.")
    #
    #     self.mean_track_intensities = np.zeros(shape=(len(self.track_ids), 3), )
    #     self.track_intensities = np.zeros(shape=(self.tracks.shape[0], 3), )
    #
    #     index = 0
    #     for i, track_id in enumerate(self.track_ids):
    #         cur_track = self.tracks[self.tracks[:, 0] == track_id]
    #         for j in range(cur_track.shape[0]):
    #             rr, cc = draw.disk((int(cur_track[j][3]),
    #                                 int(cur_track[j][4])),
    #                                radius,
    #                                shape=movie.shape[1:])
    #
    #             self.track_intensities[index][0] = track_id
    #             self.track_intensities[index][1] = np.mean(movie[int(cur_track[j][1])][rr, cc])
    #             self.track_intensities[index][2] = np.sum(movie[int(cur_track[j][1])][rr, cc])
    #             index += 1
    #
    #         self.mean_track_intensities[i][0] = track_id
    #         self.mean_track_intensities[i][1] = np.mean(self.track_intensities[index-cur_track.shape[0]:index, 1])
    #         self.mean_track_intensities[i][2] = np.std(self.track_intensities[index-cur_track.shape[0]:index, 1])
    #
    #     return self.mean_track_intensities
    #
    # def angle(self, track_id):
    #     if self.tracks is None:
    #         raise Exception(f"Error in angle: track data is empty.")
    #
    #     if track_id not in self.track_ids:
    #         raise Exception(f"Error in angle: track_id not found.")
    #
    #     traj = self.tracks[self.tracks[:, 0] == track_id]










