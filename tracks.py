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

    file_formats = ['mosaic', 'trackmate', 'trackpy']
    file_columns = {'mosaic': ['trajectory', 'frame', 'y', 'x', 'z'],
                    'trackmate': ['track_id', 'frame', 'position_y', 'position_x', 'position_z'],
                    'trackpy': ['particle', 'frame', 'y', 'x', 'z']}
    skip_rows = {'mosaic': None, 'trackmate': [1, 2, 3], 'trackpy': None}

    def __init__(self, tracks=None):
        """
            Initialize class instance.

            Parameters
            ----------
            tracks : numpy 2d array
            the track data with columns in this order: Track ID, Frame, z, y, x (optional)

            Returns
            -------
            None
        """

        if tracks is not None:
            if (type(np.array([])) == type(tracks)) and (len(tracks.shape) == 2) and (tracks.shape[1] >= 5):
                # format looks correct
                self.tracks = tracks
            else:
                raise Exception(f"Error initializing track data: invalid format.")
        else:
            self.tracks = None

        self.track_data = None
        self.track_lengths = None
        self.mean_track_intensities = None
        self.track_intensities = None

    def read_track_file(self, path, file_format='mosaic', sep=','):
        """
            Read track data from a file.

            Reads the track data from Mosaic, Trackmate or trackpy format files.  Track data will be extracted and
            stored as a numpy array in the class variable, 'tracks'.

            Parameters
            ----------
            path : str
                Full path to the track file.
            file_format : str
                String indicating type of file: 'mosaic', 'trackpy' or 'trackmate'
            sep : str
                String indicating field separator for file, e.g. ',' or '\t'

            An Exception is raised if the file format is invalid or file_format is not one of:
            'mosiac', 'trackmate', 'trackpy'

            Returns
            -------
            (n, 5) ndarray: (n=number of tracks), column order is [track_id, t, z, y, x]
        """

        if not (file_format in ParticleTracks.file_formats):
            raise Exception(f"Error in reading {path}: file format '{file_format}' is not recognized.")

        self.track_data = pd.read_csv(path, sep=sep, header=0, skiprows=ParticleTracks.skip_rows[file_format])
        for col in ParticleTracks.file_columns[file_format][:-1]:
            if not (col in self.track_data.columns):
                raise Exception(
                    f"Error in reading {path}: required column {col} is missing for file format {file_format}.")
        if not (ParticleTracks.file_columns[file_format][-1] in self.track_data.columns):
            self.track_data.columns[ParticleTracks.file_columns[file_format][-1]] = 0

        self.tracks = self.track_data[ParticleTracks.file_columns[file_format]].to_numpy()
        return self.tracks

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
            self.track_lengths = np.unique(self.tracks[:, 0], return_counts=True)
        else:
            raise Exception(f"Error find_track_lengths: track data is empty.")

        return self.track_lengths

    def find_track_intensities(self, movie, radius):
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
        self.mean_track_intensities = np.zeros(shape=(len(track_ids), 3), )
        self.track_intensities = np.zeros(shape=(len(track_ids), 3), )

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
