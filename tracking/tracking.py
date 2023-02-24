"""
This file contains tracking related functionality in an implementation of the ParticleTracks class.

Internally, track data is stored as a numpy array in the napari format for the 'tracks' layer.
See: https://napari.org/stable/howtos/layers/tracks.html#tracks-data

The class can be initialized with a numpy array of track data.  It also has a function to read in tracking data from a
file.  Mosaic, Trackmate and trackpy file formats are supported.

"""

import os
import pandas as pd
import numpy as np
from skimage import io, draw


class ParticleTracks:

    read_file_formats = ['mosaic', 'trackmate']
    file_extensions = {'mosaic': 'csv', 'trackmate': 'csv'}
    ext_sep = {'csv': ',', 'txt': '\t'}
    tracks = np.asarray([])
    track_data = pd.DataFrame()

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

        self.track_lengths = None
        self.track_intensities = None

    def read_track_file(self, path, file_format='mosaic'):
        """
            Read track data from a file.

            Reads the track data from Mosaic (csv) or Trackmate (csv) files.  Track data will be extracted and stored
            as a numpy array.

            Parameters
            ----------
            path : str
                Full path to the track file.
            file_format : str
                String indicating type of file: 'mosaic' or 'trackmate'

            An Exception is raised if the file format is invalid or file_format is not one of: 'mosiac', 'trackmate'.

            Returns
            -------
            None
        """

        # Check for valid extension
        ext = os.path.splitext(path)[1]
        if ext != self.file_extensions[file_format]:
            raise Exception(f"Error in reading {path}: invalid extension for {file_format} file format.")

        # Read file
        if file_format == 'mosaic':
            self.track_data = pd.read_csv(path, sep=self.ext_sep(ext))
            for col in ['Trajectory', 'frame', 'z', 'y', 'x']:
                if not (col in self.track_data.columns):
                    raise Exception(
                        f"Error in reading {path}: required column {col} is missing for file format {file_format}.")
            self.tracks = self.track_data[['Trajectory', 'frame', 'z', 'y', 'x']].to_numpy()

        elif file_format == 'trackmate':
            # First 4 rows of trackmate file is header information
            self.track_data = pd.read_csv(path, sep=self.ext_sep(ext), header=0, skiprows=[1, 2, 3])
            for col in ['TRACK_ID', 'FRAME', 'POSITION_Z', 'POSITION_Y', 'POSITION_X']:
                if not (col in self.track_data.columns):
                    raise Exception(
                        f"Error in reading {path}: required column {col} is missing for file format {file_format}.")
            self.tracks = self.track_data[['TRACK_ID', 'FRAME',
                                           'POSITION_Z', 'POSITION_Y', 'POSITION_X']].to_numpy()

        else:
            raise Exception(f"Error in reading {path}: {file_format} is an invalid file format")

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
            Compute track lengths.

            Fills the track_lengths numpy array.

            Parameters
            ----------
            None

            Returns
            -------
            (n, 2) ndarray: (n=number of tracks), columns are [track_id, track-length]
        """
        if self.tracks is not None:
            self.track_lengths = np.unique(self.tracks[:, 0], return_counts=True)
        else:
            raise Exception(f"Error track_lengths: track data is None.")
        return self.track_lengths

    def find_track_intensities(self, image, radius):
        """
            Compute mean track intensities.

            Fills the track_intensities numpy array.

            Parameters
            ----------
            image : (t, N, M) ndarray
                image to extract intensity information.  first dimension corresponds to time (frames) of movie

            radius : int
                Disk radius around each point for intensity calculation

            Returns
            -------
            (n, 2) ndarray: (n=number of tracks), columns are [track_id, mean-intensity]
        """

        if (type(np.array([])) == type(image)) and (len(image.shape) == 3):
            if self.tracks[:, 1].max() >= image.shape[0]:
                raise Exception("Error find_track_intensities: track frames exceed size of image first dimension.")
            track_ids = np.unique(self.tracks[:, 0])
            self.track_intensities = np.zeros(shape=(len(track_ids), 2), )

            for i, track_id in enumerate(track_ids):
                cur_track = self.tracks[self.tracks[:, 0] == track_id]
                intensity_list = []
                for j in range(cur_track.shape[0]):
                    rr, cc = draw.disk((int(cur_track[j][3]),
                                        int(cur_track[j][4])),
                                       radius,
                                       shape=image.shape[1])

                    intensity_list.append(np.mean(image[int(cur_track[j][1])][rr, cc]))

                self.track_intensities[i][0] = track_id
                self.track_intensities[i][1] = np.mean(intensity_list)
                self.track_intensities[i][2] = np.std(intensity_list)
        else:
            raise Exception(f"Error find_track_intensities: image is not of valid format.")

        return self.track_intensities

    def filter_tracks(self, by="length", cutoff=11):
        """
            Filters tracks by track length.

            Track data will be filtered and any tracks less than min_len will be removed.

            Parameters
            ----------
            by : str
                Type of filtering applied.  Allowed values are: length, mean_intensity

            cutoff : float
                Filter cutoff to apply: if by == "length", cutoff is the min. track length
                If by == "mean_intensity", cutoff is the min. mean_intensity

            Returns
            -------
            None
        """

        if by == "length":
            if self.track_lengths is not None:
                self.tracks = self.tracks[np.isin(self.tracks[:, 0],
                                                  self.track_lengths[self.track_lengths[:, 1] >= cutoff][:, 0])]
            else:
                raise Exception(f"Error filter_tracks: track_lengths is None.")

        elif by == "mean_intensity":
            if self.track_intensities is not None:
                self.tracks = self.tracks[np.isin(self.tracks[:, 0],
                                                  self.track_intensities[self.track_intensities[:, 1] >= cutoff][:, 0])]
            else:
                raise Exception(f"Error filter_tracks: track_intensities is None.")

        else:
            raise Exception(f"Error filter_tracks: invalid value for 'by' parameter.")








