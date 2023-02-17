"""
This file contains tracking related functionality in an implementation of the ParticleTracks class.

Internally, track data is stored as a numpy array in the napari format for the 'tracks' layer.
See: https://napari.org/stable/howtos/layers/tracks.html#tracks-data

The class can be initialized with a numpy array of track data
It also has functions to:
(1) Read in tracking data from a file.  Support for Mosaic and Trackmate file formats is supported.
(2) Perform particle detection and tracking using trackpy

"""

import os
import pandas as pd
import numpy as np
from skimage import io
import trackpy as tp


class ParticleTracks:

    read_file_formats = ['mosaic', 'trackmate']
    file_extensions = {'mosaic': 'csv', 'trackmate': 'csv'}
    ext_sep = {'csv': ',', 'txt': '\t'}
    tracks = np.asarray([])
    track_data = pd.DataFrame()

    def init(self, tracks=np.asarray([])):
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

        self.tracks = tracks

    def detect_and_track(self, movie, size=11):
        f = tp.locate(movie[0], size)

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









