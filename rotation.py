"""
Rotation utilities for coordinate transformations.
"""

import numpy as np


class Rotation:
    """
    Static methods for rotating arrays and coordinate transformations.
    """

    @classmethod
    def rotate_2d(cls, array, angle):
        """
        Rotate (n, 2) array around origin by angle.

        Parameters
        ----------
        array : np.ndarray, shape (n, 2)
            Array of 2D points
        angle : float
            Rotation angle in radians

        Returns
        -------
        np.ndarray, shape (n, 2)
            Rotated array
        """
        array = array if isinstance(array, np.ndarray) else np.asarray(array)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return array @ rotation_matrix.T

    @classmethod
    def rotate_3d_x(cls, array, angle):
        """
        Rotate (n, 3) array around origin by angle about x.

        Parameters
        ----------
        array : np.ndarray, shape (n, 3)
            Array of 3D points
        angle : float
            Rotation angle in radians

        Returns
        -------
        np.ndarray, shape (n, 3)
            Rotated array
        """
        array = array if isinstance(array, np.ndarray) else np.asarray(array)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
        return array @ rotation_matrix.T

    @classmethod
    def rotate_3d_y(cls, array, angle):
        """
        Rotate (n, 3) array around origin by angle about y.

        Parameters
        ----------
        array : np.ndarray, shape (n, 3)
            Array of 3D points
        angle : float
            Rotation angle in radians

        Returns
        -------
        np.ndarray, shape (n, 3)
            Rotated array
        """
        array = array if isinstance(array, np.ndarray) else np.asarray(array)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        return array @ rotation_matrix.T

    @classmethod
    def rotate_3d_z(cls, array, angle):
        """
        Rotate (n, 3) array around origin by angle about z.

        Parameters
        ----------
        array : np.ndarray, shape (n, 3)
            Array of 3D points
        angle : float
            Rotation angle in radians

        Returns
        -------
        np.ndarray, shape (n, 3)
            Rotated array
        """
        array = array if isinstance(array, np.ndarray) else np.asarray(array)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        return array @ rotation_matrix.T
