import numpy as np
import math

from typing import List


"""
Vector class with length=1
"""


class Vector:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        try:
            self.z = math.sqrt(1 - self.x**2 - self.y**2)
        except ValueError:
            print("[WARNING] z is set to None")
            self.z = None

    @staticmethod
    def spherical2cartesian(azimuth, elevation, radius):
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)

        x = radius * np.sin(elevation_rad) * np.cos(azimuth_rad)
        y = radius * np.sin(elevation_rad) * np.sin(azimuth_rad)

        return Vector(x, y)


    def toSpherical(self):

        elevation = math.acos(self.z)
        azimuth = math.atan2(self.y, self.x)

        return np.degrees(elevation), np.degrees(azimuth)

    @classmethod
    def from_iterable(cls, array: np.ndarray) -> "Vector":
        return cls(array[0], array[1])

    @staticmethod
    def angle_between_vectors(first_vector: "Vector", second_vector: "Vector", degrees: bool = True) -> float | None:
        if isinstance(first_vector, list):
            first_numpy = np.array(first_vector)
            second_numpy = np.array(second_vector)
        else:
            first_numpy = first_vector.to_numpy()
            second_numpy = second_vector.to_numpy()

        try:
            angle = np.arccos(np.dot(first_numpy, second_numpy))
        except:
            return None
        if not degrees:
            return angle
        return np.rad2deg(angle)

    def to_list(self) -> List:
        return [self.x, self.y, self.z]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list())

    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        return f"[{self.x}, {self.y}, {self.z}]"
