import numpy as np
import math


def spherical2cartesian(azimuth, elevation, radius):
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)

    x = radius * np.sin(elevation_rad) * np.cos(azimuth_rad)
    y = radius * np.sin(elevation_rad) * np.sin(azimuth_rad)
    z = np.sqrt(1 - x**2 - y**2)

    return [x, y, z]


def cartesian2spherical(x, y):
    z = np.sqrt(1 - x**2 - y**2)
    elevation = math.acos(z)
    azimuth = math.atan2(y, x)

    return np.degrees(elevation), np.degrees(azimuth)


def reinterpret(point_arr: np.ndarray) -> np.ndarray:
    angle_arr = np.empty_like(prototype=point_arr, dtype=float)

    for index, point in enumerate(point_arr):
        el, az = cartesian2spherical(point[0], point[1])
        if az < 0:
            az = abs(az) + 180
        angle_arr[index] = (abs(el), abs(az))

    return angle_arr
