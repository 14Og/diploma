import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import yaml
from typing import List, Tuple, Iterable, Dict
from functools import cmp_to_key
from skimage.measure import CircleModel
from collections import namedtuple


class GridCreator:
    """
    Utility class for manipulations with calibration grids
    (used in YAML format) for calibration stand application.
    Consist mostly of static methods bc no instances are created
    in main application.
    """

    def __init__(
        self,
        pathname: str,
        filename: str = "",
        max_el: int = 45,
        cartesian_linspace_size=32,
    ):
        self.pathname = pathname
        self.filename = (
            filename
            or f"{0}_{max_el}"
            + f"_{0}_{360}_size_{cartesian_linspace_size}x{cartesian_linspace_size}"
        )
        self.x_linspace = self._get_linspace(
            size=cartesian_linspace_size, max_el=max_el
        )

        self.y_linspace = self._get_linspace(
            size=cartesian_linspace_size, max_el=max_el
        )

        self.cartesian_sample_space, c_grid1, c_grid2 = self._get_sample_space(
            self.x_linspace, self.y_linspace
        )

        self.cartesian_grid, self.spherical_grid = self.create_stand_calibration_data(
            self.cartesian_sample_space, max_el=max_el
        )

    @staticmethod
    def _get_sample_space(
        linspace1: np.ndarray, linspace2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # sample_space, grid1, grid2
        """Creates sample space (grid before filtering).
        Args:
            linspace1 (np.ndarray): first linspace (X or Y)
            linspace2 (np.ndarray): second linspace (X or Y, respectively)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: spherical sample space and 2 meshgrids
        """
        grid1, grid2 = np.meshgrid(linspace1, linspace2)
        spherical_sample_space = list(
            map(tuple, np.column_stack((grid1.ravel(), grid2.ravel())))
        )
        return np.array(spherical_sample_space), grid1, grid2

    @staticmethod
    def _get_linspace(size: int, max_el: int) -> np.ndarray:
        """Creates numpy linspace according to maximum elevation angle and size of grid.

        Args:
            size (int): size of grid
            max_el (int): maximum elevation angle as grid boundary

        Returns:
            np.ndarray: linspace for grid
        """
        _max = math.sin(math.radians(max_el))
        _min = -_max
        _range = np.linspace(_min, _max, size)
        return _range

    @staticmethod
    def _cartesian_grid_to_spherical(
        x: float,
        y: float,
        z: float | None = None,
    ) -> List[float]:
        """
        Transforms X,Y,Z coordinated of sun vector to spherical angles,
        understandable for calibration stand.

        Args:
            x (float): vector coordinate
            y (float): vector coordinate
            z (float | None, optional): vector coordinate. Defaults to None.

        Returns:
            List[float]: azimuth and elevation angles.
        """
        if not z:
            z = (1 - x**2 - y**2) ** 0.5
        elevation = math.degrees(math.acos(z))
        azimuth = math.atan2(y, x)
        azimuth = math.degrees(2 * math.pi + azimuth if azimuth < 0 else azimuth)

        azimuth = round(azimuth, 4)
        elevation = round(elevation, 4)
        return [azimuth, elevation]

    @staticmethod
    def create_second_run_grid(
        interpolated_frame: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Creates data for second stand run. Simply adds spherical angles to dataframe.
        Input dataframe have to be interpolated and sorted by tsp before using this method.

        Args:
            interpolated_frame (pd.DataFrame): Interpolated by first stage frame that sorted by tsp

        Returns:
            pd.DataFrame: Dataframe ready to dump into yaml.
        """
        spherical_angles = []
        second_run_grid = pd.DataFrame.copy(interpolated_frame, deep=True)
        for X, Y in zip(
            second_run_grid["X"].to_numpy(),
            second_run_grid["Y"].to_numpy(),
        ):
            spherical_angles.append(GridCreator._cartesian_grid_to_spherical(X, Y))
        spherical_angles = np.array(spherical_angles)
        second_run_grid["elevation"] = spherical_angles[:, 1]
        second_run_grid["azimuth"] = spherical_angles[:, 0]

        return second_run_grid

    @staticmethod
    def create_regular_sensor_squared_grid(
        first_run_data: pd.DataFrame, binary_degree: int = 8
    ) -> np.ndarray:
        """
        Creates squared grid with given dataframe.

        Args:
            first_run_data (pd.DataFrame): Dataframe from first run.
            binary_degree (int, optional): Binary degree of created grid (power of 2 / 10000). Defaults to 8.

        Returns:
            np.ndarray: Regular sensor grid with squared shape and step = 2**binary_degree / 10000.
            Normalizes shape if x and y shapes are not equal
        
        """
        binary_step_f = 2**binary_degree / 10000.0
        binary_rounder = lambda num: (
            binary_step_f
            * (
                (math.ceil(num / binary_step_f))
                if num > 0
                else (math.floor(num / binary_step_f))
            )
        ).__round__(4)

        x_data = sorted(first_run_data["x_light"].to_numpy())
        y_data = sorted(first_run_data["y_light"].to_numpy())

        max_x = binary_rounder(np.mean(x_data[-1:-6:-1]))
        min_x = binary_rounder(np.mean(x_data[0:5]))
        max_y = binary_rounder(np.mean(y_data[-1:-6:-1]))
        min_y = binary_rounder(np.mean(y_data[0:5]))
        
        print("Boundaries: ", min_x, max_x, min_y, max_y)
        
        size_x = int(((max_x - min_x) / binary_step_f).__round__(4) + 1)
        size_y = int(((max_y - min_y) / binary_step_f).__round__(4) + 1)
        
        print(f"Size x: {size_x}, Size y: {size_y}")
        
        if size_x != size_y:
            if size_x < size_y:
                if max_x <= abs(min_x):
                    max_x += binary_step_f
                else:
                    min_x -= binary_step_f
            elif size_y < size_x:
                if max_y <= abs(min_y):
                    max_y += binary_step_f
                else:
                    min_y -= binary_step_f

            size_x = int(((max_x - min_x) / binary_step_f).__round__(4) + 1)
            size_y = int(((max_y - min_y) / binary_step_f).__round__(4) + 1)
            min_x = round(min_x, 4)
            max_x = round(max_x, 4)
            min_y = round(min_y, 4)
            max_y = round(max_y, 4)
            
            print(
                f"Size x after correction: {size_x}, Size y after correction: {size_y}"
            )

        x_range = np.linspace(
            min_x,
            max_x,
            size_y,
            endpoint=True
        )
        y_range = np.linspace(
            min_y,
            max_y,
            size_y,
            endpoint=True
        )
        print(x_range, y_range, sep="\n")
        X, Y = np.meshgrid(x_range, y_range[::-1])

        regular_sensor_grid = np.column_stack((X.ravel(), Y.ravel()))
        return regular_sensor_grid

    def save_gcode_grid(self, spherical_grid: Dict, pathname: str) -> None:
        """
        Save given grid co gcode to test speed and acceleration of stand
        Args:
            spherical_grid (Dict): Spherical grid with elevation as key and azimuth as value.
            pathname (str): Path where to save it
        """
        with open(pathname + ".gcode", "w") as gcode_file:
            for elevation, azimuth in spherical_grid:
                calibration_node = f"GRID_CALIBRATION ELEVATION={elevation} AZIMUTH={azimuth}\nG4 P200\n"
                gcode_file.write(calibration_node)

    def create_stand_calibration_data(
        self, cartesian_sample_space: np.ndarray, max_el: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates X-Y stand normalized grid for first run.

        Args:
            cartesian_sample_space (np.ndarray): Sample space of nodes where stand have to go. X-Y.
            max_el (int): Maximum elevation angle which stand will reach in first run.

        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple with filtered and sorted grid in X-Y coordinates
            and spherical angles.
        """
        radius = math.sin(math.radians(max_el))
        spherical_grid = []
        cartesian_grid = []
        for pair in cartesian_sample_space:
            x = pair[0]
            y = pair[1]
            if math.sqrt(x**2 + y**2) <= radius:
                z = math.sqrt(1 - x**2 - y**2)
                cartesian_grid.append([x, y, z])
                spherical_grid.append(
                    self._cartesian_grid_to_spherical(
                        x, y, z, match_to_resolution=False
                    )
                )

        def sorter(left, right):
            if left[1] == right[1]:
                if left[-1]:
                    if left[0] > right[0]:
                        return 1
                    else:
                        return -1
                else:
                    if left[0] > right[0]:
                        return -1
                    else:
                        return 1
            else:
                return 1 if left[1] > right[1] else -1

        spherical_dict = {}
        for az, el in spherical_grid:
            if el in spherical_dict:
                spherical_dict[el].append(az)
            else:
                spherical_dict[el] = [az]
        i = 1
        spherical_grid.clear()
        for elevation, azimuths in spherical_dict.items():
            for az in azimuths:
                spherical_grid.append([az, elevation, i % 2 == 0])
            i += 1

        spherical_grid.sort(key=cmp_to_key(sorter))
        spherical_grid_dict = [
            {"elevation": value[1], "azimuth": value[0]} for value in spherical_grid
        ]
        with open(
            os.path.join(self.pathname, self.filename + "yaml"), "w"
        ) as grid_file:
            grid_file.write(
                yaml.dump(data=spherical_grid_dict, default_flow_style=False)
            )
        spherical_grid = np.array(spherical_grid)
        spherical_grid = spherical_grid[:, 0:2]
        cartesian_grid = np.array(cartesian_grid)

        return cartesian_grid, spherical_grid

    @staticmethod
    def calculate_light_shape(
        data: pd.DataFrame, mode="max coordinates", points_counter=20
    ) -> Tuple[float, float, float]:  # xc, yc, r
        """
        Calculates first run sensor measurments shape as a cirle.
        Uses lss method based on several modes using different approaches
        to first run data. (deprecated)

        Args:
            data (pd.DataFrame): Dataframe from first stand run
            mode (str, optional): Mode for calculating. Defaults to "max coordinates".
            points_counter (int, optional): How many points will be used in calculation. Defaults to 20

        Raises:
            Exception: If linear model could not calculate light shape

        Returns:
            Tuple[float, float, float]: xc, yc and r of calculated cicrle
        """
        light = data.loc[:, "x_light":"y_light"].to_numpy()
        max_x = light[np.argsort(light[:, 0])][-1 : -(points_counter + 1) : -1]
        min_x = light[np.argsort(light[:, 0])][0:points_counter]
        max_y = light[np.argsort(light[:, 1])][-1 : -(points_counter + 1) : -1]
        min_y = light[np.argsort(light[:, 1])][0:points_counter]
        light_shape = CircleModel()
        if mode == "max coordinates":
            success = light_shape.estimate(
                np.vstack((max_x, min_x, max_y, min_y))
            )  # estimation by max coordinates

        elif mode == "radius":
            sorted_by_radius = np.array(
                sorted(
                    light, key=lambda h: (h[0] ** 2 + h[1] ** 2) ** 0.5, reverse=True
                )[0:points_counter]
            )
            success = light_shape.estimate(sorted_by_radius)  # estimation by radiuses

        elif mode == "last points":
            success = light_shape.estimate(
                data.loc[
                    data.shape[0] - points_counter :,
                    "x_light":"y_light",
                ].to_numpy()
            )  # estimation by last points

        if not success:
            raise Exception("Could not fit circle to light sensor data")

        LightParams = namedtuple("LightParams", ["xc", "yc", "r"])
        mean_xc, mean_yc = data["x_light"].mean(), data["y_light"].mean()
        return LightParams(*light_shape.params)

    @staticmethod
    def save_grid(
        grid: pd.DataFrame,
        path: str,
        name: str,
        mode: str = "XY",
        left_bounds: bool = False,
    ) -> None:
        """
        Saves grid into yaml file.

        Args:
            grid (pd.DataFrame): Grid to save. Contains elevation and azimuth or X and Y.
            path (str): Where to save grid
            name (str): Name of saving file
            mode (str, optional): Which parametes shoud be taken. Defaults to "XY".
            left_bounds (bool, optional): True to save left bounds of grid for binary file. 
            Defaults to False.
        """
        parameter_1 = "X" if mode == "XY" else "elevation"
        parameter_2 = "Y" if mode == "XY" else "azimuth"
        grid_list = grid.loc[:, [parameter_1, parameter_2]].to_numpy().tolist()
        grid_dict = [{parameter_1: v[0], parameter_2: v[1]} for v in grid_list]

        with open(os.path.join(path, name) + ".yaml", "w") as grid_file:
            if left_bounds:
                bounds = [
                    {
                        "X": float(grid["x_light"].min()),
                        "Y": float(grid["y_light"].min()),
                    }
                ]
                grid_file.write(yaml.dump(bounds))
            grid_file.write(yaml.dump(grid_dict))
