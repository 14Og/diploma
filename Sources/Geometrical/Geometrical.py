import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(path))


from Vector import Vector
from CalibrationPerformer import AbstractCalibrationPerformer

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from enum import Enum

from typing import Tuple, List, Any, final


class GeometricalTransformation:
    @staticmethod
    def transform_unbiased(sensor_coordinate, d0) -> float:
        return -np.arcsin(np.arctan(sensor_coordinate / d0))

    @staticmethod
    def transform_with_linear_offset(sensor_coordinate, d0, center_offset) -> float:
        return np.arcsin(np.arctan((center_offset - sensor_coordinate) / d0))

    @staticmethod
    def transform_with_all_offsets_x(
        sensor_coordinate, d0, center_offset_x, center_offset_y, alpha0, beta0, phi0
    ) -> float:
        rotation_matrix = np.array([np.cos(phi0), -np.sin(phi0), np.sin(phi0), np.cos(phi0)]).reshape(2, 2)
        rotated_xc0, rotated_yc0 = rotation_matrix @ np.array([center_offset_x, center_offset_y])
        return np.arcsin(np.arctan(((rotated_xc0 - sensor_coordinate) / d0) * np.cos(beta0)) + alpha0)

    @staticmethod
    def transform_with_all_offsets_y(
        sensor_coordinate, d0, center_offset_x, center_offset_y, alpha0, beta0, phi0
    ) -> float:
        rotation_matrix = np.array([np.cos(phi0), -np.sin(phi0), np.sin(phi0), np.cos(phi0)]).reshape(2, 2)
        rotated_xc0, rotated_yc0 = rotation_matrix @ np.array([center_offset_x, center_offset_y])
        return np.arcsin(np.arctan(((rotated_yc0 - sensor_coordinate) / d0) * np.cos(alpha0)) + beta0)

    @staticmethod
    def unbiased_loss_function(d0, sensor_x, sensor_y, X, Y) -> float:
        return (X - GeometricalTransformation.transform_unbiased(sensor_x, d0)) ** 2 + (
            Y - GeometricalTransformation.transform_unbiased(sensor_y, d0)
        ) ** 2

    @staticmethod
    def linear_offstes_loss_function(args, sensor_x, sensor_y, X, Y) -> float:
        d0, xc0, yc0 = args
        return (
            X
            - GeometricalTransformation.transform_with_linear_offset(
                sensor_coordinate=sensor_x, d0=d0, center_offset=xc0
            )
        ) ** 2 + (
            Y
            - GeometricalTransformation.transform_with_linear_offset(
                sensor_coordinate=sensor_y, d0=d0, center_offset=yc0
            )
        ) ** 2

    @staticmethod
    def all_offsets_loss_function(args, sensor_x, sensor_y, X, Y) -> float:
        d0, xc0, yc0, alpha0, beta0, phi0 = args
        return (
            X
            - GeometricalTransformation.transform_with_all_offsets_x(
                sensor_coordinate=sensor_x,
                d0=d0,
                center_offset_x=xc0,
                center_offset_y=yc0,
                alpha0=alpha0,
                beta0=beta0,
                phi0=phi0,
            )
        ) ** 2 + (
            Y
            - GeometricalTransformation.transform_with_all_offsets_y(
                sensor_coordinate=sensor_y,
                d0=d0,
                center_offset_x=xc0,
                center_offset_y=yc0,
                alpha0=alpha0,
                beta0=beta0,
                phi0=phi0,
            )
        ) ** 2


class GeometricalFitter(AbstractCalibrationPerformer):

    class OffsetsType(Enum):
        NO_OFFSTES = (1,)
        LINEAR_OFFSETS = (2,)
        ALL_OFFSETS = 3

    def __init__(self, module_path=path):
        super().__init__(module_path=module_path)
        self.d0: float | None = 0.2
        self.xc0: float | None = -0.004
        self.yc0: float | None = 0.003
        self.alpha0: float | None = 0
        self.beta0: float | None = 0
        self.phi0: float | None = 0

    def __get_calibration_parameters(self, panel_name: str, offset_type: OffsetsType) -> Tuple:
        initial_data = pd.concat([self.data_base[panel_name].initial_data, 
                                self.data_base[panel_name].second_run_data])
        sensor_x = initial_data["x_light"]
        sensor_y = initial_data["y_light"]
        X = initial_data["X"]
        Y = initial_data["Y"]

        match offset_type:
            case self.OffsetsType.NO_OFFSTES:
                return least_squares(
                    fun=GeometricalTransformation.unbiased_loss_function,
                    x0=self.d0,
                    args=(sensor_x, sensor_y, X, Y),
                ).x[0]
            case self.OffsetsType.LINEAR_OFFSETS:
                return least_squares(
                    fun=GeometricalTransformation.linear_offstes_loss_function,
                    x0=(self.d0, self.xc0, self.yc0),
                    args=(sensor_x, sensor_y, X, Y),
                ).x
            case self.OffsetsType.ALL_OFFSETS:
                return least_squares(
                    fun=GeometricalTransformation.all_offsets_loss_function,
                    x0=(self.d0, self.xc0, self.yc0, self.alpha0, self.beta0, self.phi0),
                    args=(sensor_x, sensor_y, X, Y),
                ).x
            case _:  # unreachable
                raise

    @final
    def perform_training(self, panel_name: str) -> pd.DataFrame:
        d_unbiased = self.__get_calibration_parameters(panel_name, offset_type=self.OffsetsType.NO_OFFSTES)
        d_linear, xc_linear, yc_linear = self.__get_calibration_parameters(
            panel_name, offset_type=self.OffsetsType.LINEAR_OFFSETS
        )
        d_all, xc_all, yc_all, alpha, beta, phi = self.__get_calibration_parameters(
            panel_name, offset_type=self.OffsetsType.ALL_OFFSETS
        )

        train_data = pd.concat([self.data_base[panel_name].initial_data, 
                                self.data_base[panel_name].second_run_data])

        train_data[["X_unbiased", "Y_unbiased"]] = train_data.apply(
            lambda row: pd.Series(
                [
                    GeometricalTransformation.transform_unbiased(row["x_light"], d_unbiased),
                    GeometricalTransformation.transform_unbiased(row["y_light"], d_unbiased),
                ]
            ),
            axis=1,
        )

        train_data[["X_linear_offsets", "Y_linear_offsets"]] = train_data.apply(
            lambda row: pd.Series(
                [
                    GeometricalTransformation.transform_with_linear_offset(row["x_light"], d_linear, xc_linear),
                    GeometricalTransformation.transform_with_linear_offset(row["y_light"], d_linear, yc_linear),
                ]
            ),
            axis=1,
        )

        train_data[["X_all_offsets", "Y_all_offsets"]] = train_data.apply(
            lambda row: pd.Series(
                [
                    GeometricalTransformation.transform_with_all_offsets_x(
                        row["x_light"],
                        d0=d_all,
                        center_offset_x=xc_all,
                        center_offset_y=yc_all,
                        alpha0=alpha,
                        beta0=beta,
                        phi0=phi,
                    ),
                    GeometricalTransformation.transform_with_all_offsets_y(
                        row["y_light"],
                        d0=d_all,
                        center_offset_x=xc_all,
                        center_offset_y=yc_all,
                        alpha0=alpha,
                        beta0=beta,
                        phi0=phi,
                    ),
                ]
            ),
            axis=1,
        )

        return train_data

    @final
    def perform_testing(self, panel_name: str) -> pd.DataFrame:
        (
            d_unbiased,
            (d_linear, xc_linear, yc_linear),
            (d_all, xc_all, yc_all, alpha, beta, phi),
        ) = self.get_calibration_parameters(panel_name=panel_name)

        result_data = pd.DataFrame(
            {
                "x_light": self.data_base[panel_name].test_data["x_light"],
                "y_light": self.data_base[panel_name].test_data["y_light"],
                "X": self.data_base[panel_name].test_data["X"],
                "Y": self.data_base[panel_name].test_data["Y"],
                "elevation": self.data_base[panel_name].test_data["elevation"],
                "azimuth": self.data_base[panel_name].test_data["azimuth"],
            }
        )
        result_data.index.name = "Measurement"

        result_data[["X_unbiased", "Y_unbiased"]] = result_data.apply(
            lambda row: pd.Series(
                [
                    GeometricalTransformation.transform_unbiased(row["x_light"], d_unbiased),
                    GeometricalTransformation.transform_unbiased(row["y_light"], d_unbiased),
                ]
            ),
            axis=1,
        )

        result_data[["X_linear_offsets", "Y_linear_offsets"]] = result_data.apply(
            lambda row: pd.Series(
                [
                    GeometricalTransformation.transform_with_linear_offset(row["x_light"], d_linear, xc_linear),
                    GeometricalTransformation.transform_with_linear_offset(row["y_light"], d_linear, yc_linear),
                ]
            ),
            axis=1,
        )

        result_data[["X_all_offsets", "Y_all_offsets"]] = result_data.apply(
            lambda row: pd.Series(
                [
                    GeometricalTransformation.transform_with_all_offsets_x(
                        row["x_light"],
                        d0=d_all,
                        center_offset_x=xc_all,
                        center_offset_y=yc_all,
                        alpha0=alpha,
                        beta0=beta,
                        phi0=phi,
                    ),
                    GeometricalTransformation.transform_with_all_offsets_y(
                        row["y_light"],
                        d0=d_all,
                        center_offset_x=xc_all,
                        center_offset_y=yc_all,
                        alpha0=alpha,
                        beta0=beta,
                        phi0=phi,
                    ),
                ]
            ),
            axis=1,
        )

        return result_data

    @final
    def get_calibration_parameters(self, panel_name: str) -> Any:
        return (
            self.__get_calibration_parameters(panel_name=panel_name, offset_type=self.OffsetsType.NO_OFFSTES),
            self.__get_calibration_parameters(panel_name=panel_name, offset_type=self.OffsetsType.LINEAR_OFFSETS),
            self.__get_calibration_parameters(panel_name=panel_name, offset_type=self.OffsetsType.ALL_OFFSETS),
        )

    @final
    def make_calibration_statistics(self) -> pd.DataFrame | None:
        panel_names = self.final_dataframe.index.get_level_values(0).drop_duplicates().tolist()
        self.errors_dataframe = self.final_dataframe.loc[
            :,
            [
                "X",
                "Y",
                "azimuth",
                "elevation",
                "X_unbiased",
                "Y_unbiased",
                "X_linear_offsets",
                "Y_linear_offsets",
                "X_all_offsets",
                "Y_all_offsets",
            ],
        ]
        self.errors_dataframe["angle_error_unbiased"] = self.errors_dataframe.apply(
            lambda row: pd.Series(
                Vector.angle_between_vectors(Vector(row["X"], row["Y"]), Vector(row["X_unbiased"], row["Y_unbiased"]))
            ),
            axis=1,
        )
        self.errors_dataframe["angle_error_linear_offsets"] = self.errors_dataframe.apply(
            lambda row: pd.Series(
                Vector.angle_between_vectors(
                    Vector(row["X"], row["Y"]),
                    Vector(row["X_linear_offsets"], row["Y_linear_offsets"]),
                )
            ),
            axis=1,
        )
        self.errors_dataframe["angle_error_all_offsets"] = self.errors_dataframe.apply(
            lambda row: pd.Series(
                Vector.angle_between_vectors(
                    Vector(row["X"], row["Y"]), Vector(row["X_all_offsets"], row["Y_all_offsets"])
                )
            ),
            axis=1,
        )
        mean_error_unbiased = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_unbiased"].mean() for panel in panel_names]
        )
        mean_error_linear_offsets = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_linear_offsets"].mean() for panel in panel_names]
        )
        mean_error_all_offsets = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_all_offsets"].mean() for panel in panel_names]
        )
        std_unbiased = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_unbiased"].std() for panel in panel_names]
        )
        std_linear_offsets = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_linear_offsets"].std() for panel in panel_names]
        )
        std_all_offsets = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_all_offsets"].std() for panel in panel_names]
        )

        self.errors_dataframe.drop(
            columns=[
                "X_unbiased",
                "Y_unbiased",
                "X_linear_offsets",
                "Y_linear_offsets",
                "X_all_offsets",
                "Y_all_offsets",
            ],
            inplace=True,
        )
        self.stats_dataframe = pd.DataFrame(
            data={
                "Panel": panel_names,
                "mean_error_unbiased": mean_error_unbiased,
                "standart_deviation_unbiased": std_unbiased,
                "mean_error_linear_offsets": mean_error_linear_offsets,
                "standart_deviation_linear_offsets": std_linear_offsets,
                "mean_error_all_offsets": mean_error_all_offsets,
                "standart_deviation_all_offsets": std_all_offsets,
            }
        )
        if self.path:
            full_path_err = os.path.join(self.path, self.calibration_errors_name)
            full_path_stats = os.path.join(self.path, self.calibration_stats_name)
            self.stats_dataframe.to_csv(full_path_stats, float_format="%f")
            self.errors_dataframe.to_csv(full_path_err, float_format="%f")
        else:
            return self.stats_dataframe

    @final
    def exec(self) -> pd.DataFrame | Tuple[pd.DataFrame] | None:
        results: List[pd.DataFrame] = []
        for panel in self.data_base.panel_list:
            try:
                res = self.perform_testing(panel_name=panel)
                res.index = self.get_multi_index(panel, res.index)
                results.append(res)
                print(res)
            except:
                print(f"[WARNING]: least squares error for panel {panel}")

        if len(results) < len(self.data_base.panel_list):
            print(f"[WARNING]: {len(self.data_base.panel_list) - len(results)} were lost")
        self.final_dataframe = pd.concat(results)
        if self.path:
            full_path = os.path.join(self.path, self.calibration_results_name)
            self.final_dataframe.to_csv(full_path, float_format="%f")
        else:
            return self.final_dataframe


if __name__ == "__main__":
    test = GeometricalFitter()
    test.perform_training("g02-0002")
