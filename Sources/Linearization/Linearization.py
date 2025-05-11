import sys, os

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(path))

# import Utils
from Vector import Vector
from Database import PanelDataBase
from CalibrationPerformer import AbstractCalibrationPerformer

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from typing import List, Tuple, final, Any


class LinearTransformationPerformer:
    """
    2x1 @ 2x1 + 2x1 = 2x1
    """

    @staticmethod
    def perform_linear_transformation_then_bias(
        input_vector: np.ndarray, linear_operator: np.ndarray, bias: np.ndarray
    ) -> np.ndarray:
        desired_shape = (2, 1)
        if input_vector.shape != desired_shape:
            input_vector = input_vector.reshape(desired_shape)
        if bias.shape != desired_shape:
            bias = bias.reshape(desired_shape)

        return (linear_operator @ input_vector) + bias

    @staticmethod
    def perform_bias_then_linear_transformation(
        nput_vector: np.ndarray, linear_operator: np.ndarray, bias: np.ndarray
    ) -> np.ndarray:
        desired_shape = (2, 1)
        if input_vector.shape != desired_shape:
            input_vector = input_vector.reshape(desired_shape)
        if bias.shape != desired_shape:
            bias = bias.reshape(desired_shape)

        return linear_operator @ (input_vector + bias)


class LinearFitter(AbstractCalibrationPerformer, LinearTransformationPerformer):
    def __init__(self, module_path=path):
        super().__init__(module_path=module_path)

        """
        Creates linear operator and bias matrix (A, b)
        for following data transformation.
        returns transformation parameters.
        """

    def __get_linearization_matrix(self, panel_name: str) -> Tuple[np.ndarray, np.ndarray]:

        train_data: pd.DataFrame = pd.concat([self.data_base[panel_name].initial_data, 
                                self.data_base[panel_name].second_run_data])

        sensor_x = train_data["x_light"].to_numpy()
        sensor_y = train_data["y_light"].to_numpy()

        vector_X = train_data["X"].to_numpy()
        vector_Y = train_data["Y"].to_numpy()
        vector_Z = np.sqrt(1 - vector_X**2 - vector_Y**2)

        dataset_shape = sensor_x.shape

        X_linearization_matrix = np.column_stack(
            [sensor_x, sensor_y, np.ones(dataset_shape), np.zeros((dataset_shape[0], 3))]
        )
        Y_linearization_matrx = np.column_stack(
            [np.zeros((dataset_shape[0], 3)), sensor_x, sensor_y, np.ones(dataset_shape)]
        )
        linearization_matrix = np.vstack([X_linearization_matrix, Y_linearization_matrx])
        vectors = np.hstack([vector_X, vector_Y])

        output, _, _, _ = np.linalg.lstsq(linearization_matrix, vectors, rcond=None)
        linear_operator = np.array([output[0], output[1], output[3], output[4]]).reshape(2, 2)
        bias = np.array([output[2], output[5]])  # bias_x, bias_y

        return linear_operator, bias

    @final
    def perform_training(self, panel_name: str) -> pd.DataFrame:
        linear_operator, bias = self.__get_linearization_matrix(panel_name=panel_name)
        train_data = pd.concat([self.data_base[panel_name].initial_data, 
                                self.data_base[panel_name].second_run_data])
        train_data[["X_linearized", "Y_linearized"]] = train_data.apply(
            lambda row: pd.Series(
                self.perform_linear_transformation_then_bias(
                    np.array([row["x_light"], row["y_light"]]),
                    linear_operator=linear_operator,
                    bias=bias,
                ).reshape(-1)
            ),
            axis=1,
        )

        return train_data

    @final
    def perform_testing(self, panel_name: str) -> pd.DataFrame:
        linear_operator, bias = self.__get_linearization_matrix(panel_name=panel_name)
        test_data: pd.DataFrame = self.data_base[panel_name].test_data.copy()

        result_data = pd.DataFrame(
            {
                "x_light": test_data["x_light"],
                "y_light": test_data["y_light"],
                "X": test_data["X"],
                "Y": test_data["Y"],
                "elevation": test_data["elevation"],
                "azimuth": test_data["azimuth"],
            }
        )
        result_data.index.name = "Measurement"

        result_data[["X_linearized", "Y_linearized"]] = result_data.apply(
            lambda row: pd.Series(
                self.perform_linear_transformation_then_bias(
                    np.array([row["x_light"], row["y_light"]]),
                    linear_operator=linear_operator,
                    bias=bias,
                ).reshape(-1)
            ),
            axis=1,
        )

        return result_data

    @final
    def exec(self) -> pd.DataFrame | None:
        results = []
        for panel in self.data_base.panel_list:
            result = self.perform_testing(panel)
            result.index = self.get_multi_index(panel, result.index)

            results.append(result)
            print(result)

        if len(results) < len(self.data_base.panel_list):
            print(f"[WARNING]: {len(self.data_base.panel_list) - len(results)} were lost")

        self.final_dataframe = pd.concat(results)

        if self.path:
            full_path = os.path.join(self.path, self.calibration_results_name)
            self.final_dataframe.to_csv(full_path, float_format="%f")
        else:
            return self.final_dataframe

    @final
    def get_calibration_parameters(self, panel_name: str) -> Any:
        return self.__get_linearization_matrix(panel_name=panel_name)

    @final
    def make_calibration_statistics(self) -> pd.DataFrame | None:
        panel_names = self.final_dataframe.index.get_level_values(0).drop_duplicates().tolist()
        self.errors_dataframe = self.final_dataframe.loc[
            :,
            ["X", "Y", "azimuth", "elevation", "X_linearized", "Y_linearized"],
        ]
        self.errors_dataframe["angle_error_linearized"] = self.errors_dataframe.apply(
            lambda row: pd.Series(
                Vector.angle_between_vectors(
                    Vector(row["X"], row["Y"]), Vector(row["X_linearized"], row["Y_linearized"])
                )
            ),
            axis=1,
        )
        mean_error = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_linearized"].mean() for panel in panel_names]
        )
        standart_deviation = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_linearized"].std() for panel in panel_names]
        )
        self.errors_dataframe.drop(columns=["X_linearized", "Y_linearized"], inplace=True)
        self.stats_dataframe = pd.DataFrame(
            data={
                "Panel": panel_names,
                "mean_error_linearized": mean_error,
                "standart_deviation_linearized": standart_deviation,
            }
        )
        if self.path:
            full_path_err = os.path.join(self.path, self.calibration_errors_name)
            full_path_stats = os.path.join(self.path, self.calibration_stats_name)
            self.stats_dataframe.to_csv(full_path_stats, float_format="%f")
            self.errors_dataframe.to_csv(full_path_err, float_format="%f")
        else:
            return self.stats_dataframe


if __name__ == "__main__":
    tester = LinearFitter()
    data = tester.exec(True, "/home/kolya/Yandex.Disk/diploma/Sources/Linearization/stats.csv")
