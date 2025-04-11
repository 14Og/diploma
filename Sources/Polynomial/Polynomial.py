import sys, os

path = (os.path.dirname(os.path.realpath(__file__))) 
sys.path.append(os.path.dirname(path))

from CalibrationPerformer import AbstractCalibrationPerformer
from Database import PanelDataBase, PanelData
from Vector import Vector

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

from typing import Tuple, final, Any


class PolynomialFitter(AbstractCalibrationPerformer):
    def __init__(self, module_path = path):
        super().__init__(module_path=module_path)


    def __get_polynom_coefficients(self, panel_name: str, degree: int) -> Tuple[Tuple, Tuple]:
        """
        Returns polynomial coefs for X and Y, respectively

        Parameters
        ----------
        panel_name : str

        degree : int
            degree of fitting polynom

        Returns
        -------
        Tuple[Tuple, Tuple]
            X coefs, Y coefs
        """
        model_X, model_Y, _ = self.__get_testing_model(panel_name=panel_name, degree=degree)

        return (model_X.coef_, model_Y.coef_)

    def __get_training_model(self, panel_name: str, degree: int) -> Tuple[LinearRegression]:
        poly = PolynomialFeatures(degree, include_bias=False)
        init_data = self.data_base[panel_name].initial_data

        X = init_data["X"]
        Y = init_data["Y"]
        x_light = init_data["x_light"]
        y_light = init_data["y_light"]
        input = np.column_stack((x_light, y_light))
        combination = poly.fit_transform(input)

        model_X = LinearRegression().fit(combination, X)
        model_Y = LinearRegression().fit(combination, Y)
        return model_X, model_Y, combination

    def __get_testing_model(self, panel_name: str, degree: int) -> Tuple[LinearRegression]:
        poly = PolynomialFeatures(degree, include_bias=False)
        init_data = self.data_base[panel_name].test_data

        X = init_data["X"]
        Y = init_data["Y"]
        x_light = init_data["x_light"]
        y_light = init_data["y_light"]
        input = np.column_stack((x_light, y_light))
        combination = poly.fit_transform(input)

        model_X = LinearRegression().fit(combination, X)
        model_Y = LinearRegression().fit(combination, Y)
        return model_X, model_Y, combination

    @final
    def perform_training(self, panel_name: str) -> pd.DataFrame:
        sq_x, sq_y, sq_comb = self.__get_training_model(panel_name, 2)
        cube_x, cube_y, cube_comb = self.__get_training_model(panel_name, 3)
        quad_x, quad_y, quad_comb = self.__get_training_model(panel_name, 4)

        training_data = self.data_base[panel_name].initial_data
        X = training_data["X"]
        Y = training_data["Y"]
        x_light = training_data["x_light"]
        y_light = training_data["y_light"]

        training_data[["X_sq", "Y_sq"]] = np.array([sq_x.predict(sq_comb), sq_y.predict(sq_comb)]).T
        training_data[["X_cube", "Y_cube"]] = np.array([cube_x.predict(cube_comb), cube_y.predict(cube_comb)]).T
        training_data[["X_quad", "Y_quad"]] = np.array([quad_x.predict(quad_comb), quad_y.predict(quad_comb)]).T

        return training_data

    @final
    def perform_testing(self, panel_name: str) -> pd.DataFrame:

        result_data = pd.DataFrame(
            data={
                "x_light": self.data_base[panel_name].test_data["x_light"],
                "y_light": self.data_base[panel_name].test_data["y_light"],
                "X": self.data_base[panel_name].test_data["X"],
                "Y": self.data_base[panel_name].test_data["Y"],
                "elevation": self.data_base[panel_name].test_data["elevation"],
                "azimuth": self.data_base[panel_name].test_data["azimuth"],
            }
        )

        sq_x, sq_y, sq_comb = self.__get_testing_model(panel_name, 2)
        cube_x, cube_y, cube_comb = self.__get_testing_model(panel_name, 3)
        quad_x, quad_y, quad_comb = self.__get_testing_model(panel_name, 4)

        result_data[["X_sq", "Y_sq"]] = np.array([sq_x.predict(sq_comb), sq_y.predict(sq_comb)]).T
        result_data[["X_cube", "Y_cube"]] = np.array([cube_x.predict(cube_comb), cube_y.predict(cube_comb)]).T
        result_data[["X_quad", "Y_quad"]] = np.array([quad_x.predict(quad_comb), quad_y.predict(quad_comb)]).T

        return result_data

    @final
    def exec(self) -> pd.DataFrame | Tuple[pd.DataFrame] | None:
        results = []
        for panel in self.data_base.panel_list:
            res = self.perform_testing(panel)
            res.index = self.get_multi_index(panel, res.index)
            results.append(res)
            print(res)

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
        return (self.__get_polynom_coefficients(panel_name, i) for i in (2, 3, 4))

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
                "X_sq",
                "Y_sq",
                "X_cube",
                "Y_cube",
                "X_quad",
                "Y_quad",
            ],
        ]
        print(self.errors_dataframe)
        self.errors_dataframe["angle_error_sq"] = self.errors_dataframe.apply(
            lambda row: pd.Series(
                Vector.angle_between_vectors(Vector(row["X"], row["Y"]), Vector(row["X_sq"], row["Y_sq"]))
            ),
            axis=1,
        )
        self.errors_dataframe["angle_error_cube"] = self.errors_dataframe.apply(
            lambda row: pd.Series(
                Vector.angle_between_vectors(Vector(row["X"], row["Y"]), Vector(row["X_cube"], row["Y_cube"]))
            ),
            axis=1,
        )
        self.errors_dataframe["angle_error_quad"] = self.errors_dataframe.apply(
            lambda row: pd.Series(
                Vector.angle_between_vectors(Vector(row["X"], row["Y"]), Vector(row["X_quad"], row["Y_quad"]))
            ),
            axis=1,
        )

        self.errors_dataframe.drop(
            columns=[
                "X_sq",
                "Y_sq",
                "X_cube",
                "Y_cube",
                "X_quad",
                "Y_quad",
            ],
            inplace=True,
        )

        mean_sq = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_sq"].mean() for panel in panel_names]
        )
        mean_cube = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_cube"].mean() for panel in panel_names]
        )
        mean_quad = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_quad"].mean() for panel in panel_names]
        )

        std_sq = np.array([self.errors_dataframe.dropna().loc[panel]["angle_error_sq"].std() for panel in panel_names])
        std_cube = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_cube"].std() for panel in panel_names]
        )
        std_quad = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_quad"].std() for panel in panel_names]
        )

        self.stats_dataframe = pd.DataFrame(
            data={
                "Panel": panel_names,
                "mean_error_sq": mean_sq,
                "mean_error_cube": mean_cube,
                "mean_error_quad": mean_quad,
                "standart_deviation_sq": std_sq,
                "standart_deviation_cube": std_cube,
                "standart_deviation_quad": std_quad,
            }
        )

        if self.path:
            full_path_err = os.path.join(self.path, self.calibration_errors_name)
            full_path_stats = os.path.join(self.path, self.calibration_stats_name)
            self.stats_dataframe.to_csv(full_path_stats, float_format="%f")
            self.errors_dataframe.to_csv(full_path_err, float_format="%f")
        else:
            return self.stats_dataframe
