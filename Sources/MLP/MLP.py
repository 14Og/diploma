import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(path))

from Vector import Vector
from Database import PanelDataBase
from CalibrationPerformer import AbstractCalibrationPerformer

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from typing import final, Tuple


class MLPFitter(AbstractCalibrationPerformer):
    params = {
        "hidden_layer_sizes": (32, 8, 4),
        "solver": "lbfgs",
        "alpha": 0.0001,
        "activation": "tanh",
        "max_iter": 2000,
        "early_stopping": True,
        "random_state": 21,
        "learning_rate_init": 0.01,
        "learning_rate": "adaptive",
    }

    def __init__(self, module_path=path):

        super().__init__(module_path=module_path)
        self.model_X = make_pipeline(StandardScaler(), MLPRegressor(**self.params))
        self.model_Y = make_pipeline(StandardScaler(), MLPRegressor(**self.params))

    def __fit(self, train_features: pd.DataFrame, train_outputs: pd.DataFrame) -> None:

        self.model_X.fit(train_features, train_outputs["X"])
        self.model_Y.fit(train_features, train_outputs["Y"])

    @final
    def perform_training(self, panel_name: str) -> pd.DataFrame:
        train_dframe = pd.concat([self.data_base[panel_name].initial_data, 
                                self.data_base[panel_name].second_run_data])
        
        features = train_dframe.loc[:, ["x_light", "y_light"]]

        self.__fit(features, train_dframe)

        train_dframe["X_mlp"] = self.model_X.predict(features)
        train_dframe["Y_mlp"] = self.model_Y.predict(features)

        return train_dframe

    @final
    def perform_testing(self, panel_name: str) -> pd.DataFrame:
        train_dframe = pd.concat([self.data_base[panel_name].initial_data, 
                                  self.data_base[panel_name].second_run_data])
        train_features = train_dframe.loc[:, ["x_light", "y_light"]]

        test_dframe = self.data_base[panel_name].second_run_data.copy()
        test_features = test_dframe.loc[:, ["x_light", "y_light"]]

        self.__fit(train_features=train_features, train_outputs=train_dframe)

        test_dframe["X_mlp"] = self.model_X.predict(test_features)
        test_dframe["Y_mlp"] = self.model_Y.predict(test_features)

        return test_dframe

    @final
    def exec(self):
        results = []
        for panel in self.data_base.panel_list:
            res = self.perform_testing(panel_name=panel)
            res.index = self.get_multi_index(panel, res.index)
            results.append(res)
            print(res)

        if len(results) < len(self.data_base.panel_list):
            print(f"[WARNING]: {len(self.data_base.panel_list) - len(results)} were lost")

        self.final_dataframe = pd.concat(results)
        if self.path:
            full_path = os.path.join(self.path, self.calibration_results_name)
            self.final_dataframe.to_csv(full_path, float_format="%f")

        return self.final_dataframe

    @final
    def get_calibration_parameters(self, panel_name: str) -> Tuple[MLPRegressor]:
        self.__fit(panel_name)
        return self.model_X, self.model_Y

    @final
    def make_calibration_statistics(self):
        panel_names = self.final_dataframe.index.get_level_values(0).unique().to_list()
        self.errors_dataframe = self.final_dataframe.loc[
            :,
            ["X", "Y", "azimuth", "elevation", "X_mlp", "Y_mlp"],
        ]

        self.errors_dataframe["angle_error_mlp"] = self.errors_dataframe.apply(
            lambda row: pd.Series(
                Vector.angle_between_vectors(Vector(row["X"], row["Y"]), Vector(row["X_mlp"], row["Y_mlp"]))
            ),
            axis=1,
        )
        mean_error = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_mlp"].mean() for panel in panel_names]
        )
        standart_deviation = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_mlp"].std() for panel in panel_names]
        )
        self.errors_dataframe.drop(columns=["X_mlp", "Y_mlp"], inplace=True)
        self.stats_dataframe = pd.DataFrame(
            data={
                "Panel": panel_names,
                "mean_error_mlp": mean_error,
                "standart_deviation_mlp": standart_deviation,
            }
        )
        if self.path:
            full_path_err = os.path.join(self.path, self.calibration_errors_name)
            full_path_stats = os.path.join(self.path, self.calibration_stats_name)
            self.stats_dataframe.to_csv(full_path_stats, float_format="%f")
            self.errors_dataframe.to_csv(full_path_err, float_format="%f")
        else:
            return self.stats_dataframe
