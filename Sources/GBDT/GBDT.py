import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(path))

from Vector import Vector
from Database import PanelDataBase
from CalibrationPerformer import AbstractCalibrationPerformer

import pandas as pd
import numpy as np
import xgboost as xgb

from typing import final, Tuple


class GBDTFitter(AbstractCalibrationPerformer):
    params = {
        "objective": "reg:squarederror",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 1,
        "colsample_bytree": 1.0,
        "reg_lambda": 4,
        "reg_alpha": 0,

    }

    def __init__(self, module_path=path):
        super().__init__(module_path=module_path)
        self.x_regressor = xgb.XGBRegressor(**self.params)
        self.y_regressor = xgb.XGBRegressor(**self.params)

    def __fit(self, panel_name: str) -> None:
        features = self.data_base[panel_name].initial_data.loc[:, ["x_light", "y_light"]]
        self.x_regressor.fit(features, self.data_base[panel_name].initial_data["X"])
        self.y_regressor.fit(features, self.data_base[panel_name].initial_data["Y"])

    @final
    def perform_training(self, panel_name: str) -> pd.DataFrame:
        train_data = self.data_base[panel_name].initial_data.copy()
        features = train_data.loc[:, ["x_light", "y_light"]]

        self.__fit(panel_name=panel_name)

        train_data["X_boosted"] = self.x_regressor.predict(features)
        train_data["Y_boosted"] = self.y_regressor.predict(features)

        return train_data

    @final
    def perform_testing(self, panel_name: str) -> pd.DataFrame:
        test_data = self.data_base[panel_name].test_data.copy()
        samples = test_data.loc[:, ["x_light", "y_light"]]
        self.__fit(panel_name=panel_name)

        test_data["X_boosted"] = self.x_regressor.predict(samples)
        test_data["Y_boosted"] = self.y_regressor.predict(samples)
        
        return test_data

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
    def get_calibration_parameters(self, panel_name: str) -> Tuple[xgb.XGBRegressor]:
        self.__fit(panel_name)
        return self.x_regressor, self.y_regressor

    @final
    def make_calibration_statistics(self):
        panel_names = self.final_dataframe.index.get_level_values(0).unique().to_list()
        self.errors_dataframe = self.final_dataframe.loc[
            :,
            ["X", "Y", "azimuth", "elevation", "X_boosted", "Y_boosted"],
        ]
        
        self.errors_dataframe["angle_error_boosted"] = self.errors_dataframe.apply(
            lambda row: pd.Series(
                Vector.angle_between_vectors(
                    Vector(row["X"], row["Y"]), Vector(row["X_boosted"], row["Y_boosted"])
                )
            ),
            axis=1,
        )
        mean_error = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_boosted"].mean() for panel in panel_names]
        )
        standart_deviation = np.array(
            [self.errors_dataframe.dropna().loc[panel]["angle_error_boosted"].std() for panel in panel_names]
        )
        self.errors_dataframe.drop(columns=["X_boosted", "Y_boosted"], inplace=True)
        self.stats_dataframe = pd.DataFrame(
            data={
                "Panel": panel_names,
                "mean_error_boosted": mean_error,
                "standart_deviation_boosted": standart_deviation,
            }
        )
        if self.path:
            full_path_err = os.path.join(self.path, self.calibration_errors_name)
            full_path_stats = os.path.join(self.path, self.calibration_stats_name)
            self.stats_dataframe.to_csv(full_path_stats, float_format="%f")
            self.errors_dataframe.to_csv(full_path_err, float_format="%f")
        else:
            return self.stats_dataframe