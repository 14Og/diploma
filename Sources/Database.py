import sys
import os


import pandas as pd
import numpy as np

import os
import glob
from enum import Enum
from typing import Tuple, List


class DataInstances(Enum):
    INIT_DATA = "first_run_data.csv"
    TEST_DATA = "error_data.csv"
    FOV_DATA = "fov_data.csv"
    SECOND_RUN_DATA = "second_run_data.csv"


class PanelData:

    def __init__(
        self,
        name: str,
        init_data: pd.DataFrame | None = None,
        test_data: pd.DataFrame | None = None,
        trained_data: pd.DataFrame | None = None,
        fov_data: pd.DataFrame | None = None,
        second_run_data: pd.DataFrame | None = None
    ):
        self.__initial_data = init_data
        self.__test_data = test_data
        self.__trained_data = trained_data
        self.__fov_data = fov_data
        self.__second_run_data = second_run_data

        self.name = name

    @property
    def trained_data(self) -> pd.DataFrame:
        return self.__trained_data

    @trained_data.setter
    def trained_data(self, trained_data: pd.DataFrame) -> None:
        self.__trained_data = trained_data

    @property
    def test_data(self) -> pd.DataFrame:
        return self.__test_data

    @test_data.setter
    def test_data(self, test_data: pd.DataFrame) -> None:
        self.__test_data = test_data

    @property
    def initial_data(self) -> pd.DataFrame:
        return self.__initial_data

    @initial_data.setter
    def initial_data(self, initial_data: pd.DataFrame) -> None:
        self.__initial_data = initial_data

    @property
    def fov_data(self) -> pd.DataFrame:
        return self.__fov_data

    @fov_data.setter
    def fov_data(self, fov_data: pd.DataFrame) -> None:
        self.__fov_data = fov_data
        
    @property
    def second_run_data(self) -> pd.DataFrame:
        return self.__second_run_data

    @second_run_data.setter
    def second_run_data(self, second_run_data: pd.DataFrame) -> None:
        self.__second_run_data = second_run_data


class PanelDataBase:

    DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Data")
    PANEL_PATH_PATTERN = "g**-****"
    FOV_THRESHOLD = 60

    def __init__(self):
        print("Database target directory: ", self.DATA_DIRECTORY)
        self.panel_list, self.bad_panels_list = self.find_panels_lists()

    def __getitem__(self, key: str) -> PanelData:
        panel = PanelData(key)
        panel.initial_data, panel.test_data, panel.fov_data, panel.second_run_data = self.__read_dataframes(key)
        return panel

    def __read_dataframes(self, panel: str) -> Tuple[pd.DataFrame]:
        def read(path, cut=True):
            frame = pd.read_csv(path, index_col=None)
            return frame[frame["elevation"] <= PanelDataBase.FOV_THRESHOLD] if cut else frame

        return (
            read(self.__join(panel, DataInstances.INIT_DATA.value)),
            read(self.__join(panel, DataInstances.TEST_DATA.value)),
            read(self.__join(panel, DataInstances.FOV_DATA.value), False),
            read(self.__join(panel, DataInstances.SECOND_RUN_DATA.value))
        )

    @staticmethod
    def __join(*args) -> str:
        return os.path.join(PanelDataBase.DATA_DIRECTORY, *args)

    @staticmethod
    def find_panels_lists() -> Tuple[List[str]]:
        good_panels = []
        bad_panels = []
        paths = glob.glob(PanelDataBase.PANEL_PATH_PATTERN, recursive=False, root_dir=PanelDataBase.DATA_DIRECTORY)
        for path in paths:
            if PanelDataBase.is_panel_good(PanelDataBase.__join(path, DataInstances.FOV_DATA.value)):
                good_panels.append(path)
            else:
                bad_panels.append(path)

        return sorted(good_panels), sorted(bad_panels)

    @staticmethod
    def is_panel_good(absolute_pathname: str, fov=FOV_THRESHOLD) -> bool:
        data = pd.read_csv(absolute_pathname)
        elevation = PanelDataBase.__estimate_fov(data)
        return elevation >= fov

    def __estimate_fov(fov_dataframe: pd.DataFrame) -> float:
        azimuths = fov_dataframe["azimuth"].drop_duplicates().tolist()
        samples = [fov_dataframe[fov_dataframe["azimuth"] == az] for az in azimuths]
        fov_scopes = []
        for sample in samples:
            elevations = sample["elevation"].drop_duplicates().tolist()
            divided = [sample[sample["elevation"] == i] for i in elevations]
            r = np.array([np.linalg.norm([s["x_light"].mean(), s["y_light"].mean()]) for s in divided]).round(4)
            R = np.array([np.linalg.norm([s["X"].mean(), s["Y"].mean()]) for s in divided]).round(4)
            r_dot = np.gradient(r, R)
            negative_derivatives = np.where(r_dot < 0)[0]
            fov = elevations[negative_derivatives[0]] if negative_derivatives.size > 0 else elevations[-1]
            fov_scopes.append(fov)
        return min(fov_scopes)
