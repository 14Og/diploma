from abc import ABC, abstractmethod

from Database import PanelDataBase

from typing import Any, Tuple

import numpy as np
import pandas as pd


class AbstractCalibrationPerformer(ABC):

    def __init__(self, module_path: str):
        self.calibration_stats_name  = "stats.csv"
        self.calibration_errors_name  = "error_stats.csv"
        self.calibration_results_name  = "results.csv"

        self.path = module_path
        self.data_base = PanelDataBase()
        self.final_dataframe: pd.DataFrame | None = None
        self.errors_dataframe: pd.DataFrame | None = None
        self.stats_dataframe: pd.DataFrame | None = None
        np.set_printoptions(precision=5, suppress=True)

    @abstractmethod
    def perform_training(self, panel_name: str) -> pd.DataFrame:
        """
        This function performs training process for one particular panel
        from panel database. It uses initial dataset and obliged to use only it

        Parameters
        ----------
        panel_name : str
            panel name, (g0x-00xx).

        Returns
        -------
        pd.DataFrame
        dataframe with trained values to probe calibration method.
        """
        ...

    @abstractmethod
    def perform_testing(self, panel_name: str) -> pd.DataFrame:
        """
        This function performs testing process for one particular panel from
        panel database. It uses test dataset.

        Parameters
        ----------
        panel_name : str
            panel name, (g0x-00xx).

        Returns
        -------
        pd.DataFrame
            dataframe with tested values.
        """
        ...

    @abstractmethod
    def get_calibration_parameters(self, panel_name: str) -> Any:
        """
        This function returns calibration coefficients/parameters for
        one particular panel from panel database.

        Parameters
        ----------
        panel_name : str
            panel name, (g0x-00xx).

        Returns
        -------
        Any
           Some sequence of calibration parameters that can be applied to any
           data from panel
        """
        ...

    @abstractmethod
    def exec() -> pd.DataFrame | Tuple[pd.DataFrame] | None:
        """
        Executes calibration process for all panels from database.
        Can return multiindexed dataframe or save by given path.

        Parameters
        ----------
        save_result : bool, optional
            by default False
        path_to_save : str | None, optional
            by default None

        Returns
        -------
        pd.DataFrame | Tuple[pd.DataFrame] | None
            Calibration result if self.path is None
        """
        ...

    @abstractmethod
    def make_calibration_statistics(self) -> pd.DataFrame | None:
        """
        Creates calibration result statistics data for each panel.
        Can return dataframe or save it by given path.

        Parameters
        ----------
        save_result : bool, optional
            by default False

        Returns
        -------
        pd.DataFrame | None
            Calibration statistics if self.path is None
        """
        ...

    def get_multi_index(self, panel_name: str, old_index) -> pd.MultiIndex:
        return pd.MultiIndex.from_product([[panel_name], old_index], names=["Panel", "Measurement"])