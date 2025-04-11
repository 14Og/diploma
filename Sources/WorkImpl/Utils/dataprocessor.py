import pandas as pd
import numpy as np
import os
from pathlib import Path
from enum import Enum
from typing import Tuple, Dict, Iterable, Union, Sequence, Callable

from .gridcreator import GridCreator


class Stage(Enum):
    """
    Enum that represents calibration stage.
    Used for convenience
    """

    FIRST = (1,)
    SECOND = (2,)

    @classmethod
    def from_str(cls, name: str):
        return cls.__getitem__(name.upper())


class DataType(Enum):
    """
    Class that represents associations between data entities and their names
    """

    FIRST_RUN_DATA = (1,)
    FIRST_CALIBRATION_DATA = (2,)
    SECOND_RUN_DATA = (3,)
    SECOND_GRID_DATA = (4,)
    SECOND_CALIBRATION_DATA = (5,)
    FOV_DATA = (6,)
    ERROR_DATA = 7

    def __call__(self) -> str:
        return f"__{self.name.lower()}"

    def filename(self) -> str:
        return f"{self.name.lower()}.csv"

    def filename_no_extension(self) -> str:
        return self.name.lower()


class DataProcessor:
    """
    Class that operates with stand logs for following calibration.
    Behaviour depends on main.py arguments. Basically class loads and saves
    data used in calibration process.
    """

    def __init__(
        self, log_path: str, stage: Stage, panel: str, mode: str, reverse: bool = False
    ) -> None:
        self.projectpath: str = (
            Path(os.path.dirname(__file__)).parent.absolute().__str__()
        )
        self.mode = mode
        self.panelpath: str = self.__calibration_directory(panel=panel)
        self.frames_factory()

        if log_path:
            if stage == Stage.FIRST:
                self.first_run_data = self.add_adc(self.load_csv(log_path=log_path))
                self.__load(DataType.FOV_DATA)
            elif stage == Stage.SECOND:
                self.__load(
                    DataType.FIRST_RUN_DATA,
                    DataType.FIRST_CALIBRATION_DATA,
                    DataType.FOV_DATA,
                )
                frame = self.add_adc(self.load_csv(log_path=log_path))
                if reverse:
                    self.second_run_data = frame.iloc[::-1].reset_index(drop=True)
                else:
                    self.second_run_data = frame
        else:
            self.__load(
                DataType.FIRST_RUN_DATA,
                DataType.FIRST_CALIBRATION_DATA,
                DataType.SECOND_GRID_DATA,
                DataType.SECOND_RUN_DATA,
                DataType.SECOND_CALIBRATION_DATA,
                DataType.FOV_DATA,
                DataType.ERROR_DATA
            )

    @property
    def first_run_data(self) -> pd.DataFrame:
        if self.__get_frame(DataType.FIRST_RUN_DATA).empty:
            print("[WARNING] First run dataframe is empty")
        return self.__get_frame(DataType.FIRST_RUN_DATA)

    @first_run_data.setter
    def first_run_data(self, frame: pd.DataFrame) -> None:
        self.__set_frame(DataType.FIRST_RUN_DATA, frame)
        if self.mode != "plotter":
            self.save_csv(
                self.__get_frame(DataType.FIRST_RUN_DATA),
                DataType.FIRST_RUN_DATA.filename(),
            )

    @property
    def first_calibration_data(self) -> pd.DataFrame:
        if self.__get_frame(DataType.FIRST_CALIBRATION_DATA).empty:
            print("[WARNING] First calibration dataframe is empty")
        return self.__get_frame(DataType.FIRST_CALIBRATION_DATA)

    @first_calibration_data.setter
    def first_calibration_data(self, frame: pd.DataFrame) -> None:
        self.__set_frame(DataType.FIRST_CALIBRATION_DATA, frame)
        if self.mode != "plotter":
            self.save_csv(
                self.__get_frame(DataType.FIRST_CALIBRATION_DATA),
                DataType.FIRST_CALIBRATION_DATA.filename(),
            )

    @property
    def second_grid_data(self) -> pd.DataFrame:
        if self.__get_frame(DataType.SECOND_GRID_DATA).empty:
            print("[WARNING] Second grid dataframe is empty")
        return self.__get_frame(DataType.SECOND_GRID_DATA)

    @second_grid_data.setter
    def second_grid_data(self, frame: pd.DataFrame) -> None:
        self.__set_frame(DataType.SECOND_GRID_DATA, frame)
        if self.mode != "plotter":
            self.save_csv(
                self.__get_frame(DataType.SECOND_GRID_DATA),
                DataType.SECOND_GRID_DATA.filename(),
            )

    @property
    def second_run_data(self) -> pd.DataFrame:
        if self.__get_frame(DataType.SECOND_RUN_DATA).empty:
            print("[WARNING] Second run dataframe is empty")
        return self.__get_frame(DataType.SECOND_RUN_DATA)

    @second_run_data.setter
    def second_run_data(self, frame: pd.DataFrame) -> None:
        self.__set_frame(DataType.SECOND_RUN_DATA, frame)
        if self.mode != "plotter":
            self.save_csv(
                self.__get_frame(DataType.SECOND_RUN_DATA),
                DataType.SECOND_RUN_DATA.filename(),
            )

    @property
    def second_calibration_data(self) -> pd.DataFrame:
        if self.__get_frame(DataType.SECOND_CALIBRATION_DATA).empty:
            print("[WARNING] Second calibration dataframe is empty")
        return self.__get_frame(DataType.SECOND_CALIBRATION_DATA)

    @second_calibration_data.setter
    def second_calibration_data(self, frame: pd.DataFrame) -> None:
        self.__set_frame(DataType.SECOND_CALIBRATION_DATA, frame)
        if self.mode != "plotter":
            self.save_csv(
                self.__get_frame(DataType.SECOND_CALIBRATION_DATA),
                DataType.SECOND_CALIBRATION_DATA.filename(),
            )

    @property
    def fov_data(self) -> pd.DataFrame:
        if self.__get_frame(DataType.FOV_DATA).empty:
            print("[WARNING] FOV DATAFRAME IS EMPTY")
        return self.__get_frame(DataType.FOV_DATA)

    @fov_data.setter
    def fov_data(self, frame: pd.DataFrame) -> None:
        self.__set_frame(DataType.FOV_DATA, frame)
        if self.mode != "plotter":
            self.save_csv(
                self.__get_frame(DataType.FOV_DATA),
                DataType.FOV_DATA.filename(),
            )

    @property
    def error_data(self) -> pd.DataFrame:
        if self.__get_frame(DataType.ERROR_DATA).empty:
            print("[WARNING] Error dataframe is empty")
        return self.__get_frame(DataType.ERROR_DATA)

    @error_data.setter
    def error_data(self, frame: pd.DataFrame) -> None:
        self.__set_frame(DataType.ERROR_DATA, frame)
        if self.mode != "plotter":
            self.save_csv(
                self.__get_frame(DataType.ERROR_DATA),
                DataType.ERROR_DATA.filename(),
            )

    def __load(self, *names: Sequence[DataType]) -> None:
        """
        Loads data based on given names.

        Parameters
        ----------
        names : Sequence[DataTypes]
            Sequence of DataTypes entities
        """
        for name in names:
            frame = self.load_csv(self.panelpath, name.filename())
            setattr(self, name(), frame)

    def __calibration_directory(self, panel: str) -> None:
        """
        Setup calibration if it doesn't exist or save path to it.

        Parameters
        ----------
        panel : str
            Panel name
        """
        panelpath = os.path.join(self.projectpath, "SunSensorData", panel)
        if not os.path.exists(panelpath):
            try:
                os.mkdir(panelpath)
            except FileNotFoundError as e:
                print("Could not create directory.")
                return
        return panelpath

    def process_stand_data(
        self, raw_data: pd.DataFrame, angles_type="spherical"
    ) -> pd.DataFrame:
        """
        Processing given data from stand if it is represented in spherical angles.

        Parameters:
        ----------
        raw_data (pd.DataFrame): raw dataframe from stand YAML file processed by log processor.
        angles_type (str, optional): angles type. Defaults to spherical.

        Returns:
        ----------
        new_frame (pd.DataFrame): new dataframe with reformatted data (Azimuth, Elevation -> X, Y)

        """
        if angles_type != "spherical":
            return raw_data
        angles = raw_data.loc[:, "elevation":"azimuth"].apply(np.radians)
        X = np.multiply(
            np.sin(angles.loc[:, "elevation"]), np.cos(angles.loc[:, "azimuth"])
        )
        Y = np.multiply(
            np.sin(angles.loc[:, "elevation"]), np.sin(angles.loc[:, "azimuth"])
        )
        new_frame = pd.DataFrame(
            {
                "X": X,
                "Y": Y,
                "x_light": raw_data.loc[:, "x"],
                "y_light": raw_data.loc[:, "y"],
            }
        )
        return new_frame

    @staticmethod
    def load_csv(log_path: str, name: str = None) -> pd.DataFrame:
        """
        Load csv file with given path and filename. No exceptions.

        Parameters
        ----------
        log_path : str
            Path to log to read
        name : str, optional
            Filename, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe, loaded from csv file
        """
        if name is not None:
            log_path = os.path.join(log_path, name)
        try:
            return pd.read_csv(log_path, index_col=0)
        except Exception as e:
            print(f"[WARNING]: COULD NOT READ LOG: '{log_path}'. {e.args[-1]}.")
            return pd.DataFrame()

    @staticmethod
    def add_adc(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds adc percentage data.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with adc data in it (adc0-adc3).

        Returns
        -------
        pd.DataFrame
            Modified dataframe.
        """
        df["adcSum"] = df.loc[:, "adc0":"adc3"].sum(axis=1).round(3)
        for i in range(4):
            df[f"adc{i}_percent"] = round(df[f"adc{i}"] * 100 / df["adcSum"], 3)
        df["min_adc"] = df.loc[:, "adc0_percent":"adc3_percent"].min(axis=1).round(3)
        return df

    def save_csv(self, df: pd.DataFrame, name: str) -> None:
        """

        Saving csv file in calibration diirectory.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to save.
        name : str
            Filename.

        Raises
        ------
        Exception
            If directory doesn't exist.
        """
        if self.panelpath is None:
            raise Exception("No panel path.")
        df.to_csv(os.path.join(self.panelpath, name), index=True, na_rep="NaN")

    @staticmethod
    def csv_loader(func: Callable):
        """
        Decorator to transfer dataframe to function
        even if it's not represented as df but as
        path to it.

        Parameters
        ----------
        func : Callable
            Function that works with dataframe.
        """

        def wrapper(
            first_log: pd.DataFrame | str,
            second_log: pd.DataFrame | str = None,
            *args,
            **kwargs,
        ):
            modified_logs = []
            for log in (first_log, second_log):
                if isinstance(log, str):
                    try:
                        frame = pd.read_csv(log)
                        modified_logs.append(frame)
                    except Exception as e:
                        print(
                            f"Caught exception while reading csv in '{log}': {e.args[-1]}."
                        )
                        return
                elif log is not None:
                    modified_logs.append(log)
                    pass
            return func(*modified_logs, *args, **kwargs)

        return wrapper

    @staticmethod
    def adc_modifier(func: Callable):
        """
        Decorator to add adc percentage data
        to given dataframe and tarnsfer it to
        decorated function
        Parameters
        ----------
        func : Callable
            Function that works with dataframes
        """

        def wrapper(
            first_log: pd.DataFrame | str,
            second_log: pd.DataFrame | str = None,
            *args,
            **kwargs,
        ):
            if not isinstance(first_log, pd.DataFrame):
                raise Exception("First argument has to be a dataframe.")
            modified_frames = []
            for dframe in [first_log, second_log]:
                if isinstance(dframe, pd.DataFrame):
                    dframe["adcSum"] = dframe.loc[:, "adc0":"adc3"].sum(axis=1).round(3)
                    for i in range(4):
                        dframe[f"adc{i}_percent"] = (
                            dframe[f"adc{i}"] * 100 / dframe["adcSum"]
                        ).round(3)

                    dframe["min_adc"] = (
                        dframe.loc[:, "adc0_percent":"adc3_percent"]
                        .min(axis=1)
                        .round(3)
                    )
                    modified_frames.append(dframe)
            return func(*modified_frames, *args, **kwargs)

        return wrapper

    def frames_factory(self) -> None:
        """
        Sets dataframes fields from DataType entities
        for object.
        """
        for value in DataType:
            setattr(self, value(), None)

    def __get_frame(self, type: DataType) -> Union[pd.DataFrame, None]:
        return getattr(self, type())

    def __set_frame(self, type: DataType, frame: pd.DataFrame) -> None:
        setattr(self, type(), frame)

    @classmethod
    def load_to_report(
        cls, *data: Sequence[DataType], path: str
    ) -> Tuple[pd.DataFrame]:
        """
        Loads data from calibration path to use it in
        report notebook.

        Parameters
        ----------
        data: Sequence[DataType]
            Sequence of DataType entities.
        path : str
            Path where to read files.
        Returns
        -------
        Tuple[pd.DataFrame]
            Tuple with dataframes that were read.
        """
        dfs = []
        for name in data:
            dfs.append(cls.load_csv(path, name.filename()))
        return dfs


if __name__ == "__main__":
    raise Exception("Utility module.")
