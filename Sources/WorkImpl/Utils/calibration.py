import pandas as pd
import numpy as np
from typing import Callable, Union
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


from .dataprocessor import DataProcessor, Stage, DataType
from .plotter import Plotter
from .gridcreator import GridCreator
from .tsp import Voyager
from SolarPanelReport.report import create_report


pd.set_option("display.max.columns", None)
pd.set_option("display.max.rows", None)
np.set_printoptions(threshold=10000)
np.set_printoptions(suppress=True)


class Calibration:
    """
    Main class that implements interpolation for first and second stages.
    Uses bilinear or bicubic interpolation for first stage (bicubic is  considerably more preferred)
    Uses bilinear or bicubic interpolation +
    bicubic extrapolation based on multiple polynomial regression for second stage.
    """

    def __init__(self, kwargs) -> None:
        self.__arguments_factory(kwargs=kwargs)
        if self.stage:
            self.stage = Stage.from_str(self.stage)
        self.data_processor = DataProcessor(
            stage=self.stage,
            panel=self.panel,
            log_path=self.path,
            mode=self.mode,
            reverse=self.reverse,
        )
        self.squared_regular_sensor_grid: Union[pd.DataFrame, np.ndarray, None] = None
        self.X_interpolator: Union[Callable, None] = None
        self.Y_interpolator: Union[Callable, None] = None
        Plotter.set_plotter_mode(self.mode)

    @staticmethod
    def extrapolator(func: Callable) -> Callable:
        """
        Decorator to find data to extrapolate
        for following extrapolation function.

        Parameters
        ----------
        func : Callable
            Extrapolation function

        Returns
        -------
        Callable
            Modified function
        """

        def wrapper(dataframe: pd.DataFrame, *args) -> pd.DataFrame:
            c = dataframe.copy(deep=True)
            c.loc[:, "Extrapolated"] = c.loc[:, "X"].isna().to_numpy()
            return func(c, *args)

        return wrapper

    def emit(self):
        """
        Main event-managing method.
        All the events happen consistenly and run at main app thread.
        """
        if self.mode != "plotter":
            if self.stage == Stage.FIRST:
                fov_boundary = Calibration.estimate_fov(self.data_processor.fov_data)
                fov_cutted_data = self.data_processor.first_run_data[
                    self.data_processor.first_run_data["elevation"] <= fov_boundary
                ]
                self.squared_regular_sensor_grid = (
                    GridCreator.create_regular_sensor_squared_grid(
                        fov_cutted_data, binary_degree=8
                    )
                )
                self.__first_calibration(
                    fov_cutted_data,
                    self.squared_regular_sensor_grid,
                    self.interpolation,
                )
                self.data_processor.second_grid_data = self.__pathfinder()
                GridCreator.save_grid(
                    grid=self.data_processor.second_grid_data,
                    path=self.data_processor.panelpath,
                    name=DataType.SECOND_GRID_DATA.filename_no_extension(),
                    mode="angles",
                )

            elif self.stage == Stage.SECOND:
                self.squared_regular_sensor_grid = (
                    self.data_processor.first_calibration_data.loc[
                        :, ["x_light", "y_light"]
                    ].to_numpy()
                )
                self.__second_calibration(
                    first_run_data=self.data_processor.first_run_data[
                        self.data_processor.first_run_data["elevation"]
                        < Calibration.estimate_fov(self.data_processor.fov_data)
                    ],
                    second_run_data=self.data_processor.second_run_data,
                    interpolation_grid_data=self.squared_regular_sensor_grid,
                    mode=self.interpolation,
                    grid_type="squared",
                )
                if self.mode == "report":
                    create_report(
                        panelpath=self.data_processor.panelpath, reportname=self.panel
                    )
                elif self.mode == "continue":
                    GridCreator.save_grid(
                        self.data_processor.second_calibration_data,
                        path=self.data_processor.panelpath,
                        name=DataType.SECOND_CALIBRATION_DATA.filename_no_extension(),
                        mode="XY",
                        left_bounds=True,
                    )

        else:
            # use Plotter if you want (with -mode=plotter)
            pass

    def __first_calibration(
        self,
        input_data: pd.DataFrame,
        regular_sensor_grid: np.ndarray,
        mode: str = "linear",
    ) -> None:
        """
        Performs first calibration stage.
        Creates interpoated X and Y data for Sun vector
        Inside of regular sensor grid.

        Parameters:
        ----------
        input_data : pd.DataFrame
            Dataframe with first log data.
        regular_sensor_grid : np.ndarray
            Sensor grid to find sun vector coordinates for.
        """
        sensor_x = input_data.loc[:, "x_light"].to_numpy()
        sensor_y = input_data.loc[:, "y_light"].to_numpy()
        stand_x = input_data.loc[:, "X"].to_numpy()
        stand_y = input_data.loc[:, "Y"].to_numpy()

        if mode == "linear":
            self.X_interpolator = LinearNDInterpolator(
                list(zip(sensor_x, sensor_y)), stand_x
            )
            self.Y_interpolator = LinearNDInterpolator(
                list(zip(sensor_x, sensor_y)), stand_y
            )
        else:
            self.X_interpolator = CloughTocher2DInterpolator(
                list(zip(sensor_x, sensor_y)), stand_x
            )
            self.Y_interpolator = CloughTocher2DInterpolator(
                list(zip(sensor_x, sensor_y)), stand_y
            )

        interpolated = pd.DataFrame(
            {
                "X": self.X_interpolator(
                    regular_sensor_grid[:, 0],
                    regular_sensor_grid[:, 1],
                ).round(5),
                "Y": self.Y_interpolator(
                    regular_sensor_grid[:, 0], regular_sensor_grid[:, 1]
                ).round(5),
                "x_light": regular_sensor_grid[:, 0].round(5),
                "y_light": regular_sensor_grid[:, 1].round(5),
            },
        )
        if interpolated.isnull().sum().sum():
            print("WARNING: INTEPOLATED DATA HAS NaNs")
        interpolated.index.name = "id"
        self.data_processor.first_calibration_data = interpolated

    def __second_calibration(
        self,
        first_run_data: pd.DataFrame,
        second_run_data: pd.DataFrame,
        interpolation_grid_data: pd.DataFrame,
        mode: str = "linear",
        grid_type: str = "circled",
    ) -> None:
        """
        Performs second calibration stage with following extrapolation.
        After this stage you will have enough data to save it in .bin file.

        Parameters
        ----------
        first_run_data : pd.DataFrame
            Data from first stand run. It have to be valid i. e. all
            points have to be inside sensor FOV. Cut data if it's dirty.
        second_run_data : pd.DataFrame
            Data from second stand run.
        interpolation_grid_data : pd.DataFrame
            Regular sensor grid.
        mode : str, optional
            Interpolation type, by default "linear"
        grid_type : str, optional
            Sensor grid type, by default "circled"
        """
        merged_data = pd.concat([first_run_data, second_run_data], ignore_index=True)
        stand_X = merged_data.loc[:, "X"].to_numpy()
        stand_Y = merged_data.loc[:, "Y"].to_numpy()
        sensor_x = merged_data.loc[:, "x_light"].to_numpy()
        sensor_y = merged_data.loc[:, "y_light"].to_numpy()

        if mode == "linear":
            self.X_interpolator = LinearNDInterpolator(
                points=list(zip(sensor_x, sensor_y)), values=stand_X
            )
            self.Y_interpolator = LinearNDInterpolator(
                points=list(zip(sensor_x, sensor_y)), values=stand_Y
            )
        else:
            self.X_interpolator = CloughTocher2DInterpolator(
                points=list(zip(sensor_x, sensor_y)), values=stand_X
            )
            self.Y_interpolator = CloughTocher2DInterpolator(
                points=list(zip(sensor_x, sensor_y)), values=stand_Y
            )

        interpolated = pd.DataFrame(
            {
                "X": self.X_interpolator(
                    interpolation_grid_data[:, 0],
                    interpolation_grid_data[:, 1],
                ).round(5),
                "Y": self.Y_interpolator(
                    interpolation_grid_data[:, 0],
                    interpolation_grid_data[:, 1],
                ).round(5),
                "x_light": interpolation_grid_data[:, 0].round(5),
                "y_light": interpolation_grid_data[:, 1].round(5),
            },
        )

        if interpolated.isnull().sum().sum():
            print("[INFO]: INTERPOLATED DATA HAS NANS. THERE WILL BE EXTRAPOLATION.")
        interpolated.index.name = "id"
        if grid_type == "circled":
            self.data_processor.second_calibration_data = interpolated
        else:
            self.data_processor.second_calibration_data = (
                self.__least_squares_extrapolation(
                    interpolated, self.X_interpolator, self.Y_interpolator
                )
            )

    @staticmethod
    @extrapolator
    def __average_extrapolation(interpolated_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Average extrapolation. Shows pretty bad results, so it's deprecated.

        Parameters
        ----------
        interpolated_frame : pd.DataFrame
            Dataframe to extrapolate

        Returns
        -------
        pd.DataFrame
            Extrapolated frame
        """
        for i, row in interpolated_frame.iterrows():
            if row["Extrapolated"]:
                same_x = interpolated_frame[
                    interpolated_frame["x_light"] == row["x_light"]
                ]["X"]
                same_y = interpolated_frame[
                    interpolated_frame["y_light"] == row["y_light"]
                ]["Y"]
                if same_x.isna().sum() != same_x.shape[0]:
                    interpolated_frame.at[i, "X"] = same_x[same_x.notna()].mean()
                else:
                    interpolated_frame.at[i, "X"] = (
                        interpolated_frame["X"].min()
                        if row["x_light"] > 0
                        else interpolated_frame["X"].max()
                    )
                if same_y.isna().sum() != same_y.shape[0]:
                    interpolated_frame.at[i, "Y"] = same_y[same_y.notna()].mean()
                else:
                    interpolated_frame.at[i, "Y"] = (
                        interpolated_frame["Y"].min()
                        if row["y_light"] > 0
                        else interpolated_frame["Y"].max()
                    )
        return interpolated_frame

    @staticmethod
    @extrapolator
    def __nn_extrapolation(interpolated_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Neighrest neighbour extrapolation method. Deprecated.

        Parameters
        ----------
        interpolated_frame : pd.DataFrame
            Dataframe to extrapolate.

        Returns
        -------
        pd.DataFrame
            Extrapolated frame.
        """
        existing_values = interpolated_frame.dropna().reset_index(drop=True)

        nans_idx = interpolated_frame["X"].isna()
        nans_idx = nans_idx[nans_idx == True].index.tolist()

        norm_matrix = (
            np.subtract.outer(
                existing_values.loc[:, "x_light"].to_numpy(),
                interpolated_frame.loc[nans_idx, "x_light"].to_numpy(),
            )
            ** 2
            + np.subtract.outer(
                existing_values.loc[:, "y_light"].to_numpy(),
                interpolated_frame.loc[nans_idx, "y_light"].to_numpy(),
            )
            ** 2
        ).T

        nn_idx = np.argmin(norm_matrix, axis=1)
        for idx, X, Y in zip(
            nans_idx, existing_values.loc[nn_idx, "X"], existing_values.loc[nn_idx, "Y"]
        ):
            interpolated_frame.loc[idx, "X"] = X
            interpolated_frame.loc[idx, "Y"] = Y

        return interpolated_frame

    @staticmethod
    @extrapolator
    def __least_squares_extrapolation(
        interpolated_frame: pd.DataFrame,
        X_interpolator: Callable,
        Y_interpolator: Callable,
    ) -> pd.DataFrame:
        """
        Least squares extrapolation. Shows the best result.

        Parameters
        ----------
        interpolated_frame : pd.DataFrame
            Dataframe to extrapolate.
        X_interpolator : Callable
            Second calibration interpolator for X coordinate.
        Y_interpolator : Callable
            Second calibration interpolator for Y coordinate.

        Returns
        -------
        pd.DataFrame
            Extrapolated frame.
        """
        x_origin = interpolated_frame.dropna().loc[:, "x_light"].mean()
        y_origin = interpolated_frame.dropna().loc[:, "y_light"].mean()

        nans_idx = interpolated_frame["X"].isna()
        nans_idx = nans_idx[nans_idx == True].index.tolist()

        points_to_extrapolate = interpolated_frame.loc[nans_idx, ["x_light", "y_light"]]
        points_to_extrapolate.loc[:, "slope"] = (
            points_to_extrapolate.loc[:, "y_light"] - y_origin
        ) / (points_to_extrapolate.loc[:, "x_light"] - x_origin)
        
        points_to_extrapolate.loc[:, "intercept"] = (
            y_origin - x_origin * points_to_extrapolate.loc[:, "slope"]
        )


        for i, row in points_to_extrapolate.iterrows():
            sample_space = np.linspace(
                x_origin,
                row["x_light"],
                1000,
            )
            y_data = row["intercept"] + row["slope"] * sample_space
            X_data = X_interpolator(sample_space, y_data)
            Y_data = Y_interpolator(sample_space, y_data)
            nans = np.argwhere(np.isnan(X_data))
            x_data = np.delete(sample_space, nans)
            y_data = np.delete(y_data, nans)
            X_data = np.delete(X_data, nans)
            Y_data = np.delete(Y_data, nans)

            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_features = poly.fit_transform(np.array([x_data, y_data]).T, X_data)
            Y_features = poly.fit_transform(np.array([x_data, y_data]).T, Y_data)
            regressor = LinearRegression()
            data_transform = poly.fit_transform(
                np.array([row["x_light"], row["y_light"]]).reshape(1, -1), X_data
            )

            regressor.fit(X_features, X_data)
            interpolated_frame.loc[i, "X"] = regressor.predict(data_transform)

            regressor.fit(Y_features, Y_data)
            interpolated_frame.loc[i, "Y"] = regressor.predict(data_transform)

            cutter = lambda point: (
                point if abs(point) <= 0.95 else 0.95 if point > 0.95 else -0.95
            )

        interpolated_frame["X"] = interpolated_frame["X"].apply(cutter)
        interpolated_frame["Y"] = interpolated_frame["Y"].apply(cutter)

        return interpolated_frame

    def __pathfinder(self) -> pd.DataFrame:
        """
        Uses tsp solver algorithms to calculate
        Fastest path for second stand run.

        Returns
        -------
        pd.DataFrame
            Sorted dataframe.
        """
        updated_calibration_frame = GridCreator.create_second_run_grid(
            interpolated_frame=self.data_processor.first_calibration_data
        )
        self.data_processor.first_calibration_data = updated_calibration_frame
        print(f"Total squared grid points: {self.squared_regular_sensor_grid.shape}")
        voyager = Voyager(mode="release")
        points_arr = (
            self.data_processor.first_calibration_data.loc[:, ["X", "Y"]]
            .dropna()
            .to_numpy()
        )
        print(f"Starting voyager with {points_arr.shape[0]} points.")
        optimized_path = voyager.create_path(point_arr=points_arr)

        if len(optimized_path.indices) != points_arr.shape[0]:
            if len(optimized_path.indices) > points_arr.shape[0]:
                optimized_path.indices = optimized_path.indices[:-1]

        second_run_grid = self.data_processor.first_calibration_data.loc[
            :, ["X", "Y", "azimuth", "elevation", "x_light", "y_light"]
        ].dropna()
        second_run_grid["new_indexes"] = optimized_path.indices
        second_run_grid.sort_values(by=["new_indexes"], inplace=True)
        return second_run_grid

    @staticmethod
    def estimate_fov(fov_data: pd.DataFrame) -> float:
        """
        Parameters
        ----------
        fov_data : pd.DataFrame
            dataframe from FovSeeker research.
        Returns
        -------
        float
            field of view boundary.
        """
        try:
            azimuths = fov_data["azimuth"].drop_duplicates().tolist()
            samples = [fov_data[fov_data["azimuth"] == az] for az in azimuths]
            fov_scopes = []
            for sample in samples:
                elevations = sample["elevation"].drop_duplicates().tolist()
                divided = [sample[sample["elevation"] == i] for i in elevations]
                r = np.array(
                    [
                        np.linalg.norm([s["x_light"].mean(), s["y_light"].mean()])
                        for s in divided
                    ]
                ).round(4)
                R = np.array(
                    [np.linalg.norm([s["X"].mean(), s["Y"].mean()]) for s in divided]
                ).round(4)
                r_dot = np.gradient(r, R)
                negative_derivatives = np.where(r_dot < 0)[0]
                fov = (
                    elevations[negative_derivatives[0]]
                    if negative_derivatives.size > 0
                    else elevations[-1]
                )
                fov_scopes.append(fov)
            return min(fov_scopes)
        except:
            print(
                "[WARNING] NO FOV DATA IN CALIBRATION DIRECTORY. RETURNING 70 AS DEFAULT."
            )
            return 70

    def __arguments_factory(self, kwargs: dict) -> None:
        """
        Creates object fields from argparser data.

        Parameters:
        ----------
            kwargs (dict): kwargs from argparser
        """
        for key, val in kwargs.items():
            setattr(self, key, val)
