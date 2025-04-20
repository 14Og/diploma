import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from random import randint
from skimage.measure import CircleModel
from scipy.interpolate import griddata, make_smoothing_spline, make_interp_spline
from typing import List, Tuple
from matplotlib.patches import FancyArrowPatch, Circle, Patch
import scienceplots
import warnings
from .dataprocessor import DataProcessor

plt.style.use(["grid", "notebook", "science"])
cmaps = list(plt.colormaps)

class Plotter:

    stand_color = "green"
    sensor_color = "black"
    colors = lambda r: [(1 - i / (1.1 * r), 0.5 * i / r, 0.3 * i / r) for i in range(r)]

    @classmethod
    def set_plotter_mode(cls, mode: str = "plotter") -> None:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.3
        plt.rcParams["grid.linestyle"] = "--"


        # plt.rc('text.latex',preamble=r'\\usepackage[utf8]{inputenc}')
        # plt.rc('text.latex',preamble=r'\\usepackage[russian]{babel}')
                
        if mode == "plotter":
            plt.rcParams["figure.figsize"] = (10, 10)
            plt.rcParams["figure.dpi"] = 100
        elif mode == "report":
            plt.rcParams["figure.figsize"] = (12, 12)
            plt.rcParams["figure.dpi"] = 300

    @classmethod
    def color(cls, i):
        colors = cls.colors(i)
        for color in colors:
            yield color

    @staticmethod
    @DataProcessor.csv_loader
    def plot_first_grid_measurements(log: pd.DataFrame | str) -> None:

        fig, (stand_ax, sensor_ax) = plt.subplots(1, 2)

        stand_ax.set_title("Stand coordinates")
        stand_ax.axhline(linewidth=0.8, color="black", linestyle="--")
        stand_ax.axvline(linewidth=0.8, color="black", linestyle="--")

        sensor_ax.set_title("Light coordinates")
        sensor_ax.axhline(linewidth=0.8, color="black", linestyle="--")
        sensor_ax.axvline(linewidth=0.8, color="black", linestyle="--")

        [ax.set_aspect("equal", adjustable="box", anchor="C") for ax in (stand_ax, sensor_ax)]

        stand_ax.set_xlabel("Stand X")
        stand_ax.set_ylabel("Stand Y")

        sensor_ax.set_xlabel("Sensor x")
        sensor_ax.set_ylabel("Sensor y")

        cmap = Plotter.colors(log.shape[0])

        stand_ax.scatter(log["X"], log["Y"], marker=".", s=5, c=Plotter.stand_color)
        sensor_ax.scatter(log["x_light"], log["y_light"], marker=".", s=5, c=Plotter.sensor_color)
        stand_ax.scatter(log["X"].mean(), log["Y"].mean(), s=15, c="red")
        sensor_ax.scatter(log["x_light"].mean(), log["y_light"].mean(), s=15, c="red")
        plt.show()

    @staticmethod
    def plot_interpolation_results(calibrated_frame: pd.DataFrame) -> None:
        fig, ax = plt.subplots(1, 2)
        [axi.set_aspect("equal", adjustable="box") for axi in ax.ravel()]
        sun_vector_ax, sensor_ax = ax
        sun_vector_ax.set_title("Interpolated sun vector coordinates")
        sun_vector_ax.set_xlabel("stand X")
        sun_vector_ax.set_ylabel("stand Y")

        sensor_ax.set_title("Regular sensor grid")
        sensor_ax.set_xlabel("sensor x")
        sensor_ax.set_ylabel("sensor y")

        sensor_ax.axhline(linewidth=0.8, color="black", linestyle="--")
        sensor_ax.axvline(linewidth=0.8, color="black", linestyle="--")
        sun_vector_ax.axhline(linewidth=0.8, color="black", linestyle="--")
        sun_vector_ax.axvline(linewidth=0.8, color="black", linestyle="--")

        for i, row in calibrated_frame.iterrows():
            try:
                c = Plotter.stand_color if not row["Extrapolated"] else "red"
            except:
                c = Plotter.stand_color
            sun_vector_ax.scatter(
                row["X"],
                row["Y"],
                marker=".",
                s=5,
                c=c,
            )
            try:
                c = Plotter.sensor_color if not row["Extrapolated"] else "red"
                s = 5 if not row["Extrapolated"] else 1
            except:
                c = Plotter.sensor_color if not pd.isna(row["X"]) else "red"
                s = 5 if not pd.isna(row["X"]) else 1
            sensor_ax.scatter(
                row["x_light"],
                row["y_light"],
                marker=".",
                s=s,
                c=c,
            )
        sun_vector_ax.scatter(
            calibrated_frame["X"].mean(),
            calibrated_frame["Y"].mean(),
            s=15,
            c="red",
        )
        sensor_ax.scatter(
            calibrated_frame["x_light"].mean(),
            calibrated_frame["y_light"].mean(),
            s=15,
            c="red",
        )
        plt.show()

    @staticmethod
    @DataProcessor.csv_loader
    def plot_adc_chanels(first_log: pd.DataFrame | str) -> None:
        fig, ax = plt.subplots(2, 2)
        ax = ax.ravel()
        [axi.set_aspect("equal", adjustable="box") for axi in ax]
        [axi.grid(False) for axi in ax]
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
        for i in range(4):
            contour = ax[i].tricontourf(
                first_log["X"],
                first_log["Y"],
                first_log[f"adc{i}_percent"],
                levels=80,
                cmap="BuPu",
            )
            ax[i].set_xlabel("stand X")
            ax[i].set_ylabel("stand Y")
            ax[i].set_title(f"adc {i}")
        ticks = np.linspace(
            first_log.loc[:, "adc0_percent":"adc3_percent"].to_numpy().min(),
            first_log.loc[:, "adc0_percent":"adc3_percent"].to_numpy().max(),
            10,
            endpoint=True,
        )
        fig.colorbar(
            contour,
            ax=ax,
            label="adc percent",
            orientation="horizontal",
            shrink=0.9,
            fraction=0.05,
            pad=0.1,
            ticks=ticks,
        )
        plt.show()

    @staticmethod
    @DataProcessor.csv_loader
    def plot_light_gradient(first_log: pd.DataFrame | str, gridspec: int = 50) -> None:
        warnings.filterwarnings("ignore")
        vector_fig, vector_ax = plt.subplots(2, 1, figsize=(8, 16))
        cor_fig, cor_ax = plt.subplots(2, 1, figsize=(8, 16))

        [axi.set_aspect("equal", adjustable="box") for axi in np.concatenate((vector_ax.ravel(), cor_ax.ravel()))]
        [axi.set_aspect("equal", adjustable="box") for axi in np.concatenate((vector_ax.ravel(), cor_ax.ravel()))]

        for i, light in enumerate(["x_light", "y_light"]):
            X = first_log["X"]
            Y = first_log["Y"]

            X_grid, Y_grid = np.meshgrid(
                np.linspace(X.min(), X.max(), gridspec),
                np.linspace(Y.min(), Y.max(), gridspec),
            )
            sensor_function = griddata(
                np.column_stack((X, Y)),
                first_log[light].to_numpy(),
                (X_grid, Y_grid),
                method="cubic",
            )

            grad_x, grad_y = np.nan_to_num(np.gradient(sensor_function), nan=0)
            mean_grad_x, mean_grad_y = np.mean(grad_x), np.mean(grad_y)
            np.set_printoptions(threshold=np.inf)
            mean_grad_products = np.dot(np.dstack([grad_x, grad_y]), [mean_grad_x, mean_grad_y]) / (
                np.linalg.norm([mean_grad_x, mean_grad_y]) * np.linalg.norm(np.dstack([grad_x, grad_y]), axis=2)
            )  # taking dot product and then finding cos of vectors
            # mean_grad_products = np.nan_to_num(mean_grad_products, nan=0)
            mean_grad_products = np.rad2deg(np.arccos(mean_grad_products))  #  from dot product to angle between vectors

            arrow = FancyArrowPatch(
                (0, 0),
                (30 * mean_grad_x, 30 * mean_grad_y),
                arrowstyle="simple",
                color="red",
                mutation_scale=10,
                label="mean gradient" if i == 0 else "",
            )
            contour = cor_ax[i].contourf(
                X_grid,
                Y_grid,
                mean_grad_products,
                levels=100,
                cmap="viridis_r",
                # vmax=40,
            )
            cor_ax[i].scatter(first_log["X"], first_log["Y"], c="black", marker=".", s=2)
            vector_ax[i].quiver(
                X_grid,
                Y_grid,
                grad_x,
                grad_y,
                color="black",
                scale=0.3,
                width=0.003,
                units="xy",
                label="gradient" if i == 0 else "",
            )
            vector_ax[i].add_patch(arrow)

        elevation_levels = np.arange(10,60,10)
        
        circles_x = [plt.Circle((0,0), np.sin(math.radians(elevation)), color='grey', fill=False) for elevation in elevation_levels]
        circles_y = [plt.Circle((0,0), np.sin(math.radians(elevation)), color='grey', fill=False) for elevation in elevation_levels]


        for circle_x, circle_y in zip(circles_x, circles_y):
            cor_ax[0].add_patch(circle_x)
            cor_ax[1].add_patch(circle_y)

        [axi.set_xlabel("X") for axi in np.concatenate((vector_ax.ravel(), cor_ax.ravel()))]
        [axi.set_ylabel("Y") for axi in np.concatenate((vector_ax.ravel(), cor_ax.ravel()))]
        vector_ax[0].set_title("Grad(x)")
        vector_ax[1].set_title("Grad(y)")
        cor_ax[0].set_title("Grad(x) angle with mean gradient")
        cor_ax[1].set_title("Grad(y) angle with mean gradient")

        ticks = np.linspace(0, 90, 11, endpoint=True)
        vector_fig.colorbar(
            contour,
            ax=(cor_ax),
            label="Angle with mean gradient vector,$^\\circ$",
            orientation="horizontal",
            shrink=0.9,
            fraction=0.05,
            pad=0.1,
        )
        warnings.filterwarnings("default")
        plt.show()

    @staticmethod
    @DataProcessor.csv_loader
    def plot_interpolated_functions(first_interpolation: pd.DataFrame | str) -> None:
        fig, (X_ax, Y_ax) = plt.subplots(1, 2, figsize=(16, 8))
        [axi.set_aspect("equal", adjustable="box") for axi in (X_ax, Y_ax)]
        first_interpolation["X"].fillna(first_interpolation["X"].max(), inplace=True)
        first_interpolation["Y"].fillna(first_interpolation["Y"].max(), inplace=True)

        X_ax.tricontourf(
            first_interpolation["x_light"],
            first_interpolation["y_light"],
            first_interpolation["X"],
            levels=20,
        )
        Y_ax.tricontourf(
            first_interpolation["x_light"],
            first_interpolation["y_light"],
            first_interpolation["Y"],
            levels=20,
        )
        plt.show()

    @staticmethod
    def plot_regular_sensor_grid_parameters(first_run: pd.DataFrame, first_calirbration: pd.DataFrame) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(
            first_run["x_light"],
            first_run["y_light"],
            c=Plotter.sensor_color,
            marker=".",
            s=1,
            label="first run points",
        )
        ax.scatter(
            first_calirbration["x_light"],
            first_calirbration["y_light"],
            c="red",
            marker=".",
            s=10,
            label="regular grid points",
        )
        ax.set_xlabel("sensor x")
        ax.set_ylabel("sensor y")
        ax.set_title("Regular grid to first run grid comparison")
        fig.legend()
        plt.show()

    @staticmethod
    @DataProcessor.csv_loader
    @DataProcessor.adc_modifier
    def plot_intensity_grid(first_log: pd.DataFrame | str, percent_treshold: float = 22.0) -> None:
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.set_xlabel("stand X")
        ax.set_ylabel("stand Y")
        a = ax.tricontourf(
            first_log["X"],
            first_log["Y"],
            first_log["min_adc"],
            levels=100,
            cmap="viridis",
            # vmin=percent_treshold,
        )
        for i, row in first_log.iterrows():

            ax.scatter(
                row["X"],
                row["Y"],
                marker=".",
                s=5,
                c="black" if row["min_adc"] >= percent_treshold else "red",
            )
        fig.colorbar(
            a,
            ax=ax,
            label="min adc percent",
            orientation="horizontal",
            shrink=0.9,
            fraction=0.05,
            pad=0.1,
        )
        ax.set_title(f"ADC treshold:{percent_treshold}%")
        plt.show()

    @staticmethod
    @DataProcessor.csv_loader
    @DataProcessor.adc_modifier
    def plot_3d_adc_percent(first_log: pd.DataFrame | str) -> None:
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        ax.grid(False)
        ax.scatter3D(first_log["X"], first_log["Y"], first_log["min_adc"], c="black", s=3)
        plt.show()

    @staticmethod
    def plot_3d_interpolation(interpolated_frame: pd.DataFrame | str, stage="second") -> None:
        with plt.style.context("science"):
            fig, (X_ax, Y_ax) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(20, 20))
            [axi.set_aspect("equal") for axi in (X_ax, Y_ax)]
            X_ax.grid(False)
            Y_ax.grid(False)
            X_ax.set_zlabel("X")
            Y_ax.set_zlabel("Y")
            Y_ax.view_init(0, 40, 0)
            X_ax.view_init(0, 320, 0)
            [axi.set_xlabel("Sensor x") for axi in (X_ax, Y_ax)]
            [axi.set_ylabel("Sensor y") for axi in (X_ax, Y_ax)]
            if stage == "second":
                ext = False
                int = False
                for i, row in interpolated_frame.iterrows():
                    label = None
                    if row["Extrapolated"]:
                        c = "red"
                        if not ext:
                            label = "Extrapolated"
                            ext = True
                    else:
                        c = "blue"
                        if not int:
                            label = "Interpolated"
                            int = True
                    X_ax.plot3D(
                        row["x_light"],
                        row["y_light"],
                        row["X"],
                        c=c,
                        marker=".",
                        label=label,
                    )
                    Y_ax.plot3D(
                        row["x_light"],
                        row["y_light"],
                        row["Y"],
                        c="red" if row["Extrapolated"] else "blue",
                        marker=".",
                    )

                X_ax.legend()
            else:
                X_ax.scatter3D(
                    interpolated_frame["x_light"],
                    interpolated_frame["y_light"],
                    interpolated_frame["X"],
                    c="blue",
                    marker=".",
                )
                Y_ax.scatter3D(
                    interpolated_frame["x_light"],
                    interpolated_frame["y_light"],
                    interpolated_frame["Y"],
                    c="blue",
                    marker=".",
                )
            plt.show()

    @staticmethod
    def plot_extrapolation_results(extrapolated_frame: pd.DataFrame) -> None:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        [axi.set_aspect("equal", adjustable="box") for axi in ax]
        [axi.set_xlabel("Sensor x") for axi in ax]
        [axi.set_ylabel("Sensor y") for axi in ax]
        X_ax, Y_ax = ax
        X_ax.set_title("X (extrapolated)")
        Y_ax.set_title("Y (extrapolated)")
        contour = X_ax.tricontourf(
            extrapolated_frame["x_light"],
            extrapolated_frame["y_light"],
            extrapolated_frame["X"],
            levels=100,
        )
        Y_ax.tricontourf(
            extrapolated_frame["x_light"],
            extrapolated_frame["y_light"],
            extrapolated_frame["Y"],
            levels=100,
        )
        [
            ax.set_xlim(
                extrapolated_frame["x_light"].min() - 0.01,
                extrapolated_frame["x_light"].max() + 0.01,
            )
            for ax in [X_ax, Y_ax]
        ]
        [
            ax.set_ylim(
                extrapolated_frame["y_light"].min() - 0.01,
                extrapolated_frame["y_light"].max() + 0.01,
            )
            for ax in [X_ax, Y_ax]
        ]
        for i, row in extrapolated_frame.iterrows():
            X_ax.scatter(
                row["x_light"],
                row["y_light"],
                c="red" if row["Extrapolated"] else "black",
                marker=".",
                s=5,
            )
            Y_ax.scatter(
                row["x_light"],
                row["y_light"],
                c="red" if row["Extrapolated"] else "black",
                marker=".",
                s=5,
            )
        ticks = np.linspace(-1, 1, 11, endpoint=True)
        fig.colorbar(
            contour,
            ax=([X_ax, Y_ax]),
            label="Sun vector coordinates",
            orientation="horizontal",
            shrink=0.9,
            fraction=0.05,
            pad=0.1,
            ticks=ticks,
        )
        plt.show()

    @staticmethod
    def plot_all_data_points(first_run_data: pd.DataFrame, second_run_data: pd.DataFrame, regular_grid_data) -> None:

        fig, (stand_ax, sensor_ax) = plt.subplots(1, 2)

        stand_ax.set_title("Stand coordinates")
        stand_ax.axhline(linewidth=0.8, color="black", linestyle="--")
        stand_ax.axvline(linewidth=0.8, color="black", linestyle="--")

        sensor_ax.set_title("Light coordinates")
        sensor_ax.axhline(linewidth=0.8, color="black", linestyle="--")
        sensor_ax.axvline(linewidth=0.8, color="black", linestyle="--")

        [ax.set_aspect("equal", adjustable="box", anchor="C") for ax in (stand_ax, sensor_ax)]
        stand_ax.set_xlabel("X")
        stand_ax.set_ylabel("Y")
        sensor_ax.set_xlabel("x")
        sensor_ax.set_ylabel("y")

        stand_ax.scatter(
            first_run_data["X"],
            first_run_data["Y"],
            c="black",
            s=3,
            marker=".",
            label="first run stand points",
        )
        stand_ax.scatter(
            second_run_data["X"],
            second_run_data["Y"],
            c="green",
            s=5,
            marker="*",
            label="second run stand points",
        )

        sensor_ax.scatter(
            first_run_data["x_light"],
            first_run_data["y_light"],
            c="blue",
            s=3,
            marker=".",
            label="first run sensor points",
        )
        sensor_ax.scatter(
            second_run_data["x_light"],
            second_run_data["y_light"],
            c="red",
            s=5,
            marker="*",
            label="second run sensor points",
        )

        sensor_ax.scatter(
            regular_grid_data["x_light"],
            regular_grid_data["y_light"],
            c="purple",
            s=10,
            marker=".",
            label="regular grid data",
        )

        fig.legend()
        plt.show()

    @staticmethod
    @DataProcessor.csv_loader
    def plot_first_calibration_vectors(first_log: pd.DataFrame | str, second_log: pd.DataFrame | str) -> None:
        if first_log.shape[0] != second_log.shape[0]:
            print(f"Warning! shapes ain't equal: {first_log.shape[0]} and {second_log.shape[0]}")
            error = abs(first_log.shape[0] - second_log.shape[0])
            if first_log.shape[0] > second_log.shape[0]:
                first_log.drop(first_log.tail(error).index, inplace=True)
            else:
                second_log.drop(second_log.tail(error).index, inplace=True)

        first_x = first_log["x_light"].to_numpy()
        first_y = first_log["y_light"].to_numpy()

        second_x = second_log["x_light"].to_numpy()
        second_y = second_log["y_light"].to_numpy()

        fig, ax = plt.subplots(1, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Sensor x")
        ax.set_ylabel("Sensor y")

        ax.scatter(first_x, first_y, marker=".", c="red", s=3, label="First calibration points")
        ax.scatter(
            second_x,
            second_y,
            marker=".",
            c="blue",
            s=3,
            label="Second calibration points",
        )
        ax.quiver(
            first_x,
            first_y,
            second_x - first_x,
            second_y - first_y,
            color="black",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            label="Error vectors",
        )
        ax.set_title("Vector error field")

        plt.show()

    @staticmethod
    @DataProcessor.csv_loader
    def plot_first_calibration_error_rate(first_log: pd.DataFrame | str, second_log: pd.DataFrame | str) -> None:
        if first_log.shape[0] != second_log.shape[0]:
            error = abs(first_log.shape[0] - second_log.shape[0])
            if first_log.shape[0] > second_log.shape[0]:
                first_log.drop(first_log.tail(error).index, inplace=True)
            else:
                second_log.drop(second_log.tail(error).index, inplace=True)

        first_x = first_log["x_light"].to_numpy()
        first_y = first_log["y_light"].to_numpy()

        second_x = second_log["x_light"].to_numpy()
        second_y = second_log["y_light"].to_numpy()
        vectors = np.column_stack((first_x - second_x, first_y - second_y))
        error = np.linalg.norm(vectors, axis=1)

        fig, ax = plt.subplots(1, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.set_xlabel("sensor x")
        ax.set_ylabel("sensor y")

        a = ax.tricontourf(first_x, first_y, error, cmap="Reds", levels=50)
        ax.set_title("Calibration error")
        fig.colorbar(a, label="error rate", shrink=0.7)
        plt.show()

    @staticmethod
    @DataProcessor.csv_loader
    def plot_light_shape_parameters(
        first_run_frame: pd.DataFrame,
        first_calibration_frame: pd.DataFrame | str,
        light_shape: Tuple,
    ) -> None:
        xc, yc, r = light_shape
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_aspect("equal", adjustable="box")
        circle = Circle(
            (xc, yc),
            r,
            fill=False,
            color="red",
            linewidth=1,
            label=f"xc:{xc:.5f}, yc:{yc:.5f}, r:{r:.5f}",
        )
        ax.scatter(
            first_calibration_frame["x_light"],
            first_calibration_frame["y_light"],
            marker=".",
            s=5,
        )
        ax.scatter(
            first_run_frame["x_light"],
            first_run_frame["y_light"],
            marker="*",
            s=5,
            c="black",
        )
        ax.add_patch(circle)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_noise_measurements(frame: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.grid(True, c="grey")
        ax.set_title("Noise in 4 different points")
        ax.set_xlabel("sensor x")
        ax.set_ylabel("sensor y")
        clusters = frame["angles"].drop_duplicates().tolist()
        clustered_data = []
        colors = ["blue", "black", "grey", "green"]
        centers = []
        for cluster in clusters:
            clustered_data.append(frame[frame["angles"] == cluster])
        for data, cluster, color in zip(clustered_data, clusters, colors):
            x_center = np.mean(data["light_x"].to_numpy())
            y_center = np.mean(data["light_y"].to_numpy())
            centers.append([x_center, y_center])

            std = np.std((data["light_x"].to_numpy() ** 2 + data["light_y"].to_numpy() ** 2) ** 0.5)
            x_std = np.std(data["light_x"]).round(5)
            y_std = np.std(data["light_y"]).round(5)
            std_circle = plt.Circle((x_center, y_center), std, color="red", linestyle="--", fill=False)
            ax.add_patch(std_circle)
            ax.scatter(
                data["light_x"],
                data["light_y"],
                marker=".",
                c=color,
                s=10,
                label=f"{cluster}",
            )
            ax.scatter(x_center, y_center, c="red", marker="+")
            plt.text(
                x_center + 0.001,
                y_center + 0.0006,
                f"deviation={round(std,5)}\nx deviation={x_std}\ny deviation={y_std}",
            )
        ax.plot(
            [centers[0][0], centers[3][0]],
            [centers[0][1], centers[3][1]],
            c="red",
            linestyle="--",
            linewidth=1,
        )
        ax.plot(
            [centers[1][0], centers[2][0]],
            [centers[1][1], centers[2][1]],
            c="red",
            linestyle="--",
            linewidth=1,
        )

        fig.legend(title="Azimuth angles, deg")
        plt.show()

    @staticmethod
    def plot_accumulation_deviation(data: pd.DataFrame) -> None:
        noises = data["accumulation"].drop_duplicates().tolist()
        noise_frames = [data[data["accumulation"] == noise] for noise in noises]
        fig = plt.figure()
        gs = fig.add_gridspec(2, len(noises))
        axes = []
        for i, frame in enumerate(noise_frames):
            if i > 0:
                axes.append(
                    fig.add_subplot(
                        gs[0, i],
                        sharex=axes[i - 1],
                        sharey=axes[i - 1],
                        title=f"accumulation:{noises[i]}",
                    )
                )
            else:
                axes.append(fig.add_subplot(gs[0, i], title=f"accumulation:{noises[i]}"))

            axes[i].grid(True, c="grey")
            axes[i].tick_params(axis="both", which="major", labelsize=8)
            axes[i].scatter(frame["x_light"], frame["y_light"], marker=".", s=3, c="black")

        std_plot = fig.add_subplot(gs[1, :])
        stds = []
        for frame in noise_frames:
            x = frame["x_light"].to_numpy() ** 2
            y = frame["y_light"].to_numpy() ** 2
            std = np.std((x + y) ** 0.5)
            stds.append(std)
        std_plot.grid(True, c="grey")
        std_plot.set_xlabel("accumulation rate")
        std_plot.set_ylabel("standart deviation")
        std_plot.plot(noises, stds, c="red")
        plt.show()

    @staticmethod
    @DataProcessor.csv_loader
    def plot_spherical_grid(grid: pd.DataFrame | str) -> None:
        if not "elevation" in grid.keys() or not "azimuth" in grid.keys():
            raise Exception("No angles in given grid.")

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(grid["azimuth"], grid["elevation"], marker=">", s=3, c="green")
        plt.show()

    @staticmethod
    def plot_stand_trajectory(path_frame: pd.DataFrame, times: List, steps: List, clusters: List) -> None:
        fig, ax = plt.subplots(2, 1, figsize=(20, 20))
        ax[0].set_xlabel("Azimuth")
        ax[0].set_ylabel("Elevation")
        ax[0].set_title("Trajectory")

        ax[1].set_xlabel("Clusters step")
        ax[1].set_ylabel("Total time")
        ax[1].set_title("Time consumption")

        ax[0].plot(
            path_frame["azimuth"],
            path_frame["elevation"],
            alpha=0.5,
            c="red",
            linestyle="--",
        )
        for i in range(path_frame.shape[0]):
            ax[0].text(path_frame["azimuth"][i], path_frame["elevation"][i], str(i), fontsize=5)
        for cluster in clusters:
            ax[0].axhline(y=cluster, c="black")
        ax[0].axhline(y=0, c="black")
        ax[1].plot(steps, times)

        plt.show()

    @staticmethod
    @DataProcessor.csv_loader
    def plot_second_run_results(first_calibration_log: pd.DataFrame | str, second_run_log: pd.DataFrame | str) -> None:
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))

        [axi.set_aspect("equal", adjustable="box") for axi in ax.ravel()]

        ax[0][1].set_title(label="calibrated sensor grid")
        ax[0][1].scatter(
            first_calibration_log["x_light"],
            first_calibration_log["y_light"],
            s=3,
            c="blue",
            marker=".",
        )
        ax[0][0].set_title(label="calibrated stand data")
        ax[0][0].scatter(
            first_calibration_log["X"],
            first_calibration_log["Y"],
            s=3,
            c="red",
            marker="*",
        )
        ax[1][1].set_title(label="actual sensor grid")
        ax[1][1].scatter(
            second_run_log["x_light"],
            second_run_log["y_light"],
            s=3,
            c="blue",
            marker=".",
        )
        ax[1][0].set_title(label="calibrated stand data")
        ax[1][0].scatter(
            second_run_log["X"],
            second_run_log["Y"],
            s=3,
            c="red",
            marker="*",
        )
        plt.tight_layout(h_pad=2, w_pad=2)
        plt.show()

    @staticmethod
    @DataProcessor.csv_loader
    def plot_sensor_functions(first_log: pd.DataFrame | str):
        with plt.style.context("classic"):
            fig = plt.figure()

            ax1 = fig.add_subplot(1, 2, 2, projection="3d")
            ax1.set_xlabel("stand X")
            ax1.set_ylabel("stand Y")
            ax1.set_title("sensor x")

            ax2 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2.set_xlabel("stand X")
            ax2.set_ylabel("stand Y")
            ax2.set_title("sensor y")

            ax1.scatter(
                first_log["X"],
                first_log["Y"],
                first_log["x_light"],
                c="green",
                marker=".",
            )
            ax2.scatter(
                first_log["X"],
                first_log["Y"],
                first_log["y_light"],
                c="green",
                marker=".",
            )
            plt.show()

    @staticmethod
    def plot_elevation_radius_function(log: pd.DataFrame) -> None:
        el_points = log["elevation"].drop_duplicates().tolist()
        clusters = []
        for el in el_points:
            clusters.append(log[log["elevation"] == el].reset_index())

        model = CircleModel()
        radiuses = []
        for cluster in clusters:
            model.estimate(cluster.loc["x_light":"y_light"].to_numpy())
            radiuses.append(model.params[-1])

        fig, (points_ax, formula_ax) = plt.subplots(1, 2, figsize=(8, 6))
        points_ax.grid(True, c="grey")
        formula_ax.grid(True, c="grey")
        points_ax.set_xlabel("elevation")
        points_ax.set_ylabel("points radius")
        points_ax.set_title("Radius from elevation function (by points)")

        formula_ax.set_xlabel("elevation")
        formula_ax.set_ylabel("points radius")
        formula_ax.set_title("Radius from elevation function (by formula)")
        points_ax.plot(np.deg2rad(el_points), radiuses, marker=".", c="black")
        space = np.linspace(math.radians(min(el_points)), math.radians(max(el_points)), 100)
        formula_ax.plot(space, 0.25 * space, c="red")
        print(cluster)
        plt.show()

    @staticmethod
    @DataProcessor.csv_loader
    def plot_second_calibration_comparison(
        second_run_data: pd.DataFrame | str, second_calibration_data: pd.DataFrame | str
    ) -> None:
        fig, (stand_ax, sensor_ax) = plt.subplots(1, 2, figsize=(12, 6))
        [axi.set_aspect("equal") for axi in (stand_ax, sensor_ax)]

        stand_ax.set_title("Sun vector coordinates")
        stand_ax.set_xlabel("X")
        stand_ax.set_ylabel("Y")

        sensor_ax.set_title("Sensor coordinates")
        sensor_ax.set_xlabel("x")
        sensor_ax.set_ylabel("y")

        stand_ax.scatter(
            second_run_data["X"],
            second_run_data["Y"],
            c="grey",
            marker=".",
            s=3,
            label="second run",
        )
        stand_ax.scatter(
            second_calibration_data["X"],
            second_calibration_data["Y"],
            c="red",
            marker="*",
            s=3,
            label="second calibration",
        )

        sensor_ax.scatter(
            second_run_data["x_light"],
            second_run_data["y_light"],
            c="grey",
            marker=".",
            s=3,
        )
        sensor_ax.scatter(
            second_calibration_data["x_light"],
            second_calibration_data["y_light"],
            c="red",
            marker="*",
            s=3,
        )
        fig.legend()
        plt.show()

    @staticmethod
    @DataProcessor.csv_loader
    def plot_sun_vector_functions(first_log: pd.DataFrame | str) -> None:
        fig, (x_ax, y_ax) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
        x_ax.scatter3D(first_log["X"], first_log["Y"], first_log["x_light"])
        y_ax.scatter3D(first_log["X"], first_log["Y"], first_log["y_light"])
        x_ax.grid(False)
        y_ax.grid(False)
        plt.show()

    @staticmethod
    def plot_fov_preestimation(fov_frame: pd.DataFrame) -> None:
        fig, ax = plt.subplots(2, 1, figsize=(8, 16))
        ax[0].set_xlabel("R", fontsize=20)
        ax[0].set_ylabel("r", fontsize=20)
        ax[1].set_xlabel("R", fontsize=20)
        ax[1].set_ylabel("dr/dR", fontsize=20)

        def coord2elplot(deg: List[float]) -> None:
            for d in deg:
                coordinate = math.sin(math.radians(d))
                ax[1].axvline(coordinate, c="black", linestyle="--", lw=1)
                ax[1].text(
                    coordinate + 0.008,
                    0.8,
                    f"elevation = {d}$^\\circ$",
                    rotation=90,
                    fontsize=22,
                )

        azimuths = fov_frame["azimuth"].drop_duplicates().tolist()
        az_gen = azimuths.__iter__()
        c = Plotter.color(azimuths.__len__())
        samples = [fov_frame[fov_frame["azimuth"] == az] for az in azimuths]
        for sample in samples:
            elevations = sample["elevation"].drop_duplicates().tolist()
            divided = [sample[sample["elevation"] == i] for i in elevations]
            r = np.array([np.linalg.norm([s["x_light"].mean(), s["y_light"].mean()]) for s in divided])
            R = np.array([np.linalg.norm([s["X"].mean(), s["Y"].mean()]) for s in divided])
            r_dot = np.gradient(r, R)
            r_spline = make_smoothing_spline(R, r)
            r_dot_spline = make_smoothing_spline(R, r_dot)
            R_ = np.linspace(R.max(), R.min(), 100)
            r_ = r_spline(R_)
            r_dot_ = r_dot_spline(R_)
            color = next(c)
            coord2elplot([45, 60, 65, 70])
            ax[0].plot(R_, r_, c=color, label=f"azimuth: {next(az_gen)}$^\\circ$")
            ax[1].plot(R_, r_dot_, c=color)

        ax[0].set_title("r(R)", fontsize=25)
        ax[1].set_title("dr/dR(R)", fontsize=25)
        ax[1].axhline(0, c="black", lw=3)
        ax[1].text(R_.min(), 0.03, "field of view treshold", fontsize=25)
        ax[1].set_ylim(top=1.5)
        ax[0].legend(loc=2)
        plt.show()

    @staticmethod
    def plot_error_test_results(error_frame: pd.DataFrame | str, fov_scope: float = 70) -> None:
        error_frame = error_frame.loc[error_frame["elevation"] < fov_scope]
        stand_X = error_frame["X"]
        stand_Y = error_frame["Y"]
        stand_Z = (1 - stand_X**2 - stand_Y**2) ** 0.5

        algo_X = error_frame["X_interp"]
        algo_Y = error_frame["Y_interp"]
        algo_Z = (1 - algo_X**2 - algo_Y**2) ** 0.5

        error_angle = np.degrees(np.arccos(stand_X * algo_X + stand_Y * algo_Y + stand_Z * algo_Z))

        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("Elevation, $^\\circ$")
        ax.set_ylabel("Azimuth, $^\\circ$")
        ax.set_title("Orientation algorithm error distribution")

        c = ax.tricontourf(
            error_frame["elevation"],
            error_frame["azimuth"],
            error_angle,
            levels=100,
            cmap="viridis_r",
        )
        fig.colorbar(
            c,
            ax=(ax),
            label="Total angle error, $^\\circ$",
            orientation="horizontal",
            shrink=0.9,
            fraction=0.05,
            pad=0.1,
        )
        plt.show()

    @staticmethod
    def plot_3d_error_estimation(error_frame: pd.DataFrame, fov_scope=None) -> None:
        if fov_scope:
            error_frame = error_frame.loc[error_frame["elevation"] < fov_scope]
        stand_X = error_frame["X"]
        stand_Y = error_frame["Y"]
        stand_Z = (1 - stand_X**2 - stand_Y**2) ** 0.5

        algo_X = error_frame["X_interp"]
        algo_Y = error_frame["Y_interp"]
        algo_Z = (1 - algo_X**2 - algo_Y**2) ** 0.5

        error_angle = np.degrees(np.arccos(stand_X * algo_X + stand_Y * algo_Y + stand_Z * algo_Z))
        mean_error = np.mean(error_angle)
        std_error = np.std(error_angle)

        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(12, 12))
        ax.view_init(15, 30, 0)
        ax.set_xlabel("Elevation, $^\\circ$")
        ax.set_ylabel("Azimuth,  $^\\circ$")
        ax.set_zlabel("Total error, $^\\circ$")
        ax.grid(False)

        good_point_label = False
        bad_point_label = False
        for error, (index, row) in zip(error_angle, error_frame.iterrows()):
            c = (None,)
            marker = None
            label = None
            if error > mean_error + std_error:
                marker = "*"
                c = "red"
                label = None if bad_point_label else "Error more that mean + std"
                bad_point_label = True
            else:
                marker = "."
                c = "black"
                label = None if good_point_label else "Error is less than mean + std"
                good_point_label = True

            ax.scatter3D(row["elevation"], row["azimuth"], error, marker=marker, c=c, label=label)

        extra_string = (
            f"Mean error is: {mean_error:0.3f} $^\\circ$\n Error standart deviation is: {std_error:0.3f} $^\\circ$"
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(color="none", label=extra_string))
        fig.legend(handles=handles)
        plt.tight_layout()
        plt.show()
