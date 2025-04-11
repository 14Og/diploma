import sys

sys.path.append("/home/kolya/Yandex.Disk/diploma/Sources")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import animation
from matplotlib.animation import PillowWriter

import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from itertools import chain
import statistics

from WorkImpl.Utils.plotter import Plotter
from WorkImpl.Utils.dataprocessor import DataProcessor
from Database import PanelDataBase
from Vector import Vector
from Polynomial.Polynomial import PolynomialFitter


from typing import List, Tuple

Plotter.set_plotter_mode("report")

degree_symbol = "$^\\circ$"
colormap = "magma_r"


class ResearchPlotter(Plotter):
        
    data_base = PanelDataBase()

    @staticmethod
    def plot_overall_model_stability_histogram(
        panel_name: str, stats_dataframe: pd.DataFrame, *labels: Tuple[str]
    ) -> None:
        fig, axes = plt.subplots(len(labels), 2, constrained_layout=True, figsize=(16, 5 * len(labels)))
        if len(labels) == 1:
            axes = (axes,)
        fig.suptitle(f"Model sustainability for {stats_dataframe.shape[0]} datasets", fontsize=20)

        for (ax_mean, ax_std), (label_mean, label_std) in zip(axes, labels):
            mean = stats_dataframe[label_mean]
            std = stats_dataframe[label_std]

            ax_mean.set_xlabel(f"Error mean, {degree_symbol}")
            ax_mean.set_ylabel("Frequency, datasets")
            ax_mean.set_title(label_mean)
            # ax_mean.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
            ax_mean.text(
                0.05,
                1,
                f"Overall mean: {mean.mean().__round__(3)}{degree_symbol}",
                transform=ax_mean.transAxes,
                bbox={"facecolor": "white", "edgecolor": "black"},
                fontsize=12,
            )
            ax_mean.hist(mean, bins=5, color="grey", edgecolor="black")
            ax_mean.plot(mean, 0 * mean, "d", color="green")

            ax_std.set_xlabel(f"Error deviation, {degree_symbol}")
            ax_std.set_ylabel("Frequency, datasets")
            ax_std.set_title(label_std)
            # ax_std.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
            ax_std.text(
                0.05,
                1,
                f"Overall std: {std.std().__round__(3)}{degree_symbol}",
                transform=ax_std.transAxes,
                bbox={"facecolor": "white", "edgecolor": "black"},
                fontsize=12,
            )
            ax_std.hist(std, bins=5, color="orange", edgecolor="black")
            ax_std.plot(std, 0 * std, "d", color="green")

        plt.show()

    @staticmethod
    def plot_error_colormesh(panel_name: str, errors_dataframe: pd.DataFrame, *error_labels: str) -> None:
        panel_frame = errors_dataframe.loc[panel_name]
        errors = [panel_frame[error_label].to_numpy() for error_label in error_labels]
        min_error = np.array(errors).ravel().min()
        max_error = np.array(errors).ravel().max()

        az = panel_frame["azimuth"]
        el = panel_frame["elevation"]

        az_grid, el_grid = np.mgrid[
            slice(az.min(), az.max() + 30, 30),
            slice(el.min(), el.max() + 5, 5),
        ]

        error_grids = [griddata((el, az), error, (el_grid, az_grid), method="cubic") for error in errors]

        fig, axes = plt.subplots(len(errors), 1, constrained_layout=True)
        if len(errors) == 1:
            axes = (axes,)
        # plt.suptitle("Test error visualization", fontsize=25)
        for ax, label, error in zip(axes, error_labels, error_grids):
            ax.set_xlabel(f"Elevation, {degree_symbol}", fontsize=30)
            ax.set_ylabel(f"Azimuth, {degree_symbol}", fontsize=30)
            # ax.set_title(label)
            im = ax.pcolormesh(el_grid, az_grid, error, cmap=colormap, vmin=min_error, vmax=max_error, shading="auto")
            ax.text(
                0.05,
                0.95,
                f"Mean error: {error.mean().__round__(3)}{degree_symbol}",
                transform=ax.transAxes,
                bbox={"facecolor": "white", "edgecolor": "black"},
                fontsize=30,
            )
            ax.tick_params(labelsize=25)
            # for tick in ax.xaxis.get_major_ticks():
            #     tick.label.set_fontsize(25)
            # for tick in ax.yaxis.get_major_ticks():
            #     tick.label.set_fontsize(25)

        cbar = plt.colorbar(im, ax=axes, orientation="horizontal")
        cbar.ax.tick_params(labelsize=30)

        plt.show()

        pass

    @staticmethod
    def plot_initial_data(panel_name: str, animation_path: str | None = None) -> None:
        initial_data = ResearchPlotter.data_base[panel_name].initial_data
        X = initial_data["X"]
        Y = initial_data["Y"]

        x_light = initial_data["x_light"]
        y_light = initial_data["y_light"]

        fig, (X_ax, Y_ax) = plt.subplots(
            1, 2, subplot_kw={"projection": "3d"}, figsize=(16, 8), constrained_layout=True
        )

        Y_ax.view_init(20, 120, 0)
        X_ax.view_init(20, 30, 0)
        fig.suptitle(f"{panel_name}", fontsize=20)
        [axi.set_aspect("equal") for axi in (X_ax, Y_ax)]
        X_ax.grid(False)
        Y_ax.grid(False)

        [axi.set_xlabel("x") for axi in (X_ax, Y_ax)]
        [axi.set_ylabel("y") for axi in (X_ax, Y_ax)]

        X_ax.set_zlabel("X")
        Y_ax.set_zlabel("Y")

        X_ax.scatter3D(x_light, y_light, X, marker=".", c="green")
        Y_ax.scatter3D(x_light, y_light, Y, marker=".", c="green")

        if animation_path:

            def a(i):
                X_ax.view_init(elev=X_ax.elev, azim=X_ax.azim + 1, roll=X_ax.roll)
                Y_ax.view_init(elev=Y_ax.elev, azim=Y_ax.azim + 1, roll=Y_ax.roll)

            anime = animation.FuncAnimation(fig, a, frames=360, interval=50)
            anime.save(animation_path, writer="pillow", fps=30, dpi=100)

        else:
            plt.show()

    @staticmethod
    def plot_error_fov_statistics(panel_name: str, errors_dataframe: pd.DataFrame, *error_labels: str) -> None:
        panel_frame = errors_dataframe.loc[panel_name]
        q = 12
        step = ResearchPlotter.data_base.FOV_THRESHOLD // q
        names = [f"{i*step}-{(i+1)*step}{degree_symbol}" for i in range(q)]

        panel_frame["cluster"] = pd.qcut(panel_frame["elevation"], q=q, labels=False)

        clusters = panel_frame["cluster"].drop_duplicates().tolist()
        fig, axes = plt.subplots(len(error_labels), 1, constrained_layout=True)
        fig.suptitle("Error distribution in FOV clusters", fontsize=20)
        if len(error_labels) == 1:
            axes = (axes,)
        for ax, error_label in zip(axes, error_labels):
            data = []
            means = []
            for cluster in clusters:
                cluster_data = panel_frame[panel_frame["cluster"] == cluster][error_label]
                data.append(cluster_data)
                means.append(statistics.mean(list(chain(*data))))
            stats = cbook.boxplot_stats(data, labels=names)
            ax.set_ylabel(f"Error, {degree_symbol}")
            ax.set_xlabel("Cluster")
            ax.set_title(error_label)

            ax.bxp(
                stats,
                positions=clusters,
                patch_artist=True,
                boxprops={"facecolor": "bisque"},
                showfliers=False,
            )
            ax.step(clusters, means, where="mid", color="red", linewidth=2, label="Mean accumulation")
            ax.legend()
        plt.show()

    @staticmethod
    def plot_model_training_results(
        panel_name: str, train_dataframe: pd.DataFrame, *labels: str, animation_path=None
    ) -> None:

        x_light = train_dataframe["x_light"]
        y_light = train_dataframe["y_light"]
        X_trained = train_dataframe[labels[0]]
        Y_trained = train_dataframe[labels[1]]
        X = train_dataframe["X"]
        Y = train_dataframe["Y"]

        errors = np.array(
            [
                Vector.angle_between_vectors(Vector(X[i], Y[i]), Vector(X_trained[i], Y_trained[i]))
                for i in range(X.shape[0])
            ]
        )
        norm = plt.Normalize(errors.min(), errors.max())

        fig, (X_ax, Y_ax) = plt.subplots(
            1, 2, subplot_kw={"projection": "3d"}, figsize=(16, 8), constrained_layout=True
        )
        Y_ax.view_init(10, 120, 0)
        X_ax.view_init(10, 30, 0)
        fig.suptitle("Model training results", fontsize=25)
        [axi.set_aspect("equal") for axi in (X_ax, Y_ax)]

        X_ax.grid(False)
        Y_ax.grid(False)

        [axi.set_xlabel("sensor x") for axi in (X_ax, Y_ax)]
        [axi.set_ylabel("sensor y") for axi in (X_ax, Y_ax)]

        X_ax.set_zlabel(labels[0], fontsize=20)
        Y_ax.set_zlabel(labels[1], fontsize=20)

        scatter = X_ax.scatter3D(x_light, y_light, X_trained, marker="8", c=errors, cmap=colormap, norm=norm)
        Y_ax.scatter3D(x_light, y_light, Y_trained, marker="8", c=errors, cmap=colormap, norm=norm)
        cbar = plt.colorbar(scatter, ax=[X_ax, Y_ax], orientation="horizontal", label=f"Train error,{degree_symbol} ")

        if animation_path:

            def a(i):
                X_ax.view_init(elev=X_ax.elev, azim=X_ax.azim + 1, roll=X_ax.roll)
                Y_ax.view_init(elev=Y_ax.elev, azim=Y_ax.azim + 1, roll=Y_ax.roll)

            anime = animation.FuncAnimation(fig, a, frames=360, interval=50)
            anime.save(animation_path, writer="pillow", fps=30, dpi=100)
        else:
            plt.show()

    @staticmethod
    def plot_single_dataset_error_distribution(
        panel_name: str, error_dataframes: Tuple[pd.DataFrame], *error_labels: str
    ) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(25, 12))
        fig.suptitle(f"Dataset error distribution: {panel_name}", fontsize=20)
        ax.text(
            0.05,
            1,
            f"Dataset size: {error_dataframes[0].loc[panel_name].shape[0]}",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "black"},
            fontsize=12,
        )
        for frame, labels in zip(error_dataframes, error_labels):
            for label in labels:
                error = frame.loc[panel_name][label]
                ax.set_xlabel(f"Calibration error, {degree_symbol}")
                ax.set_ylabel("Frequency, testing samples")
                ax.hist(
                    error,
                    bins=5,
                    histtype="step",
                    label=f"{label} mean: {error.mean().__round__(3)}{degree_symbol}",
                )

        ax.legend(loc="upper right")
        plt.show()
        
    @staticmethod
    def plot_calibration_stats_from_dataframes(geom_frame: pd.DataFrame, linear_frame: pd.DataFrame, poly_frame: pd.DataFrame) -> None:
        
        pass


if __name__ == "__main__":
    panel = "g01-0001"
    poly = PolynomialFitter()
    # ResearchPlotter.plot_model_training_results(panel, poly.perform_training(panel), False, "X_cube", "Y_cube")
    ResearchPlotter.plot_initial_data(panel)