import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import animation

import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from itertools import chain
import statistics

from WorkImpl.Utils.plotter import Plotter
from Database import PanelDataBase
from Vector import Vector

from typing import List, Tuple, Dict

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
    def plot_error_colormesh(panel_name: str, errors_dataframe: pd.DataFrame, plot_label:str, *error_labels: str) -> None:
        panel_frame = errors_dataframe.loc[panel_name]
        errors = [panel_frame[error_label].dropna() for error_label in error_labels]
        mean_errors = [error.mean() for error in errors]
        min_error = np.array(errors).ravel().min()
        max_error = np.array(errors).ravel().max()

        az = panel_frame["azimuth"]
        el = panel_frame["elevation"]

        az_grid, el_grid = np.mgrid[
            slice(az.min(), az.max() + 30, 30),
            slice(el.min(), el.max() + 5, 5),
        ]

        error_grids = [griddata((el, az), error, (el_grid, az_grid), method="cubic") for error in errors]

        fig, axes = plt.subplots(len(errors), 1, constrained_layout=True, figsize=(18, 12))
        if len(errors) == 1:
            axes = (axes,)
        plt.suptitle(plot_label, fontsize=40)
        for ax, label, error, mean_error in zip(axes, error_labels, error_grids, mean_errors):
            ax.set_xlabel(f"Elevation, {degree_symbol}", fontsize=40)
            ax.set_ylabel(f"Azimuth, {degree_symbol}", fontsize=40)
            # ax.set_title(label)
            im = ax.pcolormesh(el_grid, az_grid, error, cmap=colormap, vmin=min_error, vmax=max_error, shading="auto")
            ax.text(
                0.05,
                0.95,
                f"Mean error: {round(mean_error, 3)}{degree_symbol}",
                transform=ax.transAxes,
                bbox={"facecolor": "white", "edgecolor": "black"},
                fontsize=40,
            )
            ax.tick_params(labelsize=40)


        cbar = plt.colorbar(im, ax=axes, orientation="horizontal")
        cbar.ax.tick_params(labelsize=40)
        cbar.set_label(f"Error, {degree_symbol}", size=40)

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
        fig.suptitle("Error distribution within sensor FOV regions", fontsize=30)
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
            ax.set_ylabel(f"Error, {degree_symbol}", fontsize=30)
            ax.set_xlabel("Sensor FOV Region", fontsize=30)
            ax.set_title(error_label, fontsize=30)
            ax.tick_params(labelsize=20)

            ax.bxp(
                stats,
                positions=clusters,
                patch_artist=True,
                boxprops={"facecolor": "bisque"},
                showfliers=False,
            )
            ax.step(clusters, means, where="mid", color="red", linewidth=2, label="Mean error accumulation")
            ax.legend(prop={'size':30})
        plt.show()

    @staticmethod
    def plot_model_training_results(
        panel_name: str, train_dataframe: pd.DataFrame, title:str, *labels: str, animation_path=None
    ) -> None:

        x_light = train_dataframe["x_light"]
        y_light = train_dataframe["y_light"]
        X_trained = train_dataframe[labels[0]].to_numpy()
        Y_trained = train_dataframe[labels[1]].to_numpy()
        X = train_dataframe["X"].to_numpy()
        Y = train_dataframe["Y"].to_numpy()
        

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
        fig.suptitle(title, fontsize=30)
        [axi.set_aspect("equal") for axi in (X_ax, Y_ax)]

        X_ax.grid(False)
        Y_ax.grid(False)

        [axi.set_xlabel("sensor x") for axi in (X_ax, Y_ax)]
        [axi.set_ylabel("sensor y") for axi in (X_ax, Y_ax)]

        X_ax.set_zlabel(labels[0], fontsize=20)
        Y_ax.set_zlabel(labels[1], fontsize=20)

        scatter = X_ax.scatter3D(x_light, y_light, X_trained, s=40, marker=".", c=errors, cmap=colormap, norm=norm)
        Y_ax.scatter3D(x_light, y_light, Y_trained, s=40, marker=".", c=errors, cmap=colormap, norm=norm)
        cbar = plt.colorbar(scatter, fraction=0.13, ax=[X_ax, Y_ax], orientation="horizontal")
        cbar.ax.tick_params(labelsize=30)
        cbar.set_label(f"Train error,{degree_symbol} ", size=30)


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
    def plot_mlp_scaling_effect(panel_name: str) -> None:
        scaler = StandardScaler()
        sensor_data = pd.concat([ResearchPlotter.data_base[panel_name].initial_data, 
                                ResearchPlotter.data_base[panel_name].second_run_data]).loc[:, ["x_light", "y_light"]]
        scaled_data = scaler.fit_transform(sensor_data)
        
        _, ax = plt.subplots(1, 2, figsize=(18,8))
        
        ax[0].set_title("Input data before scaling", fontsize=30)
        ax[0].scatter(sensor_data["x_light"], sensor_data["y_light"], c='green', marker='.')
        ax[0].set_xlabel("x sensor", fontsize=30)
        ax[0].set_ylabel("y sensor", fontsize=30)

        ax[1].set_title("Input data after scaling", fontsize=30)
        ax[1].scatter(scaled_data[:, 0], scaled_data[:, 1], c='green', marker='.')
        ax[1].set_xlabel("x sensor", fontsize=30)
        ax[1].set_ylabel("y sensor", fontsize=30)
        
        plt.show()
        
    @staticmethod
    def plot_models_precision_analysis(model_names: Tuple, metrics: Dict, metric_colors: Tuple) -> None:
        x = np.arange(len(model_names), step=1.2)
        w = 0.25
        multiplier = 0
        
        _, ax = plt.subplots(layout='constrained', figsize=(18, 8))
        for (attr, val), color in zip(metrics.items(), metric_colors):
            offset = w * multiplier
            rects = ax.bar(x + offset, val, w, label=attr, facecolor=color, alpha=0.9)
            ax.bar_label(rects, padding=3, fontsize=20)
            multiplier += 1
            
        ax.set_ylabel("Error,$^\\circ$", fontsize=30)
        ax.set_title("Models precision analysis", fontsize=30)
        ax.set_xticks(x + w, model_names, fontsize=30)
        ax.legend(loc='upper right', ncols=2, fontsize=20)

        plt.show()
        
if __name__ == "__main__":
    pass