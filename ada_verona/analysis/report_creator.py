# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_style("darkgrid")
sns.set_theme(rc={"figure.figsize": (11.7, 8.27)})
sns.set_palette(sns.color_palette("Paired"))


class ReportCreator:
    """Render robustness-distribution plots from a result DataFrame.

    By default the plots group by ``network`` and visualise the per-input
    ``epsilon_value`` column, which matches the result frames produced by
    :class:`~ada_verona.database.experiment_repository.ExperimentRepository`.

    Both the grouping column and the value column may be overridden, so the same
    plots can compare along *any* dimension -- for example ``group_by="attack"``
    to overlay several attacks.

    Args:
        df: The result DataFrame to plot.
        group_by: Column used to split/colour the data (the plot's ``hue``).
        value_column: Column holding the per-input value to plot (the x-axis).
        value_label: Human-readable axis label for ``value_column``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        group_by: str = "network",
        value_column: str = "epsilon_value",
        value_label: str | None = None,
    ) -> None:
        self.df = df
        self.group_by = group_by
        self.value_column = value_column
        self.value_label = value_label or value_column.replace("_", " ").capitalize()

    @property
    def group_label(self) -> str:
        """Prettified name of the grouping column, used in labels and legends."""
        return self.group_by.replace("_", " ").capitalize()

    def _finalize(self, ax: plt.Axes, *, xlabel: str, ylabel: str, title: str) -> plt.Figure:
        """Apply shared labelling, detach the figure and close it for headless use."""
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(self.group_label)
        figure = ax.get_figure()
        plt.close()
        return figure

    def create_hist_figure(self) -> plt.Figure:
        ax = sns.histplot(data=self.df, x=self.value_column, hue=self.group_by, multiple="stack")
        return self._finalize(
            ax,
            xlabel=self.value_label,
            ylabel="Number of inputs",
            title=f"Robustness distribution per {self.group_label} (histogram)",
        )

    def create_box_figure(self) -> plt.Figure:
        ax = sns.boxplot(data=self.df, x=self.group_by, y=self.value_column)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        return self._finalize(
            ax,
            xlabel=self.group_label,
            ylabel=self.value_label,
            title=f"{self.value_label} per {self.group_label}",
        )

    def create_kde_figure(self) -> plt.Figure:
        ax = sns.kdeplot(data=self.df, x=self.value_column, hue=self.group_by, multiple="stack")
        return self._finalize(
            ax,
            xlabel=self.value_label,
            ylabel="Density",
            title=f"Robustness distribution per {self.group_label} (KDE)",
        )

    def create_ecdf_figure(self) -> plt.Figure:
        ax = sns.ecdfplot(data=self.df, x=self.value_column, hue=self.group_by)
        return self._finalize(
            ax,
            xlabel=self.value_label,
            ylabel="Proportion of inputs",
            title=f"Robustness distribution per {self.group_label} (ECDF)",
        )

    def create_anneplot(self):
        df = self.df
        for group in df[self.group_by].unique():
            df = df.sort_values(by=self.value_column)
            cdf_x = np.linspace(0, 1, len(df))
            plt.plot(df[self.value_column], cdf_x, label=group)
            plt.fill_betweenx(cdf_x, df[self.value_column], df.smallest_sat_value, alpha=0.3)
            plt.xlim(0, 0.35)
            plt.xlabel(self.value_label)
            plt.ylabel("Fraction critical epsilon values found")
            plt.legend(title=self.group_label)

        return plt.gca()
