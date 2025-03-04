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
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def create_hist_figure(self) -> plt.Figure:
        hist_plot = sns.histplot(data=self.df, x="epsilon_value", hue="network", multiple="stack")
        figure = hist_plot.get_figure()

        plt.close()

        return figure

    def create_box_figure(self) -> plt.Figure:
        box_plot = sns.boxplot(data=self.df, x="network", y="epsilon_value")
        box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=90)

        figure = box_plot.get_figure()

        plt.close()

        return figure

    def create_kde_figure(self) -> plt.Figure:
        kde_plot = sns.kdeplot(data=self.df, x="epsilon_value", hue="network", multiple="stack")

        figure = kde_plot.get_figure()

        plt.close()

        return figure

    def create_ecdf_figure(self) -> plt.Figure:
        ecdf_plot = sns.ecdfplot(data=self.df, x="epsilon_value", hue="network")

        figure = ecdf_plot.get_figure()

        plt.close()

        return figure

    def create_anneplot(self):
        df = self.df
        for network in df.network.unique():
            df = df.sort_values(by="epsilon_value")
            cdf_x = np.linspace(0, 1, len(df))
            plt.plot(df.epsilon_value, cdf_x, label=network)
            plt.fill_betweenx(cdf_x, df.epsilon_value, df.smallest_sat_value, alpha=0.3)
            plt.xlim(0, 0.35)
            plt.xlabel("Epsilon values")
            plt.ylabel("Fraction critical epsilon values found")
            plt.legend()

        return plt.gca()
