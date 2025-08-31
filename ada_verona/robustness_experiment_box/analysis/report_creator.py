import logging
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_style("darkgrid")
sns.set_theme(rc={"figure.figsize": (11.7, 8.27)})
sns.set_palette(sns.color_palette("Paired"))

logger = logging.getLogger(__name__)


class ReportCreator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate input data and log warnings for potential issues."""
        if self.df.empty:
            raise ValueError("DataFrame is empty")
        
        if "epsilon_value" not in self.df.columns:
            raise ValueError("DataFrame must contain 'epsilon_value' column")
        
        if "network" not in self.df.columns:
            raise ValueError("DataFrame must contain 'network' column")
        
        # Check for data quality issues
        total_points = len(self.df)
        unique_values = self.df["epsilon_value"].nunique()
        zero_variance_groups = []
        
        for network in self.df["network"].unique():
            network_data = self.df[self.df["network"] == network]["epsilon_value"]
            if network_data.nunique() <= 1:
                zero_variance_groups.append(network)
        
        if total_points < 5:
            logger.warning(f"Very few data points ({total_points}). Some plots may not be meaningful.")
        
        if unique_values < 3:
            logger.warning(f"Very few unique epsilon values ({unique_values}). KDE plots may fail.")
        
        if zero_variance_groups:
            logger.warning(f"Groups with zero variance (all identical values): {zero_variance_groups}")

    def _is_data_suitable_for_kde(self, data: pd.Series, min_unique: int = 3, min_points: int = 5) -> bool:
        """
        Check if data is suitable for KDE plotting.
        
        Args:
            data: Series of epsilon values
            min_unique: Minimum number of unique values required
            min_points: Minimum number of data points required
            
        Returns:
            bool: True if data is suitable for KDE
        """
        if len(data) < min_points:
            return False
        
        if data.nunique() < min_unique:
            return False
        
        # Check for zero variance
        if data.std() == 0:
            return False
        
        # Check for numerical issues (values too close together)
        return not data.max() - data.min() < 1e-10

    def _create_fallback_kde_plot(self, data: pd.DataFrame) -> plt.Figure:
        """
        Create a fallback plot when KDE fails.
        
        Args:
            data: DataFrame with epsilon_value and network columns
            
        Returns:
            plt.Figure: Fallback plot (histogram or text)
        """
        fig, ax = plt.subplots(figsize=(11.7, 8.27))
        
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("KDE Plot (No Data)")
        elif data["epsilon_value"].nunique() <= 1:
            # All values are identical
            unique_val = data["epsilon_value"].iloc[0]
            ax.text(0.5, 0.5, f"All epsilon values are identical: {unique_val:.6f}", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_title("KDE Plot (Zero Variance)")
        else:
            # Try histogram as fallback
            try:
                sns.histplot(data=data, x="epsilon_value", hue="network", multiple="stack", ax=ax)
                ax.set_title("KDE Plot (Fallback: Histogram)")
            except Exception as e:
                logger.error(f"Even histogram fallback failed: {e}")
                ax.text(0.5, 0.5, f"Plotting failed: {str(e)}", 
                       ha="center", va="center", transform=ax.transAxes)
                ax.set_title("KDE Plot (Error)")
        
        plt.close()
        return fig

    def create_hist_figure(self) -> plt.Figure:
        """Create histogram figure with error handling."""
        try:
            hist_plot = sns.histplot(data=self.df, x="epsilon_value", hue="network", multiple="stack")
            figure = hist_plot.get_figure()
            plt.close()
            return figure
        except Exception as e:
            logger.error(f"Failed to create histogram: {e}")
            return self._create_fallback_hist_plot()

    def _create_fallback_hist_plot(self) -> plt.Figure:
        """Create a simple fallback histogram."""
        fig, ax = plt.subplots(figsize=(11.7, 8.27))
        try:
            ax.hist(self.df["epsilon_value"], bins=min(20, len(self.df) // 2))
            ax.set_xlabel("Epsilon Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Epsilon Value Distribution")
        except Exception as e:
            ax.text(0.5, 0.5, f"Histogram failed: {str(e)}", 
                   ha="center", va="center", transform=ax.transAxes)
        plt.close()
        return fig

    def create_box_figure(self) -> plt.Figure:
        """Create box plot figure with error handling."""
        try:
            box_plot = sns.boxplot(data=self.df, x="network", y="epsilon_value")
            # Fix the tick label rotation issue
            plt.setp(box_plot.get_xticklabels(), rotation=90, ha="right")
            figure = box_plot.get_figure()
            plt.close()
            return figure
        except Exception as e:
            logger.error(f"Failed to create box plot: {e}")
            return self._create_fallback_box_plot()

    def _create_fallback_box_plot(self) -> plt.Figure:
        """Create a simple fallback box plot."""
        fig, ax = plt.subplots(figsize=(11.7, 8.27))
        try:
            # Manual box plot creation
            for i, network in enumerate(self.df["network"].unique()):
                network_data = self.df[self.df["network"] == network]["epsilon_value"]
                if not network_data.empty:
                    ax.boxplot(network_data, positions=[i], labels=[network])
            ax.set_xlabel("Network")
            ax.set_ylabel("Epsilon Value")
            ax.set_title("Epsilon Value Distribution by Network")
            plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
        except Exception as e:
            ax.text(0.5, 0.5, f"Box plot failed: {str(e)}", 
                   ha="center", va="center", transform=ax.transAxes)
        plt.close()
        return fig

    def create_kde_figure(self) -> plt.Figure:
        """Create KDE figure with robust error handling."""
        try:
            # Check if data is suitable for KDE
            if not self._is_data_suitable_for_kde(self.df["epsilon_value"]):
                logger.warning("Data not suitable for KDE, using fallback plot")
                return self._create_fallback_kde_plot(self.df)
            
            # Try KDE with error handling
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                kde_plot = sns.kdeplot(data=self.df, x="epsilon_value", hue="network", multiple="stack")
            
            figure = kde_plot.get_figure()
            plt.close()
            return figure
            
        except np.linalg.LinAlgError as e:
            logger.warning(f"KDE failed due to numerical issues: {e}")
            return self._create_fallback_kde_plot(self.df)
        except Exception as e:
            logger.error(f"Failed to create KDE plot: {e}")
            return self._create_fallback_kde_plot(self.df)

    def create_ecdf_figure(self) -> plt.Figure:
        """Create ECDF figure with error handling."""
        try:
            ecdf_plot = sns.ecdfplot(data=self.df, x="epsilon_value", hue="network")
            figure = ecdf_plot.get_figure()
            plt.close()
            return figure
        except Exception as e:
            logger.error(f"Failed to create ECDF plot: {e}")
            return self._create_fallback_ecdf_plot()

    def _create_fallback_ecdf_plot(self) -> plt.Figure:
        """Create a simple fallback ECDF plot."""
        fig, ax = plt.subplots(figsize=(11.7, 8.27))
        try:
            # Manual ECDF creation
            for network in self.df["network"].unique():
                network_data = self.df[self.df["network"] == network]["epsilon_value"].sort_values()
                if len(network_data) > 0:
                    y = np.arange(1, len(network_data) + 1) / len(network_data)
                    ax.step(network_data, y, label=network, where="post")
            ax.set_xlabel("Epsilon Value")
            ax.set_ylabel("Cumulative Probability")
            ax.set_title("Empirical CDF")
            ax.legend()
        except Exception as e:
            ax.text(0.5, 0.5, f"ECDF failed: {str(e)}", 
                   ha="center", va="center", transform=ax.transAxes)
        plt.close()
        return fig

    def create_anneplot(self) -> plt.Figure:
        """Create anneplot with error handling."""
        try:
            fig, ax = plt.subplots(figsize=(11.7, 8.27))
            
            for network in self.df["network"].unique():
                network_df = self.df[self.df["network"] == network].sort_values(by="epsilon_value")
                if len(network_df) > 0:
                    cdf_x = np.linspace(0, 1, len(network_df))
                    ax.plot(network_df["epsilon_value"], cdf_x, label=network)
                    
                    # Only fill if smallest_sat_value column exists
                    if "smallest_sat_value" in network_df.columns:
                        ax.fill_betweenx(cdf_x, network_df["epsilon_value"], 
                                        network_df["smallest_sat_value"], alpha=0.3)
            
            ax.set_xlim(0, 0.35)
            ax.set_xlabel("Epsilon values")
            ax.set_ylabel("Fraction critical epsilon values found")
            ax.legend()
            ax.set_title("Anne Plot")
            
            plt.close()
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create anne plot: {e}")
            return self._create_fallback_anne_plot()

    def _create_fallback_anne_plot(self) -> plt.Figure:
        """Create a simple fallback anne plot."""
        fig, ax = plt.subplots(figsize=(11.7, 8.27))
        try:
            # Simplified anne plot without smallest_sat_value
            for network in self.df["network"].unique():
                network_df = self.df[self.df["network"] == network].sort_values(by="epsilon_value")
                if len(network_df) > 0:
                    cdf_x = np.linspace(0, 1, len(network_df))
                    ax.plot(network_df["epsilon_value"], cdf_x, label=network)
            
            ax.set_xlim(0, 0.35)
            ax.set_xlabel("Epsilon values")
            ax.set_ylabel("Fraction critical epsilon values found")
            ax.legend()
            ax.set_title("Anne Plot (Simplified)")
        except Exception as e:
            ax.text(0.5, 0.5, f"Anne plot failed: {str(e)}", 
                   ha="center", va="center", transform=ax.transAxes)
        plt.close()
        return fig
