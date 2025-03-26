import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any, Union

class GraphPlotter:
    def __init__(self, base_dir: str = "graphs", figsize: tuple = (10, 6), 
                 grid_style: Optional[Dict[str, Any]] = {'linestyle': '--', 'alpha': 0.6},
                 font_family: str = 'Liberation serif', 
                 base_font_size: int = 12):
        """
        Initialize a plotting utility with consistent styling and automatic directory handling.
        
        Parameters:
        -----------
        base_dir : str
            Root directory for saving plots (default: 'graphs')
        figsize : tuple
            Figure dimensions in inches (default: (10, 6))
        grid_style : dict or None
            Custom grid styling (default: {'linestyle': '--', 'alpha': 0.6})
        font_family : str
            Font family for all text elements (default: 'Liberation serif')
        base_font_size : int
            Base font size in points (default: 12)
        """
        plt.rcParams.update({
            'font.family': font_family,
            'font.size': base_font_size,
            'axes.titlesize': base_font_size + 2,
            'axes.labelsize': base_font_size,
        })
        np.set_printoptions(precision=2, suppress=True)

        self.base_dir = Path(base_dir)
        self._ensure_directory_exists(self.base_dir)
        
        self.figsize = figsize
        self.grid_style = grid_style or {'linestyle': '--', 'alpha': 0.6}
        self.current_figure: Optional[plt.Figure] = None
        self.current_axes: Optional[plt.Axes] = None

    def _ensure_directory_exists(self, path: Union[str, Path]) -> None:
        """Ensure target directory exists, create if necessary."""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create directory {path}: {str(e)}")

    def create_figure(self, xlabel: str, ylabel: str, title: Optional[str] = None) -> None:
        """
        Initialize a new figure with standardized formatting.
        
        Parameters:
        -----------
        xlabel : str
            Label for x-axis (supports LaTeX notation)
        ylabel : str
            Label for y-axis (supports LaTeX notation)
        title : str, optional
            Figure title
        """
        self.current_figure, self.current_axes = plt.subplots(figsize=self.figsize)
        self.current_axes.grid(True, **self.grid_style)
        self.current_axes.set_xlabel(xlabel, labelpad=10)
        self.current_axes.set_ylabel(ylabel, labelpad=10)
        if title:
            self.current_axes.set_title(title)

    def add_plot(self, x_data: np.ndarray, y_data: np.ndarray, 
                label: Optional[str] = None, **plot_kwargs) -> None:
        """
        Add a line plot to the current figure.
        
        Parameters:
        -----------
        x_data : array-like
            X-axis data points
        y_data : array-like
            Y-axis data points
        label : str, optional
            Legend label for this plot
        **plot_kwargs
            Additional styling parameters for plt.plot()
        """
        if self.current_axes is None:
            raise RuntimeError("Call create_figure() before adding plots")
        
        default_style = {'linewidth': 1.5, 'linestyle': '-'}
        default_style.update(plot_kwargs)
        self.current_axes.plot(x_data, y_data, label=label, **default_style)

    def add_scatter(self, x_data: np.ndarray, y_data: np.ndarray,
                   label: Optional[str] = None, **scatter_kwargs) -> None:
        """
        Add a scatter plot to the current figure.
        
        Parameters:
        -----------
        x_data : array-like
            X-axis data points
        y_data : array-like
            Y-axis data points
        label : str, optional
            Legend label for this plot
        **scatter_kwargs
            Additional styling parameters for plt.scatter()
        """
        if self.current_axes is None:
            raise RuntimeError("Call create_figure() before adding plots")
        
        default_style = {'s': 20, 'alpha': 0.7}
        default_style.update(scatter_kwargs)
        self.current_axes.scatter(x_data, y_data, label=label, **default_style)

    def save(self, subdir: str = "", filename: str = "plot.png", 
         dpi: int = 300, legend: bool = True, 
         legend_opts: Optional[Dict[str, Any]] = None,
         show: bool = True) -> None:
        """
        Save the current figure to disk with automatic directory handling and optional display.
        
        Parameters:
        -----------
        subdir : str
            Subdirectory within base_dir (default: "")
        filename : str
            Output filename (supports .png, .svg, .pdf, etc.)
        dpi : int
            Output resolution in dots per inch (default: 300)
        legend : bool
            Whether to show legend (default: True)
        legend_opts : dict, optional
            Custom legend parameters
        show : bool
            Whether to display the plot in notebook (default: True)
        """
        if self.current_figure is None:
            raise RuntimeError("No active figure to save")
        
        save_dir = self.base_dir / subdir
        self._ensure_directory_exists(save_dir)
        
        if legend:
            legend_defaults = {'loc': 'best', 'frameon': True}
            if legend_opts:
                legend_defaults.update(legend_opts)
            self.current_axes.legend(**legend_defaults)
        
        if show:
            plt.show()
        
        save_path = save_dir / filename
        self.current_figure.tight_layout()
        self.current_figure.savefig(save_path, bbox_inches='tight', dpi=dpi)
        
        plt.close(self.current_figure)
        self.current_figure = None
        self.current_axes = None
        
        print(f"Plot saved: {save_path}")
            