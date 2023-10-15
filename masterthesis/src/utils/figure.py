"""
    DESCRIPTION OF MODULE:

    
"""

# Global imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Local imports
import paths

# Temporary imports
from IPython import embed


# Set global plotting parameter
custom_params = {
    "figure.figsize": (10, 10),
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
    "axes.grid": False,
}

params = plt.rcParams

params.update(custom_params)


class CustomFigure:
    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple = (10, 10),
        settings: dict = None,
        *args,
        **kwargs,
    ) -> None:
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(nrows, ncols, figsize=self.figsize, **kwargs)
        self.axes = self.ax if isinstance(self.ax, np.ndarray) else [self.ax]
        # embed()
        self.lines = (
            {"main": []}
            if not isinstance(self.ax, np.ndarray)
            else [{f"ax{i}": []} for i in range(len(self.ax.flatten()))]
        )
        if settings is not None:
            self.set_settings(settings)

    def __call__(self) -> tuple:
        return self.fig, self.ax

    def set_settings(self, settings: dict) -> plt.axis:
        try:
            self.ax.set(**settings)
        except AttributeError:
            self.ax[0].set(**settings)

    def gen_twinx(self, settings: dict) -> plt.axis:
        newax = self.ax.twinx()
        newax.set(**settings)
        self.axes.append(newax)
        return newax

    def gen_twiny(self, settings: dict) -> plt.axis:
        newax = self.ax.twiny()
        newax.set(**settings)
        self.axes.append(newax)
        return newax

    def gen_ax(self, settings: dict) -> plt.axis:
        newax = self.fig.add_axes(settings)
        self.axes.append(newax)
        return newax

    def update_ax_settings(self, ax, settings: dict) -> None:
        ax.set(**settings)


def SaveShow(
    fig: plt.Figure | CustomFigure = None,
    save_name: str = None,
    save: bool = False,
    show: bool = False,
    tight_layout: bool = True,
    dpi: int = 300,
    *args,
    **kwargs,
) -> None:
    """Handles the saving and showing of figures.

    Args:
        fig (Union[plt.Figure, CustomFigure], optional): Figure object, either as plt figure or custom figure. Defaults to None.
        save_name (str, optional): Name to save the figure as. Will be extended with .pdf and .png. Defaults to None.
        save (bool, optional): To save or not. Defaults to False.
        show (bool, optional): To show or not. Defaults to False.
        tight_layout (bool, optional): To have tight layout or not. Defaults to True.
        dpi (int, optional): DPI of high resolution pdf. Defaults to 300.

    Raises:
        ValueError: If the save_name is not specified when save=True.
        ValueError: When fig is not specified when save=True.
    """
    if isinstance(fig, CustomFigure):
        fig = fig.fig

    if tight_layout:
        fig.tight_layout()

    if save:
        if save_name is None:
            raise ValueError("save_name must be specified if save=True")
        if fig is None:
            raise ValueError("fig must be specified if save=True")

        # Save high-resolutoin pdf
        pdf_save_name = save_name.replace(".pdf", "").replace(".png", "") + ".pdf"
        with PdfPages(paths.main_figure_path / pdf_save_name) as pdf:
            pdf.savefig(fig, dpi=dpi, *args, **kwargs)
        # fig.savefig (paths.main_figure_path / pdf_save_name, dpi=dpi, *args, **kwargs)

        # Save low-resolution png
        png_save_name = save_name.replace(".png", "").replace(".pdf", "") + ".png"
        fig.savefig(paths.temp_figure_path / png_save_name, dpi=72, *args, **kwargs)
    if show:
        plt.show()


if __name__ == "__main__":
    pass
