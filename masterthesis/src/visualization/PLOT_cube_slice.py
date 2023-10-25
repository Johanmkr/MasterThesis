"""
    DESCRIPTION OF MODULE:

    
"""
# Gloabl import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Local imports
from src.data import cube
from src.utils.figure import CustomFigure, SaveShow
from src.utils import paths

redshifts = [0, 1, 10]
gravity_theories = ["gr", "newton"]

from IPython import embed


def cube_slices(seed=1234, cmap="viridis"):
    csfig = CustomFigure(
        ncols=3,
        nrows=2,
        figsize=(15, 10),
        constrained_layout=True,  # Improved layout,
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.06, "wspace": 0.06},
    )

    # Define Mpc/h values and corresponding ticks
    # mpch_ticks = np.arange(0, 5121, 20)
    # mpch_labels = [f"{val}" for val in mpch_ticks]

    for i, z in enumerate(redshifts):
        for j, t in enumerate(gravity_theories):
            idx_1d = i * 2 + j
            ax = csfig.axes[j][i]
            datacube = cube.Cube(paths.get_cube_path(seed, t, z))
            data = datacube[0]
            im = ax.imshow(data, cmap=cmap, origin="lower", extent=[0, 5120, 0, 5120])
            # ticks = ax.get_xticks()
            # embed()
            # ax.set_xticks(ticks * 20)
            # ax.set_yticks(ticks * 20)

            if j == 0:
                ax.set_title(f"Redshift: {z}")
            if j == 1:
                ax.set_xlabel("Mpc/h")

            if i == 0:
                ax.set_ylabel(
                    f"{t}".capitalize(),
                    fontdict={
                        "family": ax.title.get_fontfamily()[0],
                        "size": ax.title.get_fontsize(),
                        "weight": ax.title.get_fontweight(),
                    },
                )

    # gs = gridspec.GridSpec(ncols=3, nrows=2, width_ratios=[1, 2, 0.05], wspace=0.05)
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(data)
    # csfig.fig.subplots_adjust(right=0.85)
    # cbar_ax = csfig.fig.add_axes([1.15, 0.15, 0.05, 0.7])
    cbar = csfig.fig.colorbar(mappable, ax=csfig.axes.ravel().tolist(), label=r"$\phi$")
    # cbar_ax.set_rasterized(True)

    # Create a common colorbar to the right of the subplots
    # cbar = plt.colorbar(im, label=r"$\phi$")
    # plt.tight_layout()
    csfig.fig.suptitle(f"Cube slices with seed: {seed}")

    SaveShow(
        csfig,
        save_name="cube_slices_seed",
        save=True,
        show=True,
        tight_layout=False,
    )


def difference_in_cube_slices(seed=1234, cmap="viridis"):
    pass


if __name__ == "__main__":
    pass
