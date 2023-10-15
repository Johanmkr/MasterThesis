"""
    This module contains the paths to the data files.
"""
import pathlib as pl
import os

from .data import cube

# print(os.getcwd())
# from ..data import cube

datapath: pl.Path = pl.Path("/mn/stornext/d10/data/johanmkr/simulations")

simulation_path: pl.PosixPath = datapath / "gevolution_first_runs"

analysis_path: pl.PosixPath = datapath / "data_analysis"

class_output: pl.Path = pl.Path(
    "/uio/hume/student-u00/johanmkr/Documents/NbodySimulation/class_public/output/"
)


def get_cube_path(seed: int, gravity: str, redshift: int) -> pl.Path:
    return (
        simulation_path
        / f"seed{seed:04d}/{gravity}/{gravity}_{cube.redshift_to_snap[int(redshift)]}_phi.h5"
    )


def get_pickle_path(
    seed: int, gravity: str, redshift: int, k_res: int, theta_res: int, tag: str
) -> pl.PosixPath:
    return (
        analysis_path
        / "bispectra_dataframes"
        / f"bispectrum_{gravity}_s{seed:04d}_z{redshift}_k{k_res}_t{theta_res}_tag_{tag}.pkl"
    )


def get_power_spectra_path(seed: int, gravity: str) -> pl.Path:
    return simulation_path / f"seed{seed:04d}/{gravity}"


def get_dir_with_seed(seed: int) -> pl.Path:
    return simulation_path / f"seed{seed:04d}"


# Output paths

dataframe_path = analysis_path / "bispectra_dataframes"

figure_path: pl.Path = pl.Path(
    "/uio/hume/student-u00/johanmkr/Documents/thesis/writing/figures"
)

main_figure_path: pl.PosixPath = figure_path / "main"
temp_figure_path: pl.PosixPath = figure_path / "temp"
