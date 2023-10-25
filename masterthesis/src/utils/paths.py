"""
    This module contains the paths to the data files.
"""
import pathlib as pl
import os

from ..data import cube


################
# Input paths
################

datapath: pl.Path = pl.Path("/mn/stornext/d10/data/johanmkr/simulations")

simulation_path: pl.PosixPath = datapath / "gevolution_first_runs"

analysis_path: pl.PosixPath = datapath / "data_analysis"

class_output: pl.Path = pl.Path(
    "/uio/hume/student-u00/johanmkr/Documents/NbodySimulation/class_public/output/"
)

BFast_path: pl.Path = pl.Path("/uio/hume/student-u00/johanmkr/Documents/BFast")


################
# Output paths
################

dataframe_path_high_res = analysis_path / "bispectra_dataframes_high_res"
dataframe_path_low_res = analysis_path / "bispectra_dataframes_low_res"

figure_path: pl.Path = pl.Path(
    "/uio/hume/student-u00/johanmkr/Documents/thesis/masterthesis/writing/figures"
)

main_figure_path: pl.PosixPath = figure_path / "main"
temp_figure_path: pl.PosixPath = figure_path / "temp"

pre_computed_bispectra_path: pl.PosixPath = (
    analysis_path / "pre_computed_bispectra_bank"
)
################
# Functions
################


def get_cube_path(seed: int, gravity: str, redshift: int) -> pl.Path:
    return (
        simulation_path
        / f"seed{seed:04d}/{gravity}/{gravity}_{cube.redshift_to_snap[int(redshift)]}_phi.h5"
    )


def get_high_res_bispectrum(
    seed: int, gravity: str, redshift: int, k_res: int, theta_res: int, tag: str
) -> pl.PosixPath:
    return (
        dataframe_path_high_res
        / f"bispectrum_{gravity}_s{seed:04d}_z{redshift}_k{k_res}_t{theta_res}_tag_{tag}.pkl"
    )


def get_low_res_bispectrum(
    seed: int, gravity: str, redshift: int, type: str = "equilateral"
) -> pl.PosixPath:
    return (
        dataframe_path_low_res
        / f"{type}_bispectrum_{gravity}_s{seed:04d}_z{redshift}.pkl"
    )


def get_pre_computed_bispectra_from_bank(
    seed: int, gravity: str, redshift: int, type: str = "equilateral"
) -> pl.PosixPath:
    return (
        pre_computed_bispectra_path
        / f"seed{seed:04d}_{gravity}_{type}_rs{int(redshift):04d}.csv"
    )


def get_pre_computed_bispectra_from_bank2(
    seed: int, gravity: str, redshift: int, type: str = "equilateral"
) -> pl.PosixPath:
    return (
        pre_computed_bispectra_path
        / f"seed{seed:04d}_{gravity}_{type}_rs{int(redshift):04d}.pkl"
    )


def get_power_spectra_path(seed: int, gravity: str) -> pl.Path:
    return simulation_path / f"seed{seed:04d}/{gravity}"


def get_dir_with_seed(seed: int) -> pl.Path:
    return simulation_path / f"seed{seed:04d}"
