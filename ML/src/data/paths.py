"""
    This module contains the paths to the data files.
"""
import pathlib as pl

datapath:pl.Path = pl.Path("/mn/stornext/d10/data/johanmkr/simulations")

simulation_path:pl.PosixPath = datapath / "gevolution_first_runs"

analysis_path:pl.PosixPath = datapath / "data_analysis"

class_output:pl.Path = pl.Path("/uio/hume/student-u00/johanmkr/Documents/NbodySimulation/class_public/output/")

def get_power_spectra_path(seed:int, gravity:str) -> pl.Path:
    return simulation_path / f"seed{seed:04d}/{gravity}"

def get_dir_with_seed(seed:int) -> pl.Path:
    return simulation_path / f"seed{seed:04d}"




# Output paths

figure_path:pl.Path = pl.Path("/uio/hume/student-u00/johanmkr/Documents/thesis/writing/figures")

main_figure_path:pl.PosixPath = figure_path / "main"
temp_figure_path:pl.PosixPath = figure_path / "temp"



