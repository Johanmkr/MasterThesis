import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from h5cube import Cube
from h5collection import Collection

from IPython import embed

class VisualiseCube:
    def __init__(self, cube:Cube, save_path:str=None, save_name:str=None, show:bool=True) -> None:
        self.cube = cube
        self.savePath = save_path if save_path is not None else "./"
        self.saveName = save_name if save_name is not None else "animation"
        self.show = show

        self._initialise_figure()
        self._initialise_animation()
        self._initialise_animation_writer()
    
    def _initialise_figure(self) -> None:
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.ax.set_axis_off()
        self.im = self.ax.imshow(self.cube[0], cmap="gray")

    def _initialise_animation(self) -> None:
        self.anim = animation.FuncAnimation(self.fig, self._update_frame, frames=self.cube.data.shape[0], interval=50, blit=True)

    def _initialise_animation_writer(self) -> None:
        self.writer = animation.writers["ffmpeg"](fps=30)
        self.writer.setup(self.fig, self.savePath + self.saveName + ".mp4")



if __name__=="__main__":
    datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"
    seed_nr = "seed1506/"
    