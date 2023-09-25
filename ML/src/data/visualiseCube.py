import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cube


from IPython import embed

class VisualiseCube:
    def __init__(self, cube:cube.Cube, sim_type:str=None, save_path:str=None, save_name:str=None, show:bool=True, axis:int=0) -> None:
        self.cube = cube
        self.data = self.cube.get_gradient() if sim_type == "gradient" else self.cube.get_laplacian() if sim_type == "laplacian" else self.cube.data
        self.savePath = save_path if save_path is not None else "./animations/"
        self.saveName = save_name if save_name is not None else f"seed{self.cube.seed}_{'gr' if self.cube.gr else 'newton'}_redshift{self.cube.redshift}"
        self.show = show
        self.axis = axis
        self.name = f"Seed: {self.cube.seed}, Gravity: {'GR' if self.cube.gr else 'Newton'}, Redshift: {self.cube.redshift}"
        # embed()
        self._initialise_figure()
        self._initialise_animation()
        # self._initialise_animation_writer()

    def _update_frame(self, i:int) -> None:
        if self.axis == 0:
            data = self.data[i, :, :]
        elif self.axis == 1:
            data = self.data[:, i, :]
        elif self.axis == 2:
            data = self.data[:, :, i]
        self.im.set_array(data)
        self.title.set_text(f"{self.name}, ax {self.axis} idx: {i}")
        return self.im,

    def _initialise_figure(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        # self.ax.set_axis_off()
        self.title = self.ax.set_title(f"{self.name}, ax {self.axis} idx: 0")
        # self.fig.tight_layout()
        if self.axis == 0:
            self.im = self.ax.imshow(self.data[0,:,:], cmap="viridis", origin="lower")
            self.ax.set_xlabel("axis 1")
            self.ax.set_ylabel("axis 2")
        elif self.axis == 1:
            self.im = self.ax.imshow(self.data[:,0,:], cmap="viridis", origin="lower")
            self.ax.set_xlabel("axis 0")
            self.ax.set_ylabel("axis 2")
        elif self.axis == 2:
            self.im = self.ax.imshow(self.data[:,:,0], cmap="viridis", origin="lower")
            self.ax.set_xlabel("axis 0")
            self.ax.set_ylabel("axis 1")
        self.fig.colorbar(self.im, label=r"$\phi$")

    def _initialise_animation(self) -> None:
        self.anim = animation.FuncAnimation(self.fig, self._update_frame, frames=self.data.shape[0], interval=50, blit=False)
        # self.anim.event_source.add_callback(self._update_title)

    def _initialise_animation_writer(self) -> None:
        self.writer = animation.writers["ffmpeg"](fps=30)
        self.writer.setup(self.fig, self.savePath + self.saveName + ".mp4")

class VisualiseDifference(VisualiseCube):
    def __init__(self, seed:int, redshift:int, axis:int=0, save_path:str=None, save_name:str=None, show:bool=True) -> None:
        self.seed = seed
        self.redshift = redshift
        self.axis = axis
        self.savePath = save_path if save_path is not None else "./animations/"
        self.saveName = save_name if save_name is not None else f"seed{self.seed}_redshift{self.redshift}"
        self.show = show
        self.name = f"Difference for seed: {self.seed}, Redshift: {self.redshift}: GR - Newton"
        
        self.GRcube = cube.Cube(f"/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/seed{self.seed:04d}/gr/gr_{cube.redshift_to_snap[self.redshift]}_phi.h5", normalise=True)
        self.Newtoncube = cube.Cube(f"/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/seed{self.seed:04d}/newton/newton_{cube.redshift_to_snap[self.redshift]}_phi.h5", normalise=True)
        self.data = (self.GRcube.data - self.Newtoncube.data)

        percentile1 = np.percentile(self.data, 1)
        percentile99 = np.percentile(self.data, 99)
        self.data = self.data.clip(percentile1, percentile99)
        self._initialise_figure()
        self._initialise_animation()



if __name__=="__main__":
    datapath = "/mn/stornext/d10/data/johanmkr/simulations/gevolution_first_runs/"
    if input("Enter cube manually?") in ["y", "yes", "Y"]:
        seed_nr = int(input("Enter seed [0000 - 1999]: "))
        gravity = input("Enter gravity [gr, newton]: ")
        redshift = int(input("Enter redshift [0, 1, 5, 10, 15, 20]: "))
        axis = int(input("Enter axis [0, 1, 2]: "))
    else:
        seed_nr = 1234
        gravity = "newton"
        redshift = 1
        axis=0
    path = datapath + f"seed{seed_nr:04d}/" + gravity + f"/{gravity}_{cube.redshift_to_snap[redshift]}_phi.h5"

    # obj = cube.Cube(path)
    # vis = VisualiseCube(obj, axis=axis)
    vis = VisualiseDifference(seed_nr, redshift, axis=axis)
    plt.show()

    # embed()
    