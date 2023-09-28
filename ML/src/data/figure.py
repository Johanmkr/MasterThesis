import matplotlib.pyplot as plt


# Set global plotting parameter
custom_params = {
    "figure.figsize": (10, 10),
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14,
    "axes.grid": False
}

params = plt.rcParams

params.update(custom_params)

class CustomFigure:
    def __init__(self, figsize:tuple=(10,10)) -> None:
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)
        self.axes = [self.ax]
        
    def __call__(self) -> tuple:
        return self.fig, self.ax
    
    def set_settings(self, settings:dict) -> None:
        self.ax.set(**settings)

    def gen_twinx(self, settings:dict) -> None:
        newax = self.ax.twinx()
        newax.set(**settings)
        self.axes.append(newax)

    def gen_twiny(self, settings:dict) -> None:
        newax = self.ax.twiny()
        newax.set(**settings)
        self.axes.append(newax)

    def gen_ax(self, settings:dict) -> None:
        newax = self.fig.add_axes(settings)
        self.axes.append(newax)
    
    def update_ax_settings(self, ax, settings:dict) -> None:
        ax.set(**settings)


if __name__=="__main__":
    pass