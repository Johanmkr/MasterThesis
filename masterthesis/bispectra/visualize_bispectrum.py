import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

bispectrumlink = "/mn/stornext/d10/data/johanmkr/simulations/bispectra_analysis/"


class BispectrumVariation_AS:
    def __init__(self, seed, z):
        self.seed = str(seed).zfill(4)  # make four digit string of seed
        self.z = int(z)  # make redshift integer
        self.A_s = 2.215 * np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5])  # array of A_s
        # self.A_s = 2.215 * np.array([1e-8, 1e-7, 1e-6, 1e-5])
        specific_link = bispectrumlink + f"z_{self.z}/seed{self.seed}/"

        # Equilateral spectra
        self.B_eq_gr = pd.DataFrame()
        self.B_eq_newton = pd.DataFrame()
        self.Q_eq_gr = pd.DataFrame()
        self.Q_eq_newton = pd.DataFrame()

        # Squeezed spectra
        self.B_sq_gr = pd.DataFrame()
        self.B_sq_newton = pd.DataFrame()
        self.Q_sq_gr = pd.DataFrame()
        self.Q_sq_newton = pd.DataFrame()

        # Stretched spectra
        self.B_st_gr = pd.DataFrame()
        self.B_st_newton = pd.DataFrame()
        self.Q_st_gr = pd.DataFrame()
        self.Q_st_newton = pd.DataFrame()

        # Power spectra
        self.Pk_gr = pd.DataFrame()
        self.Pk_newton = pd.DataFrame()

        count = 0
        for A_s in self.A_s:
            A_s = f"{A_s:.3e}"
            gr_frame = pd.read_pickle(specific_link + f"{A_s}_gr.pkl")
            newton_frame = pd.read_pickle(specific_link + f"{A_s}_newton.pkl")

            if count == 0:
                # Fill k
                self.B_eq_gr["k"] = self.B_sq_gr["k"] = self.B_st_gr["k"] = gr_frame[
                    "k"
                ]
                self.Q_eq_gr["k"] = self.Q_sq_gr["k"] = self.Q_st_gr["k"] = gr_frame[
                    "k"
                ]
                self.B_eq_newton["k"] = self.B_sq_newton["k"] = self.B_st_newton[
                    "k"
                ] = newton_frame["k"]
                self.Q_eq_newton["k"] = self.Q_sq_newton["k"] = self.Q_st_newton[
                    "k"
                ] = newton_frame["k"]
                self.Pk_gr["k"] = gr_frame["k"]
                self.Pk_newton["k"] = newton_frame["k"]

            # Fill bispectra
            self.B_eq_gr[A_s] = gr_frame["B_eq"]
            self.B_sq_gr[A_s] = gr_frame["B_sq"]
            self.B_st_gr[A_s] = gr_frame["B_st"]
            self.B_eq_newton[A_s] = newton_frame["B_eq"]
            self.B_sq_newton[A_s] = newton_frame["B_sq"]
            self.B_st_newton[A_s] = newton_frame["B_st"]

            # Fill reduced bisectra
            self.Q_eq_gr[A_s] = gr_frame["Q_eq"]
            self.Q_sq_gr[A_s] = gr_frame["Q_sq"]
            self.Q_st_gr[A_s] = gr_frame["Q_st"]
            self.Q_eq_newton[A_s] = newton_frame["Q_eq"]
            self.Q_sq_newton[A_s] = newton_frame["Q_sq"]
            self.Q_st_newton[A_s] = newton_frame["Q_st"]

            # Fill power spectra
            self.Pk_gr[A_s] = gr_frame["Pk"]
            self.Pk_newton[A_s] = newton_frame["Pk"]

            count += 1

    def plot_and_compare_bispectra(self):
        fig, ax = plt.subplots(3, 3, figsize=(15, 15))
        for i, A_s in enumerate(self.A_s):
            A_s = f"{A_s:.3e}"
            ax[0, 0].loglog(self.B_eq_gr["k"], abs(self.B_eq_gr[A_s]), label=A_s)
            ax[0, 1].loglog(self.B_sq_gr["k"], abs(self.B_sq_gr[A_s]), label=A_s)
            ax[0, 2].loglog(self.B_st_gr["k"], abs(self.B_st_gr[A_s]), label=A_s)
            ax[1, 0].loglog(
                self.B_eq_newton["k"], abs(self.B_eq_newton[A_s]), label=A_s
            )
            ax[1, 1].loglog(
                self.B_sq_newton["k"], abs(self.B_sq_newton[A_s]), label=A_s
            )
            ax[1, 2].loglog(
                self.B_st_newton["k"], abs(self.B_st_newton[A_s]), label=A_s
            )
            ax[2, 0].loglog(self.Pk_gr["k"], abs(self.Pk_gr[A_s]), label=A_s)
            ax[2, 1].loglog(self.Pk_newton["k"], abs(self.Pk_newton[A_s]), label=A_s)
        ax[0, 0].set_title("Equilateral, GR")
        ax[0, 1].set_title("Squeezed, GR")
        ax[0, 2].set_title("Stretched, GR")
        ax[1, 0].set_title("Equilateral, Newton")
        ax[1, 1].set_title("Squeezed, Newton")
        ax[1, 2].set_title("Stretched, Newton")
        ax[2, 0].set_title("Power, GR")
        ax[2, 1].set_title("Power, Newton")
        for ax in ax.flatten():
            ax.legend()
            ax.set_xlabel(r"$k$ [h/Mpc]")
            ax.set_ylabel(r"$B(k1, \mu, t)$ [Mpc^6/h^6]")
        plt.show()

    def plot_bispectra_in_same_plot(self):
        long_list_of_colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
        use_list_of_colors = long_list_of_colors[: len(self.A_s)]
        fig, ax = plt.subplots(
            3, 1, figsize=(15, 15), sharex=True, gridspec_kw={"hspace": 0.5}
        )
        tx0 = ax[0].inset_axes([0.0, 1, 1, 0.5])
        tx1 = ax[1].inset_axes([0.0, 1, 1, 0.5])
        tx2 = ax[2].inset_axes([0.0, 1, 1, 0.5])

        for i, A_s in enumerate(self.A_s):
            A_s = f"{A_s:.3e}"
            ax[0].loglog(
                np.average(self.B_eq_gr["k"].to_numpy().reshape(-1, 4), axis=1),
                np.average(np.abs(self.B_eq_gr[A_s].to_numpy()).reshape(-1, 4), axis=1),
                color=use_list_of_colors[i],
                ls="solid",
                label=r"$A_s={A_s}$".format(A_s=A_s),
            )
            ax[1].loglog(
                np.average(self.B_sq_gr["k"].to_numpy().reshape(-1, 4), axis=1),
                np.average(np.abs(self.B_sq_gr[A_s].to_numpy()).reshape(-1, 4), axis=1),
                color=use_list_of_colors[i],
                ls="solid",
                label=r"$A_s={A_s}$".format(A_s=A_s),
            )
            ax[2].loglog(
                np.average(self.B_st_gr["k"].to_numpy().reshape(-1, 4), axis=1),
                np.average(np.abs(self.B_st_gr[A_s].to_numpy()).reshape(-1, 4), axis=1),
                color=use_list_of_colors[i],
                ls="solid",
                label=r"$A_s={A_s}$".format(A_s=A_s),
            )

            ax[0].loglog(
                np.average(self.B_eq_newton["k"].to_numpy().reshape(-1, 4), axis=1),
                np.average(
                    np.abs(self.B_eq_newton[A_s].to_numpy()).reshape(-1, 4), axis=1
                ),
                color=use_list_of_colors[i],
                ls="dashed",
                # label=f"newton_{A_s}",
            )
            ax[1].loglog(
                np.average(self.B_sq_newton["k"].to_numpy().reshape(-1, 4), axis=1),
                np.average(
                    np.abs(self.B_sq_newton[A_s].to_numpy()).reshape(-1, 4), axis=1
                ),
                color=use_list_of_colors[i],
                ls="dashed",
                # label=f"newton_{A_s}",
            )
            ax[2].loglog(
                np.average(self.B_st_newton["k"].to_numpy().reshape(-1, 4), axis=1),
                np.average(
                    np.abs(self.B_st_newton[A_s].to_numpy()).reshape(-1, 4), axis=1
                ),
                color=use_list_of_colors[i],
                ls="dashed",
                # label=f"newton_{A_s}",
            )

            # Small panel underneath to show the relative difference between GR and Newton and show the values along a twiny axis on the right

            tx0.loglog(
                np.average(self.B_eq_gr["k"].to_numpy().reshape(-1, 4), axis=1),
                np.average(
                    np.abs(
                        (
                            self.B_eq_gr[A_s].to_numpy()
                            - self.B_eq_newton[A_s].to_numpy()
                        )
                        / self.B_eq_newton[A_s].to_numpy()
                    ).reshape(-1, 4),
                    axis=1,
                ),
                color=use_list_of_colors[i],
                ls="dotted",
            )
            tx1.loglog(
                np.average(self.B_sq_gr["k"].to_numpy().reshape(-1, 4), axis=1),
                np.average(
                    np.abs(
                        (
                            self.B_sq_gr[A_s].to_numpy()
                            - self.B_sq_newton[A_s].to_numpy()
                        )
                        / self.B_sq_newton[A_s].to_numpy()
                    ).reshape(-1, 4),
                    axis=1,
                ),
                color=use_list_of_colors[i],
                ls="dotted",
            )
            tx2.loglog(
                np.average(self.B_st_gr["k"].to_numpy().reshape(-1, 4), axis=1),
                np.average(
                    np.abs(
                        (
                            self.B_st_gr[A_s].to_numpy()
                            - self.B_st_newton[A_s].to_numpy()
                        )
                        / self.B_st_newton[A_s].to_numpy()
                    ).reshape(-1, 4),
                    axis=1,
                ),
                color=use_list_of_colors[i],
                ls="dotted",
            )

        # rescale the small panels
        for i in range(3):
            pos1 = ax[i].get_position()
            pos2 = [
                pos1.x0,
                pos1.y0,
                pos1.width,
                pos1.height * 0.8,
            ]
            ax[i].set_position(pos2)

        ax[0].set_title("Equilateral")
        ax[1].set_title("Squeezed")
        ax[2].set_title("Stretched")
        for ax, tx in zip(ax.flatten(), [tx0, tx1, tx2]):
            ax.legend()
            ax.set_xlabel(r"$k$ [h/Mpc]")
            ax.set_ylabel(r"$B(k1, \mu, t)$ [Mpc^6/h^6]")
            tx.set_ylabel(r"$\Delta B / B$")

        fig.suptitle(f"Seed {self.seed}, z={self.z}")
        plt.show()


if __name__ == "__main__":
    BV = BispectrumVariation_AS(0, 20)
    from IPython import embed

    embed()
