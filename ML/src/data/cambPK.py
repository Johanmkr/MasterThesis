import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower

# Testin
from IPython import embed


# pars = camb.CAMBparams()
# pars.set_cosmology(H0=67.556, ombh2=0.022032, omch2=0.12038, TCMB=2.7255, standard_neutrino_neff=3.046)
# pars.InitPower.set_params(As=2.215e-9, ns=0.9619, r=0)
# pars.set_for_lmax(2500, lens_potential_accuracy=0)

# # results = camb.get_results(pars)

# # powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
# # for name in powers: print(name)

# # totCL = powers['total']
# # unlensedCL = powers['unlensed_scalar']
# # lensedCL = powers['lensed_scalar']
# # ls = np.arange(totCL.shape[0])

# # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# # ax.plot(ls, totCL[:, 0], 'k-', label='total')
# # ax.set_xlim([2, 2500])
# # plt.show()

# pars.set_matter_power(redshifts=[0., 0.8, 1.0], kmax=2.0)


# # # Linear spectra
# # pars.NxonLinear = model.NonLinear_none
# results = camb.get_results(pars)
# # kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
# pars.NonLinear = model.NonLinear_both
# results.calc_power_spectra(pars)
# kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)

# for i, line in enumerate(['-','--',':']):
#     # plt.loglog(kh, pk[i,:], color='k', ls = line)
#     plt.loglog(kh_nonlin, pk_nonlin[i,:], color='r', ls=line)
# plt.xlabel('k/h Mpc');
# plt.legend(['z=0','z=0.8','z=1.0'], loc='lower left');
# plt.title('Matter power at z=0 and z=0.8 from CAMB');
# plt.show()


class CambSpectra:
    def __init__(self) -> None:
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=67.556, ombh2=0.022032, omch2=0.12038, TCMB=2.7255, standard_neutrino_neff=3.046)
        self.pars.InitPower.set_params(As=2.215e-9, ns=0.9619, r=0)
        self.pars.set_for_lmax(2500, lens_potential_accuracy=0)
    
    def __call__(self, redshift:float=0.0):# -> tuple(np.ndarray, np.ndarray):
        self.pars.set_matter_power(redshifts=[redshift], kmax=2.0)
        results = camb.get_results(self.pars)
        self.pars.NonLinear = model.NonLinear_both
        results.calc_power_spectra(self.pars)
        kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
        return kh_nonlin, pk_nonlin[0,:]



if __name__=="__main__":
    CS = CambSpectra()
    redshift = 0.8
    vals = CS(redshift)
    # embed()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.loglog(*vals, 'g--', label='CAMB')
    # ax.set_xlim([2, 2500])
    ax.set_title(f"Matter power spectrum at z={redshift}")
    plt.show()
    # pass