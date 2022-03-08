"""
Created on Tue Feb 22 09:52:10 2022
"""
import copy
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from spectral_models import SpectralModels


def get_parameter_details(model_name):
    if model_name.lower() == 'pl':
        n_par, par_n = 3, ['a_pl', 'i1_pl', 'epiv_pl']
    elif model_name.lower() == 'pl_bb':
        n_par, par_n = 5, ['a_pl', 'i1_pl', 'epiv_pl', 'a_bb', 'kt_bb']
    elif model_name.lower() == 'sbpl':
        n_par, par_n = 6, ['a_sbpl', 'epiv_sbpl', 'i1_sbpl', 'bE', 'bS', 'i2_sbpl']
    elif model_name.lower() == 'sbpl_pl':
        n_par, par_n = 9, ['a_sbpl', 'epiv_sbpl', 'i1_sbpl', 'bE', 'bS', 'i2_sbpl', 'a_pl', 'i1_pl', 'epiv_pl']
    elif model_name.lower() == 'sbpl_bb':
        n_par, par_n = 8, ['a_sbpl', 'epiv_sbpl', 'i1_sbpl', 'bE', 'bS', 'i2_sbpl', 'a_bb', 'kt_bb']
    elif model_name.lower() == 'sbpl_pl_bb':
        n_par, par_n = 11, ['a_sbpl', 'epiv_sbpl', 'i1_sbpl', 'bE', 'bS', 'i2_sbpl', 'amp_pl', 'i1_pl', 'epiv_pl', 'a_bb', 'kt_bb']
    elif model_name.lower() == 'band':
        n_par, par_n = 4, ['a_band', 'ep_band', 'i1_band', 'i2_band']
    elif model_name.lower() == 'band_pl':
        n_par, par_n = 7, ['a_band', 'ep_band', 'i1_band', 'i2_band', 'a_pl', 'i1_pl', 'epiv_pl']
    elif model_name.lower() == 'band_bb':
        n_par, par_n = 6, ['a_band', 'ep_band', 'i1_band', 'i2_band', 'a_bb', 'kt_bb']
    elif model_name.lower() == 'band_pl_bb':
        n_par, par_n = 9, ['a_band', 'ep_band', 'i1_band', 'i2_band', 'a_pl', 'i1_pl', 'epiv_pl', 'amp_bb', 'kt_bb']
    elif model_name.lower() == 'cpl':
        n_par, par_n = 4, ['a_cpl', 'ep_cpl', 'i1_cpl', 'epiv_cpl']
    elif model_name.lower() == 'cpl_pl':
        n_par, par_n = 7, ['a_cpl', 'ep_cpl', 'i1_cpl', 'epiv_cpl', 'a_pl', 'i1_pl', 'epiv_pl']
    elif model_name.lower() == 'cpl_bb':
        n_par, par_n = 6, ['a_cpl', 'ep_cpl', 'i1_cpl', 'epiv_cpl', 'a_bb', 'kt_bb']
    elif model_name.lower() == 'cpl_pl_bb':
        n_par, par_n = 9, ['a_cpl', 'ep_cpl', 'i1_cpl', 'epiv_cpl', 'a_pl', 'i1_pl', 'epiv_pl', 'amp_bb', 'kt_bb']
    else:
        n_par, par_n = 0, []

    return n_par, par_n


def get_parameter_values(fit_extension, number_of_parameters):
    pars_ = [fit_extension.data[f'PARAM{i}'][0][0] for i in range(number_of_parameters)]
    errs_ = [fit_extension.data[f'PARAM{i}'][0][1] for i in range(number_of_parameters)]

    covariance_matrix = fit_extension.data['COVARMAT'][0][0:number_of_parameters, 0:number_of_parameters]

    return pars_, errs_, covariance_matrix


def get_median_values(mvd_array, sigma):
    _median = np.median(mvd_array)
    hi_mask, lo_mask = mvd_array >= _median, mvd_array < _median
    hi, lo = np.sort(mvd_array[hi_mask]), np.sort(mvd_array[lo_mask])
    _norm_hi, _norm_lo = hi[np.int64(sigma * np.size(hi))], lo[np.int64((1 - sigma) * np.size(lo))]
    return _median, _norm_hi, _norm_lo


def parameter_plotting(mvd, number_of_parameters, parameter_names, n_bins, sigma):
    _f = (10, 5) if 'e_isotropic' in parameter_names else (20, 5)

    f, ax = plt.subplots(1, number_of_parameters, figsize=_f, sharey='row')
    for number, parameter in zip(range(number_of_parameters), parameter_names):
        ax[number].hist(mvd[parameter], bins=n_bins, color='g')
        _median, _norm_hi, _norm_lo = get_median_values(mvd[parameter], sigma)
        [ax[number].axvline(value, color='k', ls=line_style) for value, line_style in zip([_median, _norm_lo, _norm_hi], ['-', '-.', ':'])]
        if not parameter == 'e_isotropic':
            ax[number].plot([], [], label=r'%.4f$_{-%.3E}^{+%.3E}$' % (_median, _median - _norm_lo, _norm_hi - _median), color='w')
        else:
            ax[number].plot([], [], label=r'%.4E$_{-%.3E}^{+%.3E}$' % (_median, _median - _norm_lo, _norm_hi - _median), color='w')
        if number == 0:
            ax[number].set_ylabel('Counts')
        ax[number].set_xlabel(f'{parameter}')
        ax[number].legend(loc='best', frameon=False)
        if number > 0:
            # taken from https://www.geeksforgeeks.org/how-to-remove-ticks-from-matplotlib-plots/
            ax[number].tick_params(left=False)
    plt.tight_layout()


def break_energy_to_peak_energy__sbpl(i1_sbpl, break_energy, break_scale, i2_sbpl):
    f = 0.5 * break_scale * (np.log(2 + i1_sbpl) - np.log(-2 - i2_sbpl))
    return break_energy * 10**f


class IsotropicEnergy:
    def __init__(self, model_name, multivariate_dictionary, sigma, n_iterations, t_start, t_stop, redshift=0, e_low=8, e_high=int(1e7), h0=67.4, omega_m=0.315):
        self.model_name = model_name.lower()
        self.multivariate_dictionary = copy.deepcopy(multivariate_dictionary)
        self.sigma = sigma
        self.n_iterations = n_iterations
        self.e_low = e_low
        self.e_high = e_high
        self.t_start = t_start
        self.t_stop = t_stop
        self.redshift = redshift
        self.h0 = h0
        self.omegaM = omega_m

    @staticmethod
    def __chunks(lst, n):
        # taken from https://stackoverflow.com/a/312464/3212945
        """Yield successive n-sized chunks from lst."""
        for iterator in range(0, len(lst), n):
            yield lst[iterator:iterator + n]

    def __duration(self):
        return self.t_stop - self.t_start

    def __luminosity_integral(self):
        return FlatLambdaCDM(H0=self.h0, Om0=self.omegaM).luminosity_distance(self.redshift).cgs.value

    def isotropic_energy(self, mvd_with_dl):
        e_range = np.logspace(np.log10(self.e_low), np.log10(self.e_high), int(self.n_iterations))

        _fluence = SpectralModels(e_range, mvd_with_dl[:-1], self.t_start, self.t_stop, self.model_name, 'bolometric').get_values()

        _constant = (4 * np.pi * mvd_with_dl[-1]**2) * (1 + self.redshift)**-1 * 1.60217657e-9
        return _fluence * self.__duration() * _constant

    def isotropic_energy__mp(self, luminosity_distance, n_proc):
        mvd = self.multivariate_dictionary

        mvd['_dl'] = np.repeat(luminosity_distance, self.n_iterations)

        mvd = np.array(list(mvd.values())).T

        _chunks = tuple(self.__chunks(mvd, self.n_iterations // n_proc))

        return [Pool(n_proc).map(self.isotropic_energy, chunk) for chunk in _chunks]

    def get_value_error_pairs(self, n_proc=2):

        luminosity_distance = self.__luminosity_integral()

        e_isotropic = self.isotropic_energy__mp(luminosity_distance, n_proc=n_proc)

        _values = np.array(e_isotropic)

        values = np.array([value[:, 0] for value in _values]).flatten()
        errors = np.array([error[:, 1] for error in _values]).flatten()

        return values, errors
