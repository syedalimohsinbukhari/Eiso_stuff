"""
Created on Tue Feb 22 09:52:10 2022
"""

import copy
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as si
from astropy.cosmology import FlatLambdaCDM

from spectral_models import SpectralModels


def get_parameter_details(model_name):
    model_name = model_name.lower()

    if model_name == 'pl':
        n_par, par_n = 3, ['a_pl', 'i1_pl', 'epiv_pl']
    elif model_name == 'pl_bb':
        n_par, par_n = 5, ['a_pl', 'i1_pl', 'epiv_pl', 'a_bb', 'kt_bb']
    elif model_name == 'sbpl':
        n_par, par_n = 6, ['a_sbpl', 'epiv_sbpl', 'i1_sbpl', 'bE', 'bS', 'i2_sbpl']
    elif model_name == 'sbpl_pl':
        n_par, par_n = 9, ['a_sbpl', 'epiv_sbpl', 'i1_sbpl', 'bE', 'bS', 'i2_sbpl',
                           'a_pl', 'i1_pl', 'epiv_pl']
    elif model_name == 'sbpl_bb':
        n_par, par_n = 8, ['a_sbpl', 'epiv_sbpl', 'i1_sbpl', 'bE', 'bS', 'i2_sbpl',
                           'a_bb', 'kt_bb']
    elif model_name == 'sbpl_pl_bb':
        n_par, par_n = 11, ['a_sbpl', 'epiv_sbpl', 'i1_sbpl', 'bE', 'bS', 'i2_sbpl',
                            'amp_pl', 'i1_pl', 'epiv_pl', 'a_bb', 'kt_bb']
    elif model_name == 'band':
        n_par, par_n = 4, ['a_band', 'ep_band', 'i1_band', 'i2_band']
    elif model_name == 'band_pl':
        n_par, par_n = 7, ['a_band', 'ep_band', 'i1_band', 'i2_band', 'a_pl', 'i1_pl',
                           'epiv_pl']
    elif model_name == 'band_bb':
        n_par, par_n = 6, ['a_band', 'ep_band', 'i1_band', 'i2_band', 'a_bb', 'kt_bb']
    elif model_name == 'band_pl_bb':
        n_par, par_n = 9, ['a_band', 'ep_band', 'i1_band', 'i2_band', 'a_pl', 'i1_pl',
                           'epiv_pl', 'amp_bb', 'kt_bb']
    elif model_name == 'cpl':
        n_par, par_n = 4, ['a_cpl', 'ep_cpl', 'i1_cpl', 'epiv_cpl']
    elif model_name == 'cpl_pl':
        n_par, par_n = 7, ['a_cpl', 'ep_cpl', 'i1_cpl', 'epiv_cpl', 'a_pl', 'i1_pl',
                           'epiv_pl']
    elif model_name == 'cpl_bb':
        n_par, par_n = 6, ['a_cpl', 'ep_cpl', 'i1_cpl', 'epiv_cpl', 'a_bb', 'kt_bb']
    elif model_name == 'cpl_pl_bb':
        n_par, par_n = 9, ['a_cpl', 'ep_cpl', 'i1_cpl', 'epiv_cpl', 'a_pl', 'i1_pl',
                           'epiv_pl', 'amp_bb', 'kt_bb']
    else:
        n_par, par_n = 0, []

    return n_par, par_n


def get_parameter_values(fit_extension, n_parameters):
    pars_ = [fit_extension.data[f'PARAM{i}'][0][0] for i in range(n_parameters)]
    errs_ = [fit_extension.data[f'PARAM{i}'][0][1] for i in range(n_parameters)]

    covariance_matrix = fit_extension.data['COVARMAT'][0][0:n_parameters, 0:n_parameters]

    return pars_, errs_, covariance_matrix


def get_median_values(mvd_array, sigma):
    _median = np.median(mvd_array)

    hi_mask = mvd_array >= _median
    lo_mask = mvd_array < _median

    hi = np.sort(mvd_array[hi_mask])
    lo = np.sort(mvd_array[lo_mask])

    _norm_hi = hi[np.int64(sigma * np.size(hi))]
    _norm_lo = lo[np.int64((1 - sigma) * np.size(lo))]

    return _median, _norm_hi, _norm_lo


def x_labels(axis_thing, parameter_name):
    if parameter_name in ['a_sbpl', 'a_band', 'a_pl', 'a_cpl']:
        out = 'Amplitude\n[ph/cm' + r'$^2$' + '/s]'
    if parameter_name in ['i1_sbpl', 'i1_band', 'i1_cpl']:
        out = r'$\alpha$'
    elif parameter_name in ['i2_sbpl', 'i2_band']:
        out = r'$\beta$'
    elif parameter_name == 'bE':
        out = 'Break energy\n[keV]'
    elif parameter_name == 'e_i_peak':
        out = r'E$_\mathrm{i,\ peak}$' + '\n[keV]'
    elif parameter_name == 'e_flux':
        out = 'Flux\n[erg/cm' + r'$^2$' + '/s]'
    elif parameter_name == 'e_flnc':
        out = 'Fluence\n[erg/cm' + r'$^2$]'
    elif parameter_name == 'e_isotropic':
        out = r'E$_\mathrm{iso}$' + '\n[erg]'
    elif parameter_name in ['ep_band', 'ep_cpl']:
        out = r'E$_\mathrm{peak}$ [keV]'
    else:
        out = ''

    axis_thing.set_xlabel(out)


def parameter_plotting(mvd, number_of_parameters, parameter_names, n_bins, sigma):
    f, ax = plt.subplots(1, number_of_parameters, figsize=(20, 5), sharey='row')

    if 'e_isotropic' not in parameter_names:
        color = ['tab:blue'] * len(parameter_names)
    else:
        color = ['lime', 'lightcoral', 'gold', 'gold']

    p, m, nl, nh = [], [], [], []

    for number, parameter in zip(range(number_of_parameters), parameter_names):
        ax[number].hist(mvd[parameter], bins=n_bins, color=color[number])  # , alpha=0.5)
        _median, _norm_hi, _norm_lo = get_median_values(mvd[parameter], sigma)
        [ax[number].axvline(value, color='k', ls=line_style) for value, line_style in
         zip([_median, _norm_lo, _norm_hi], ['-', '-.', ':'])]
        if parameter not in ['e_isotropic', 'e_flux', 'e_flnc']:
            ax[number].plot([], [], label=r'%.4f$_{-%.3E}^{+%.3E}$' % (
                _median, _median - _norm_lo, _norm_hi - _median), color='w')
        else:
            ax[number].plot([], [], label=r'%.4E$_{-%.3E}^{+%.3E}$' % (
                _median, _median - _norm_lo, _norm_hi - _median), color='w')
        if number == 0:
            ax[number].set_ylabel('Counts')

        x_labels(ax[number], parameter_name=parameter)

        ax[number].legend(loc='best', frameon=False)
        if number > 0:
            # taken from https://www.geeksforgeeks.org/how-to-remove-ticks-from
            # -matplotlib-plots/
            ax[number].tick_params(left=False)
        p.append(parameter)
        m.append(_median)
        nl.append(_median - _norm_lo)
        nh.append(_norm_hi - _median)

    plt.tight_layout()

    return p, m, nl, nh


def break_energy_to_peak_energy__sbpl(i1_sbpl, break_energy, break_scale, i2_sbpl):
    f = 0.5 * break_scale * (np.log(2 + i1_sbpl) - np.log(-2 - i2_sbpl))
    return break_energy * 10**f


class IsotropicEnergy:
    def __init__(self, model_name, multivariate_dictionary, sigma, n_iterations, t_start,
                 t_stop, redshift=0, e_low=8, e_high=int(1e7), h0=67.4, omega_m=0.315):
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

    def __luminosity_distance(self, int_z):
        return FlatLambdaCDM(H0=self.h0, Om0=self.omegaM).luminosity_distance(
            int_z).cgs.value

    def __luminosity_integral(self):
        return si.quad(self.__luminosity_distance, 0, self.redshift)[0]

    def __energy_range(self):
        return np.logspace(np.log10(self.e_low), np.log10(self.e_high),
                           int(self.n_iterations))

    def __generic(self, pars, _type='bolometric'):
        e_range = self.__energy_range()
        return SpectralModels(e_range, pars[:-1], self.t_start, self.t_stop,
                              self.model_name, _type).get_values()

    @staticmethod
    def __generic_mp(func, n_proc, chunks):
        return np.array([Pool(n_proc).map(func, chunk) for chunk in chunks]).flatten()

    def get_flux(self, pars):
        return self.__generic(pars, 'integrate')

    def get_fluence(self, pars):
        return self.__generic(pars, 'integrate') * self.__duration()

    def isotropic_energy(self, pars):
        _fluence = self.__generic(pars, _type='bolometric')

        _constant = (4 * np.pi * pars[-1]**2) * (1 + self.redshift)**-1

        return _fluence * self.__duration() * _constant

    def get_value_error_pairs(self, output_type='isotropic', use_multiprocessing=False,
                              n_proc=2):
        mvd = self.multivariate_dictionary
        n_proc = 1 if not use_multiprocessing else n_proc

        if output_type == 'isotropic':
            mvd['_dl'] = np.repeat(self.__luminosity_integral(), self.n_iterations)

        mvd = np.array(list(mvd.values())).T

        _chunks = tuple(self.__chunks(mvd, self.n_iterations // n_proc))

        if output_type == 'isotropic':
            out = self.__generic_mp(func=self.isotropic_energy, n_proc=n_proc,
                                    chunks=_chunks)
        elif output_type == 'fluence':
            out = self.__generic_mp(func=self.get_fluence, n_proc=n_proc, chunks=_chunks)
        else:
            out = self.__generic_mp(func=self.get_flux, n_proc=n_proc, chunks=_chunks)

        return out


def save_val_err(arr1, arr2, path, model_name, suffix):
    t1 = pd.DataFrame(arr1[1:])
    t2 = pd.DataFrame(arr2[1:])
    t1.columns = arr1[0]
    t2.columns = arr2[0]

    t3 = pd.concat([t1, t2], axis=1)

    t3.to_csv(f'{path}/{model_name.lower()}__parameter_value_err{suffix}.csv',
              index=False)


def make_dataframe(data_frame, m_name):
    wo_peak = ['e_flux', 'e_flnc', 'e_isotropic']
    w_peak = ['e_flux', 'e_flnc', 'e_i_peak', 'e_isotropic']

    pl = ['a_pl', 'i1_pl']
    bb = ['a_bb', 'kt_bb']
    sbpl = ['a_sbpl', 'i1_sbpl', 'bE', 'i2_sbpl']
    band = ['a_band', 'ep_band', 'i1_band', 'i2_band']
    cpl = ['a_cpl', 'i1_cpl', 'ep_cpl']

    if m_name == 'pl':
        pl += wo_peak
        df_ = data_frame[pl]
    elif m_name == 'pl_bb':
        pl += bb
        pl += wo_peak
        df_ = data_frame[pl]
    elif m_name == 'sbpl':
        sbpl += w_peak
        df_ = data_frame[sbpl]
    elif m_name == 'sbpl_pl':
        pl += sbpl
        pl += w_peak
        df_ = data_frame[pl]
    elif m_name == 'sbpl_bb':
        sbpl += bb
        sbpl += w_peak
        df_ = data_frame[sbpl]
    elif m_name == 'sbpl_pl_bb':
        pl += sbpl
        pl += bb
        pl += w_peak
        df_ = data_frame[pl]
    elif m_name == 'cpl':
        cpl += w_peak
        df_ = data_frame[cpl]
    elif m_name == 'cpl_pl':
        pl += cpl
        pl += w_peak
        df_ = data_frame[pl]
    elif m_name == 'cpl_bb':
        cpl += bb
        cpl += w_peak
        df_ = data_frame[cpl]
    elif m_name == 'cpl_pl_bb':
        pl += cpl
        pl += bb
        pl += w_peak
        df_ = data_frame[pl]
    elif m_name == 'band':
        band += w_peak
        df_ = data_frame[band]
    elif m_name == 'band_pl':
        pl += band
        pl += w_peak
        df_ = data_frame[pl]
    elif m_name == 'band_bb':
        band += bb
        band += w_peak
        df_ = data_frame[band]
    else:
        pl += band
        pl += bb
        pl += w_peak
        df_ = data_frame[pl]

    return df_
