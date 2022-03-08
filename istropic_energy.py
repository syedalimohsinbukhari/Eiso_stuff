"""
Created on Fri Feb 18 09:55:49 2022
"""

import math

import astropy.constants.codata2018 as constants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.random import multivariate_normal

import functions as fncs

model_name = 'BAND'

redshift = 1
c = constants.c.value

s, sigma, n_bins = int(1e4), math.erf(1 / np.sqrt(2)), 500

start, end = -0.128, 3.648
duration = end - start

model_file = fits.open(f'{model_name}.fit')

model_pars = model_file['FIT PARAMS']

n_par, par_n = fncs.get_parameter_details(model_name)

pars_, errs_, covariance_matrix = fncs.get_parameter_values(model_pars, n_par)

mvd_ = multivariate_normal(pars_, covariance_matrix, s)

np.save(f'parameters_{model_name}', mvd_)

multivariate_dictionary = {}
for i, v in enumerate(par_n):
    multivariate_dictionary[v] = mvd_[:, i]

_temp = fncs.IsotropicEnergy(model_name=model_name,
                             multivariate_dictionary=multivariate_dictionary,
                             sigma=sigma,
                             n_iterations=s,
                             e_low=8,
                             e_high=int(1e7),
                             t_start=start,
                             t_stop=end,
                             redshift=redshift)

eiso_val, _ = _temp.get_value_error_pairs()

if 'sbpl' in model_name.lower():
    a_sbpl = multivariate_dictionary['i1_sbpl']
    bE = multivariate_dictionary['bE']
    bS = multivariate_dictionary['bS']
    b_sbpl = multivariate_dictionary['i2_sbpl']

    multivariate_dictionary['ep_sbpl'] = fncs.break_energy_to_peak_energy__sbpl(a_sbpl, bE, bS, b_sbpl)

# add e_i_peak and e_iso to multivariate_dictionary
multivariate_dictionary['e_i_peak'] = multivariate_dictionary[f'ep_{model_name.lower()}'] * (1 + redshift)

multivariate_dictionary['e_isotropic'] = eiso_val

mv_df = pd.DataFrame(multivariate_dictionary)

if model_name.lower() == 'sbpl':
    mv_df_ = mv_df[['a_sbpl', 'i1_sbpl', 'bE', 'i2_sbpl', 'e_i_peak', 'e_isotropic']]
elif model_name.lower() == 'cpl':
    mv_df_ = mv_df[['a_cpl', 'i1_cpl', 'ep_cpl', 'e_i_peak', 'e_isotropic']]
else:
    mv_df_ = mv_df[['a_band', 'ep_band', 'i1_band', 'i2_band', 'e_i_peak', 'e_isotropic']]

par_n = list(mv_df_.keys())
mv_ = mv_df_.to_dict('series')

fncs.parameter_plotting(mv_, len(par_n[:-2]), par_n[:-2], n_bins, sigma)
plt.savefig(f'{model_name.lower()}_parameters__{start}_{end}__{redshift}.pdf')
plt.close()

fncs.parameter_plotting(mv_, len(par_n[-2:]), par_n[-2:], n_bins, sigma)
plt.savefig(f'{model_name.lower()}_isotropic_energy__{start}_{end}__{redshift}.pdf')
plt.close()

_save = pd.DataFrame(mv_)

np.save(f'{model_name.lower()}_isotropic_energy_details__{start}_{end}__{redshift}', _save.to_numpy())
