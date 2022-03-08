"""
Created on Mon Feb 21 15:24:34 2022
"""

import numpy as np


def pl_n(energy, pars):
    _e, [amp_pl, i1_pl, epiv_pl] = energy, pars
    return _e * amp_pl * (_e * epiv_pl**-1)**i1_pl


def bb_n(energy, pars):
    _e, [amp_bb, kt_bb] = energy, pars
    return _e * amp_bb * _e**2 * (np.exp(_e / kt_bb) - 1)**-1


def cpl_n(energy, pars):
    _e, [amp_cpl, ep_cpl, i1_cpl, epiv_cpl] = energy, pars

    return _e * amp_cpl * np.exp(-1 * _e * (2 + i1_cpl) * ep_cpl**-1) * (_e * epiv_cpl**-1)**i1_cpl


def sbpl_n(energy, pars):
    _e, [amp_sbpl, epiv_sbpl, i1_sbpl, bE, bS, i2_sbpl] = energy, pars

    m = 0.5 * (i2_sbpl - i1_sbpl)
    b = 0.5 * (i1_sbpl + i2_sbpl)

    a_piv = np.log10(epiv_sbpl * bE**-1) * bS**-1

    beta_piv_ = np.exp(a_piv) + np.exp(-1 * a_piv)
    beta_piv = m * bS * np.log(0.5 * beta_piv_)

    a = bS**-1 * np.log10(_e * bE**-1)

    beta_ = np.exp(a) + np.exp(-1 * a)
    beta = m * bS * np.log(0.5 * beta_)

    return _e * amp_sbpl * (_e * epiv_sbpl**-1)**b * 10**(beta - beta_piv)


def band_n(energy, pars):
    _e, [amp, ep, a, b] = energy, pars
    cond = (a - b) * (2 + a)**-1 * ep

    return _e * [amp * (_e * 0.01)**b * np.exp(b - a) * ((a - b) * ep * 0.01 * (a + 2)**-1)**(a - b) if _e > cond else
                 amp * (_e * 0.01)**a * np.exp(-(2 + a) * _e * ep**-1)][0]


def pl_bb_n(energy, pars):
    pl_ = pl_n(energy, pars[0:4])
    bb_ = bb_n(energy, pars[4:])

    return pl_, bb_, pl_ + bb_


def cpl_pl_n(energy, pars):
    cpl_ = cpl_n(energy, pars[0:5])
    pl_ = pl_n(energy, pars[5:])

    return cpl_, pl_, cpl_ + pl_


def cpl_bb_n(energy, pars):
    cpl_ = cpl_n(energy, pars[0:5])
    bb_ = bb_n(energy, pars[5:])

    return cpl_, bb_, cpl_ + bb_


def cpl_pl_bb_n(energy, pars):
    cpl_ = cpl_n(energy, pars[0:5])
    pl_, bb_, _ = pl_bb_n(energy, pars[5:])

    return cpl_, pl_, bb_, cpl_ + pl_ + bb_


def band_pl_n(energy, pars):
    band_ = band_n(energy, pars[0:5])
    pl_ = pl_n(energy, pars[5:])

    return band_, pl_, band_ + pl_


def band_bb_n(energy, pars):
    band_ = band_n(energy, pars[0:5])
    bb_ = bb_n(energy, pars[5:])

    return band_, bb_, band_ + bb_


def band_pl_bb_n(energy, pars):
    band_ = band_n(energy, pars[0:5])
    pl_, bb_, _ = pl_bb_n(energy, pars[5:])

    return band_, pl_, bb_, band_ + pl_ + bb_


def sbpl_pl_n(energy, pars):
    sbpl_ = sbpl_n(energy, pars[0:7])
    pl_ = pl_n(energy, pars[7:])

    return sbpl_, pl_, sbpl_ + pl_


def sbpl_bb_n(energy, pars):
    sbpl_ = sbpl_n(energy, pars[0:7])
    bb_ = bb_n(energy, pars[7:])

    return sbpl_, bb_, sbpl_ + bb_


def sbpl_pl_bb_n(energy, pars):
    sbpl_ = sbpl_n(energy, pars[0:7])
    pl_, bb_, _ = pl_bb_n(energy, pars[7:])

    return sbpl_, pl_, bb_, sbpl_ + pl_ + bb_
