"""
Created on Mon Feb 21 15:24:34 2022
"""

import numpy as np
import scipy.integrate as si
from astropy.cosmology import FlatLambdaCDM

"""
RMFIT order of models

PL:         PL
PL_BB:      PL + BB
SBPL:       SBPL
SBPL_PL:    PL + SBPL
SBPL_BB:    SBPL + BB
SBPL_PL_BB: PL + SBPL + BB
CPL:        CPL
CPL_PL:     PL + CPL
CPL_BB:     CPL + BB
CPL_PL_BB:  PL + CPL + BB
BAND:       BAND
BAND_PL:    PL + BAND
BAND_BB:    BAND + BB
BAND_PL_BB: PL + BAND + BB

"""


class SpectralModels:

    def __init__(self, energy, pars, t_start, t_stop, model: str = 'pl', model_type: str = 'counts', flux_energy=None,
                 isotropic_energy=None, redshift=None, h0=67.4, omega_m=0.315):

        self.energy = energy
        self.pars = pars

        self.t_start = t_start
        self.t_stop = t_stop

        self.model = model
        self.model_type = model_type

        self.f_ene = [10, int(1e7)] if flux_energy is None else flux_energy
        self.f_iso = [1, int(1e4)] if isotropic_energy is None else isotropic_energy

        self.z = 0 if redshift is None else redshift

        self.h0 = h0
        self.omega_m = omega_m
        self.keVtoErg = 1.6021766208e-09

        self.__raise_error()

    def __luminosity_distance(self, z_int):
        return FlatLambdaCDM(H0=self.h0, Om0=self.omega_m).luminosity_distance(z=z_int).cgs.value

    def __luminosity_integral(self):
        return si.quad(self.__luminosity_distance, 0, self.z)[0]

    def __raise_error(self):
        types_ = ['counts', 'energy', 'integrate', 'bolometric']
        if self.model_type not in types_:
            raise TypeError(f'Model type must be in {", ".join(types_[:-1])} or {types_[-1]}.')

    def __get_time_difference(self):
        return self.t_stop - self.t_start

    def __values(self):
        return self.energy, self.pars, self.model_type, self.f_ene, self.f_iso, self.z, self.keVtoErg

    def powerlaw(self, joint_pars=None):
        energy, pars, mt, f_ene, f_iso, z, kev_ = self.__values()

        pars = pars if joint_pars is None else joint_pars

        def f_pl(_e=energy, out_ene=True):
            [amp, e_piv, ind] = pars
            res = amp * (_e * e_piv**-1)**ind

            return _e * res if out_ene else res

        def int_pl(_range=f_ene):
            return si.quad(f_pl, _range[0], _range[1])[0] * kev_

        def bol_pl():
            _bol_range = [i / (1 + z) for i in f_iso]
            return int_pl(_range=_bol_range) * self.__get_time_difference()

        func_map = {'counts': f_pl(out_ene=False),
                    'energy': f_pl(),
                    'integrate': int_pl(),
                    'bolometric': bol_pl()}

        return func_map[mt]

    def blackbody(self, joint_pars=None):
        energy, pars, mt, f_ene, f_iso, z, kev_ = self.__values()

        pars = pars if joint_pars is None else joint_pars

        def f_bb(_e=energy, out_ene=True):
            [amp, kt] = pars
            res = amp * _e**2 * (np.exp(_e / kt) - 1)**-1
            return _e * res if out_ene else res

        def int_bb(_range=f_ene):
            return si.quad(f_bb, _range[0], _range[1])[0] * kev_

        def bol_bb():
            _bol_range = [i / (1 + z) for i in f_iso]
            return int_bb(_bol_range) * self.__get_time_difference()

        if mt == 'counts':
            out = f_bb(out_ene=False)
        elif mt == 'energy':
            out = f_bb()
        elif mt == 'integrate':
            out = int_bb()
        else:
            out = bol_bb()

        return out

    def cutoff_powerlaw(self, joint_pars=None):
        energy, pars, mt, f_ene, f_iso, z, kev_ = self.__values()

        pars = pars if joint_pars is None else joint_pars

        def f_cpl(_e=energy, out_ene=True):
            [amp, e_peak, i1, e_piv] = pars
            res = amp * np.exp(-1 * _e * (2 + i1) * e_peak**-1) * (_e * e_piv**-1)**i1

            return _e * res if out_ene else res

        def int_cpl(_range=f_ene):
            return si.quad(f_cpl, _range[0], _range[1])[0] * kev_

        def bol_cpl():
            _bol_range = [i / (1 + z) for i in f_iso]
            return int_cpl(_bol_range) * self.__get_time_difference()

        func_map = {'counts': f_cpl(out_ene=False),
                    'energy': f_cpl(),
                    'integrate': int_cpl(),
                    'bolometric': bol_cpl()}

        return func_map[mt]

    def smoothly_broken_powerlaw(self, joint_pars=None):
        energy, pars, mt, f_ene, f_iso, z, kev_ = self.__values()

        pars = pars if joint_pars is None else joint_pars

        def f_sbpl(_e=energy, out_ene=True):
            [amp, e_piv, i1, b_ene, b_scale, i2] = pars

            m = 0.5 * (i2 - i1)
            b = 0.5 * (i1 + i2)

            a_piv = np.log10(e_piv * b_ene**-1) * b_scale**-1

            beta_piv_ = np.exp(a_piv) + np.exp(-1 * a_piv)
            beta_piv = m * b_scale * np.log(0.5 * beta_piv_)

            a = b_scale**-1 * np.log10(_e * b_ene**-1)

            beta_ = np.exp(a) + np.exp(-1 * a)
            beta = m * b_scale * np.log(0.5 * beta_)

            res = amp * (_e * e_piv**-1)**b * 10**(beta - beta_piv)

            return _e * res if out_ene else res

        def int_sbpl(_range=f_ene):
            return si.quad(f_sbpl, _range[0], _range[1])[0] * kev_

        def bol_sbpl():
            _bol_range = [i / (1 + z) for i in f_iso]
            return int_sbpl(_bol_range) * self.__get_time_difference()

        func_map = {'counts': f_sbpl(out_ene=False),
                    'energy': f_sbpl(),
                    'integrate': int_sbpl(),
                    'bolometric': bol_sbpl()}

        return func_map[mt]

    def band_grb_function(self, joint_pars=None):
        energy, pars, mt, f_ene, f_iso, z, kev_ = self.__values()

        pars = pars if joint_pars is None else joint_pars

        def f_band(_e=energy, out_ene=True):
            [amp, e_peak, i1, i2] = pars
            cond = (i1 - i2) * (2 + i1)**-1 * e_peak

            if _e > cond:
                res = amp * (_e * 0.01)**i2 * np.exp(i2 - i1) * ((i1 - i2) * e_peak * 0.01 * (i1 + 2)**-1)**(i1 - i2)
            else:
                res = amp * (_e * 0.01)**i1 * np.exp(-(2 + i1) * _e * e_peak**-1)

            return _e * res if out_ene else res

        def int_band(_range=f_ene):
            return si.quad(f_band, _range[0], _range[1])[0] * kev_

        def bol_band():
            _bol_range = [i / (1 + z) for i in f_iso]
            return int_band(_bol_range) * self.__get_time_difference()

        func_map = {'counts': f_band(out_ene=False),
                    'energy': f_band(),
                    'integrate': int_band(),
                    'bolometric': bol_band()}

        return func_map[mt]

    def pl_bb(self):
        pl_ = self.powerlaw(self.pars[:3])
        bb_ = self.blackbody(self.pars[3:])

        return pl_, bb_, np.ufunc.reduce(np.add, [pl_, bb_])

    def cpl_pl(self):
        pl_, cpl_ = self.powerlaw(self.pars[:3]), self.cutoff_powerlaw(self.pars[3:])

        return cpl_, pl_, np.ufunc.reduce(np.add, [cpl_, pl_])

    def cpl_bb(self):
        cpl_, bb_ = self.cutoff_powerlaw(self.pars[:-2]), self.blackbody(self.pars[-2:])

        return cpl_, bb_, np.ufunc.reduce(np.add, [cpl_, bb_])

    def cpl_pl_bb(self):
        pl_ = self.powerlaw(self.pars[:3])
        cpl_ = self.cutoff_powerlaw(self.pars[3:-2])
        bb_ = self.blackbody(self.pars[-2:])

        return cpl_, pl_, bb_, np.ufunc.reduce(np.add, [cpl_, pl_, bb_])

    def band_pl(self):
        pl_, band_ = self.powerlaw(self.pars[:3]), self.band_grb_function(self.pars[3:])

        return band_, pl_, np.ufunc.reduce(np.add, [band_, pl_])

    def band_bb(self):
        band_ = self.band_grb_function(self.pars[:-2])
        bb_ = self.blackbody(self.pars[-2:])

        return band_, bb_, np.ufunc.reduce(np.add, [band_, bb_])

    def band_pl_bb(self):
        pl_ = self.powerlaw(self.pars[:3])
        band_ = self.band_grb_function(self.pars[3:-2])
        bb_ = self.blackbody(self.pars[-2:])

        return band_, pl_, bb_, np.ufunc.reduce(np.add, [band_, pl_, bb_])

    def sbpl_pl(self):
        pl_ = self.powerlaw(self.pars[:3])
        sbpl_ = self.smoothly_broken_powerlaw(self.pars[3:])

        return pl_, sbpl_, np.ufunc.reduce(np.add, [sbpl_, pl_])

    def sbpl_bb(self):
        sbpl_ = self.smoothly_broken_powerlaw(self.pars[:-2])
        bb_ = self.blackbody(self.pars[-2:])

        return sbpl_, bb_, np.ufunc.reduce(np.add, [sbpl_, bb_])

    def sbpl_pl_bb(self):
        pl_ = self.powerlaw(self.pars[:3])
        sbpl_ = self.smoothly_broken_powerlaw(self.pars[3:-2])
        bb_ = self.blackbody(self.pars[-2:])

        return sbpl_, pl_, bb_, np.ufunc.reduce(np.add, [sbpl_, pl_, bb_])

    def get_values(self):
        model_functions = {'pl': self.powerlaw,
                           'bb': self.blackbody,
                           'pl_bb': self.pl_bb,
                           'cpl': self.cutoff_powerlaw,
                           'cpl_pl': self.cpl_pl,
                           'cpl_bb': self.cpl_bb,
                           'cpl_pl_bb': self.cpl_pl_bb,
                           'band': self.band_grb_function,
                           'band_pl': self.band_pl,
                           'band_bb': self.band_bb,
                           'band_pl_bb': self.band_pl_bb,
                           'sbpl': self.smoothly_broken_powerlaw,
                           'sbpl_pl': self.sbpl_pl,
                           'sbpl_bb': self.sbpl_bb,
                           'sbpl_pl_bb': self.sbpl_pl_bb}

        model = self.model
        return model_functions[model]()
