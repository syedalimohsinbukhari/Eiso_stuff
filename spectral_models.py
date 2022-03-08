"""
Created on Mon Feb 21 15:24:34 2022
"""

import numpy as np
import scipy.integrate as si
from astropy.cosmology import FlatLambdaCDM


class SpectralModels:

    def __init__(self, energy, pars, t_start, t_stop, model: str = 'pl', model_type: str = 'counts', flux_energy=None, isotropic_energy=None, redshift=None, h0=67.4,
                 omega_m=0.315):
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
        return FlatLambdaCDM(H0=self.h0, Om0=self.omega_m).luminosity_distance(z_int).cgs.value

    def __luminosity_integral(self):
        return si.quad(self.__luminosity_distance, 0, self.z)[0]

    def __duration(self):
        return self.t_stop - self.t_start

    def __raise_error(self):
        types_ = ['counts', 'energy', 'integrate', 'bolometric']
        if self.model_type not in types_:
            raise TypeError('Model type must be in ' + ', '.join(types_[:-1]) + f' or {types_[-1]}.')

    def __time_difference(self):
        return self.t_stop - self.t_start

    def __vals(self):
        return self.energy, self.pars, self.model_type, self.f_ene, self.f_iso, self.z, self.keVtoErg

    def powerlaw(self, joint_pars=None):
        energy, pars, mt, f_ene, f_iso, z, kev_ = self.__vals()

        pars = pars if joint_pars is None else joint_pars

        def f_pl(_e=energy, out_ene=True):
            [amp, ind, e_piv] = pars
            out = amp * (_e * e_piv**-1)**ind

            return _e * out if out_ene else out

        # def ene_pl():
        #     return energy * f_pl()

        def int_pl(_range=f_ene):
            return si.quad(f_pl, _range[0], _range[1])[0] * kev_

        def bol_pl():
            _bol_range = [i / (1 + z) for i in f_iso]
            return int_pl(_range=_bol_range) * self.__time_difference()

        return f_pl(out_ene=False) if mt == 'counts' else f_pl() if mt == 'energy' else int_pl() if mt == 'integrate' else bol_pl()

    def blackbody(self, joint_pars=None):
        energy, pars, mt, f_ene, f_iso, z, kev_ = self.__vals()

        pars = pars if joint_pars is None else joint_pars

        def f_bb(_e=energy, out_ene=True):
            [amp, kt] = pars
            out = amp * _e**2 * (np.exp(_e / kt) - 1)**-1
            return _e * out if out_ene else out

        # def ene_bb():
        #     return energy * f_bb()

        def int_bb(_range=f_ene):
            return si.quad(f_bb, _range[0], _range[1])[0] * kev_

        def bol_bb():
            _bol_range = [i / (1 + z) for i in f_iso]
            return int_bb(_bol_range) * self.__time_difference()

        return f_bb(out_ene=False) if mt == 'counts' else f_bb() if mt == 'energy' else int_bb() if mt == 'integrate' else bol_bb()

    def cutoff_powerlaw(self, joint_pars=None):
        energy, pars, mt, f_ene, f_iso, z, kev_ = self.__vals()

        pars = pars if joint_pars is None else joint_pars

        def f_cpl(_e=energy, out_ene=True):
            [amp, e_peak, i1, e_piv] = pars
            out = amp * np.exp(-1 * _e * (2 + i1) * e_peak**-1) * (_e * e_piv**-1)**i1

            return _e * out if out_ene else out

        # def ene_cpl():
        #     return energy * f_cpl()

        def int_cpl(_range=f_ene):
            return si.quad(f_cpl, _range[0], _range[1])[0] * kev_

        def bol_cpl():
            _bol_range = [i / (1 + z) for i in f_iso]
            return int_cpl(_bol_range) * self.__time_difference()

        return f_cpl(out_ene=False) if mt == 'counts' else f_cpl() if mt == 'energy' else int_cpl() if mt == 'integrate' else bol_cpl()

    def smoothly_broken_powerlaw(self, joint_pars=None):
        energy, pars, mt, f_ene, f_iso, z, kev_ = self.__vals()

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

            out = amp * (_e * e_piv**-1)**b * 10**(beta - beta_piv)

            return _e * out if out_ene else out

        # def ene_sbpl(_ene):
        #     return _ene * f_sbpl()

        def int_sbpl(_range=f_ene):
            return si.quad(f_sbpl, _range[0], _range[1])[0] * kev_

        def bol_sbpl():
            _bol_range = [i / (1 + z) for i in f_iso]
            return int_sbpl(_bol_range) * self.__time_difference()

        return f_sbpl(out_ene=False) if mt == 'counts' else f_sbpl() if mt == 'energy' else int_sbpl() if mt == 'integrate' else bol_sbpl()

    def band_grb_function(self, joint_pars=None):
        energy, pars, mt, f_ene, f_iso, z, kev_ = self.__vals()

        pars = pars if joint_pars is None else joint_pars

        def f_band(_e=energy, out_ene=True):
            [amp, e_peak, i1, i2] = pars
            cond = (i1 - i2) * (2 + i1)**-1 * e_peak

            out = [amp * (_e * 0.01)**i2 * np.exp(i2 - i1) * ((i1 - i2) * e_peak * 0.01 * (i1 + 2)**-1)**(i1 - i2) if _e > cond else
                   amp * (_e * 0.01)**i1 * np.exp(-(2 + i1) * _e * e_peak**-1)][0]

            return _e * out if out_ene else out

        # def ene_band():
        #     return energy * f_band()

        def int_band(_range=f_ene):
            return si.quad(f_band, _range[0], _range[1])[0] * kev_

        def bol_band():
            _bol_range = [i / (1 + z) for i in f_iso]
            return int_band(_bol_range) * self.__time_difference()

        return f_band(out_ene=False) if mt == 'counts' else f_band() if mt == 'energy' else int_band() if mt == 'integrate' else bol_band()

    def pl_bb(self, joint_pars=None):
        pars = self.pars if joint_pars is None else joint_pars

        pl_, bb_ = self.powerlaw(pars[0:4]), self.blackbody(pars[4:])

        return pl_, bb_, pl_ + bb_

    def cpl_pl(self):
        cpl_, pl_ = self.cutoff_powerlaw(self.pars[0:5]), self.powerlaw(self.pars[5:])

        return cpl_, pl_, cpl_ + pl_

    def cpl_bb(self):
        cpl_, bb_ = self.cutoff_powerlaw(self.pars[0:5]), self.blackbody(self.pars[5:])

        return cpl_, bb_, cpl_ + bb_

    def cpl_pl_bb(self):
        cpl_ = self.energy(self.pars[0:5])
        pl_, bb_, _ = self.pl_bb(self.pars[5:])

        return cpl_, pl_, bb_, cpl_ + pl_ + bb_

    def band_pl(self):
        band_, pl_ = self.band_grb_function(self.pars[0:5]), self.powerlaw(self.pars[5:])

        return band_, pl_, band_ + pl_

    def band_bb(self):
        band_, bb_ = self.band_grb_function(self.pars[0:5]), self.blackbody(self.pars[5:])

        return band_, bb_, band_ + bb_

    def band_pl_bb(self):
        band_ = self.band_grb_function(self.pars[0:5])
        pl_, bb_, _ = self.pl_bb(self.pars[5:])

        return band_, pl_, bb_, band_ + pl_ + bb_

    def sbpl_pl(self):
        sbpl_, pl_ = self.smoothly_broken_powerlaw(self.pars[0:7]), self.powerlaw(self.pars[7:])

        return sbpl_, pl_, sbpl_ + pl_

    def sbpl_bb(self):
        sbpl_, bb_ = self.smoothly_broken_powerlaw(self.pars[0:7]), self.blackbody(self.pars[7:])

        return sbpl_, bb_, sbpl_ + bb_

    def sbpl_pl_bb(self):
        sbpl_ = self.smoothly_broken_powerlaw(self.pars[0:7])
        pl_, bb_, _ = self.pl_bb(self.pars[7:])

        return sbpl_, pl_, bb_, sbpl_ + pl_ + bb_

    def get_values(self):
        model = self.model

        return [self.powerlaw() if model == 'pl' else
                self.blackbody() if model == 'bb' else
                self.pl_bb() if model == 'pl_bb' else
                self.cutoff_powerlaw() if model == 'cpl' else
                self.cpl_pl() if model == 'cpl_pl' else
                self.cpl_bb() if model == 'cpl_bb' else
                self.cpl_pl_bb() if model == 'cpl_pl_bb' else
                self.band_grb_function() if model == 'band' else
                self.band_pl() if model == 'band_pl' else
                self.band_bb() if model == 'band_bb' else
                self.band_pl_bb() if model == 'band_pl_bb' else
                self.smoothly_broken_powerlaw() if model == 'sbpl' else
                self.sbpl_pl() if model == 'sbpl_pl' else
                self.sbpl_bb() if model == 'sbpl_bb' else self.sbpl_pl_bb()][0]
