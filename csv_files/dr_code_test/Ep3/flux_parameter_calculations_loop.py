import os
from astropy.io import fits
from pylab import *
import scipy.integrate as si

with open(f'{os.getcwd()}/function_definitions.py') as infile:
    exec(infile.read())

# List of models and their parameters in Rmfit
'''
For details look at the file photon_models.ps among rmfit files
SBPL = Smoothly Broken Power Law
6 parameters - 4 free - 2 fixed
1----Amplitude - A - vary photon/(s cm^2 keV)
2----Pivot E - Epiv - fix at 100 keV
3----Index1 - lambda1 - vary
4----Break E - Eb - vary keV
5----Break scale - Delta -  fix at 0.3 decades E
6----Index2 - lambda2 - vary

Band
4 parameters
1----Amplitude - A - vary photon/(s cm^2 keV)
2----Epeak - vary keV
3----alpha - vary
4----beta - vary

Compton
4 parameters - 3 free - 1 fixed
1----Amplitude - A - vary photon/(s cm^2 keV)
2----Epeak - vary keV
3----Index - vary
4----Pivot energy - Epiv - fix at 100 keV

Power Law
3 parameters - 2 free - 1 fixed
1----Amplitude - A - vary photon/(s cm^2 keV)
2----Pivot energy - Epiv - fix at 100 keV
3----Index - vary

Black Body
2 parameters
1----Amplitude - A - vary photon/(s cm^2 keV)
2----kT - vary (electron energy in keV)

Order in rmfit
PL SBPL Band Comptonized BlackBody

'''

# The fits file used  is the one obtained from rmfit by going to the Fit Display window then Fit Results -> Write results to file

# allmodel_names=np.asarray(['SBPL','BAND','CPL','PL','BB'])
# parnumbers=[6,4,4,3,2]

# Parameter names
# For SBPL
# parnames=['A_sbpl','Epiv_sbpl','Alpha','Ebreak','Break Scale','Beta']
# Use symbol definition from Ep.png (from Feraol). It is equivalent to the definition from photon_model.ps (rmfit).

# For Band
# parnames=['A_band','Epeak_band','alpha','beta']

# For Compton
# parnames=['A_comp','Epeak_comp','Index_comp','Epiv_comp']

# For Power Law
# parnames=['A_pl','Epiv_pl','Index_pl']

# For Black Body
# parnames=['A_bb','kT']


# =================================================================================
# ================================== INPUT ========================================
# =================================================================================

model_complete = 'CPL'
eac = False
nvect = 50000
sLim = 0.6827
Eintegral1 = 10.
Eintegral2 = 1.e7
Eiso1 = 1.
Eiso2 = 10000.
integration_steps = 1000.
Energy_flux_fluence = 10.

Tstart = 21.184
Tend = 24.32
Tint = Tend - Tstart

import sys

z = float(sys.argv[1])

h0 = 67.4 * 1e5 / 3.08568e24
omega_M = 0.315
omega_L = 1 - omega_M

keVtoErg = 1.60218e-9
c = 3e10

# =================================================================================
# ================================== MAIN ========================================
# =================================================================================

# FILE NAMES
# ==========
str_EAC = ''
if eac:
    str_EAC = '_EAC'
str_MODEL = model_complete + str_EAC
str_file = str_MODEL + '.fit'  # 020_GRB130427324 BAND_PL_BB_EAC.fit file

# str_id='z'+str('%6.4f'%z)+'_T'+str('%6.3f'%Tstart)+'_'+str('%6.3f'%Tend)+str(nvect)
str_id = 'z' + str('%6.4f' % z) + '_T' + str('%6.3f' % Tstart).strip() + '_' + str('%6.3f' % Tend).strip() + '_' + str(nvect)
filepar = str_MODEL + '_simulated_par_' + str_id + '.txt'
filepng_par = str_MODEL + '_simulated_par_' + str_id + '.png'
strfileout = str_MODEL + '_output_' + str_id + '.txt'

# open file
hdu = fits.open(str_file)
fileout = open(strfileout, "w+")
fileout.write('Tstart= ' + str(Tstart))
fileout.write('\nTend= ' + str(Tend))
fileout.write('\nRedshift= ' + str(z) + '\n\n')

# MODEL, COMPONENTS
# =================
model_components = np.array(model_complete.split('_'))
basemodel = model_components[0]
print(model_components, basemodel)
# Get array of models, array of parameter numbers, and array of parameter names in rmfit order.
models, Nparameters, parnames, parfix = get_model_details(model_complete)
indexmain = np.where(models == basemodel)[0][0]  # Find position of main model after reordering models according to rmfit order.
Nparameters_model_only = np.sum(Nparameters)  # Total number of parameters in the model.
# Get number of model components (excluding EAC)
Ncomponents = models.shape[0]
if models[Ncomponents - 1] == 'EAC':
    Ncomponents = Ncomponents - 1
# print (Ncomponents,Nparameters_model_only,indexmain)

# Find indices of fixed parameters in this model
indfix = np.where(parfix == 'f')[0]
# Find indices of variable parameters in this model
indvary = np.where(parfix == 'v')[0]

# Initialise parameter names
fullparnames = np.full(Nparameters_model_only, ' ' * 40)
fullparnames_err = np.full(Nparameters_model_only, ' ' * 40)

# Luminosity distance
lm = integral_norm(z)[0]
Dl = lm * (1 + z)  # dL (cm)

# GET COVARIANCE MATRIX AND MEAN VALUES OF ALL PARAMETERS
# ========================================================
tmpcovariance_matrix = hdu[2].data.field('COVARMAT')
if tmpcovariance_matrix[0][0].shape[0] != Nparameters_model_only:
    print('Incompatible covariance matrix size - correct for EAC')
covariance_matrix = tmpcovariance_matrix[0][0:Nparameters_model_only, 0:Nparameters_model_only]

# Replace columns for fixed parameters with zero.
for j in indfix:
    covariance_matrix[:, j] = 0.
# Replace rows for fixed parameters with zero.
for i in indfix:
    covariance_matrix[i, :] = 0.
    covariance_matrix[np.abs(covariance_matrix) < 1.e-18] = 0.
param_mean = []
param_err = []
fileout.write('From fit\n')
fileout.write(' =' * 75 + '\n')
strout = 'Parameter'.ljust(18) + '  ' + 'Value'.rjust(12) + '        ' + 'err'.rjust(12) + '\n'
fileout.write(strout)
fileout.write(' =' * 75 + '\n')

# print(covariance_matrix)

for ip in range(Nparameters_model_only):
    param_mean.append(hdu[2].data.field('PARAM' + str(ip))[0][0])
    param_err.append(hdu[2].data.field('PARAM' + str(ip))[0][1])
    strout = parnames[ip].ljust(18) + '= ' + str("%0.3e" % param_mean[ip]).rjust(12) + '    +/- ' + str("%0.3e" % param_err[ip]).rjust(12) + '\n'
    fileout.write(strout)
    print(parnames[ip], '= ', param_mean[ip], '+/-', param_err[ip])
param_mean = np.asarray(param_mean)

# exit()


# Monte Carlo simulations
# =======================
# Generate nvect vectors of parameter values
np.random.seed(0)
pararray = np.random.multivariate_normal(param_mean, covariance_matrix, nvect)

# Write to text file
fout_par = open(filepar, 'w')
for i in range(nvect):
    fout_par.write(('%15.8e    ' * len(pararray[i]) + '\n') % tuple(pararray[i]))

# Plot
fig, ax = plt.subplots(figsize=(21, 15))
subplots_adjust(hspace=0.3)
# Call plot and median calculation code for each variable and write values to output file.
fileout.write('\n\nFrom Monte Carlo simulation based on fit results')
fileout.write('\n' + '=' * 75)
strout = '\nParameter'.ljust(18) + 'Median'.rjust(12) + 'symmetr_err'.rjust(20) + '+error'.rjust(12) + '-error'.rjust(12)
fileout.write(strout)
fileout.write('\n' + '=' * 75)

iplot = 0
for ip in indvary:
    iplot = iplot + 1
    calculate_errors_median_plot(iplot, parnames[ip], parnames[ip], pararray[:, ip], 'C0')

# Calculate flux, get Epeak array
# ================================
# empty array for the flux, fluence for each of the simulated model.
flux_array = np.zeros(nvect)
fluence_array = np.zeros(nvect)
flux_BB_array = np.zeros(nvect)
flux_bolometric_array = np.zeros(nvect)

Epeak_array = np.zeros(nvect)
print('\nCalculating flux and other parameters...\n')

# Loop on simulated models
for isim in range(nvect):

    # Loop on models
    ip = 0  # number of parameters already read
    for ic, component in enumerate(models):
        base = False
        if ic == indexmain:
            base = True
        if component == 'PL':
            val_A = pararray[isim, ip]
            val_Epiv = pararray[isim, ip + 1]
            val_Epiv = 100.
            val_index = pararray[isim, ip + 2]
            # Calculate flux
            flux_component = integral_PL(val_A, val_Epiv, val_index, Eintegral1, Eintegral2)[0]  # keV/m2/s
            flux_bol_component = integral_PL(val_A, val_Epiv, val_index, Eiso1 / (1 + z), Eiso2 / (1 + z))[0]  # keV/m2/s

            flux_array[isim] = flux_array[isim] + flux_component
            flux_bolometric_array[isim] = flux_bolometric_array[isim] + flux_bol_component

        if component == 'SBPL':
            val_A = pararray[isim, ip]
            val_Epiv = pararray[isim, ip + 1]
            val_Epiv = 100.
            val_alpha = pararray[isim, ip + 2]
            val_Ebreak = pararray[isim, ip + 3]
            val_delta = pararray[isim, ip + 4]
            val_delta = 0.3
            val_beta = pararray[isim, ip + 5]
            # Epeak
            if base:
                # if val_alpha<-2. or val_beta>-2.:
                # print(isim,ic,val_alpha, val_beta)
                Epeak_array[isim] = val_Ebreak * 10**(1. / (2.) * val_delta * np.log((val_alpha + 2.) / (-val_beta - 2.)))

            # Flux and fluence
            # print('11111',isim,ic)
            if isim in [20, 2224, 2593, 3408, 4628, 5133, 5634, 6676]:
                print(val_A, val_Epiv, val_alpha, val_Ebreak, val_delta, val_beta)
            flux_component = integral_SBPL(val_A, val_Epiv, val_alpha, val_Ebreak, val_delta, val_beta, Eintegral1, Eintegral2)[0]  # keV/m2/s
            # print('aaaaa')
            flux_bol_component = integral_SBPL(val_A, val_Epiv, val_alpha, val_Ebreak, val_delta, val_beta, Eiso1 / (1 + z), Eiso2 / (1 + z))[0]  # keV/m2/s
            # print(flux_component, flux_bol_component)
            # print('bbbbb')
            flux_array[isim] = flux_array[isim] + flux_component
            # print('cccccc')
            flux_bolometric_array[isim] = flux_bolometric_array[isim] + flux_bol_component
            # print('2222',isim,ic)

        if component == 'BAND':
            val_A = pararray[isim, ip]
            val_Epeak = pararray[isim, ip + 1]
            val_alpha = pararray[isim, ip + 2]
            val_beta = pararray[isim, ip + 3]
            # Epeak
            if base:
                Epeak_array[isim] = val_Epeak

            # Flux and fluence
            flux_component = integral_BAND(val_A, val_Epeak, val_alpha, val_beta, Eintegral1, Eintegral2)[0]  # keV/m2/s
            flux_bol_component = integral_BAND(val_A, val_Epeak, val_alpha, val_beta, Eiso1 / (1 + z), Eiso2 / (1 + z))[0]  # keV/m2/s

            flux_array[isim] = flux_array[isim] + flux_component
            flux_bolometric_array[isim] = flux_bolometric_array[isim] + flux_bol_component

        if component == 'CPL':
            val_A = pararray[isim, ip]
            val_Epeak = pararray[isim, ip + 1]
            val_index = pararray[isim, ip + 2]
            val_Epiv = pararray[isim, ip + 3]

            val_Epiv = 100.
            # Epeak
            if base:
                Epeak_array[isim] = val_Epeak

            # Flux and fluence
            flux_component = integral_CPL(val_A, val_Epeak, val_index, val_Epiv, Eintegral1, Eintegral2)[0]  # keV/m2/s
            flux_bol_component = integral_CPL(val_A, val_Epeak, val_index, val_Epiv, Eiso1 / (1 + z), Eiso2 / (1 + z))[0]  # keV/m2/s

            flux_array[isim] = flux_array[isim] + flux_component
            flux_bolometric_array[isim] = flux_bolometric_array[isim] + flux_bol_component

        if component == 'BB':
            val_A = pararray[isim, ip]
            val_kT = pararray[isim, ip + 1]
            flux_component = integral_BB(val_A, val_kT, Eintegral1, Eintegral2, integration_steps)  # keV/m2/s
            flux_bol_component = integral_BB(val_A, val_kT, Eiso1 / (1 + z), Eiso2 / (1 + z), integration_steps)  # keV/m2/s

            flux_array[isim] = flux_array[isim] + flux_component
            flux_BB_array[isim] = flux_BB_array[isim] + flux_component
            flux_bolometric_array[isim] = flux_bolometric_array[isim] + flux_bol_component
            # print('aaaa','%.3e'%val_A,'%.3f'%val_kT,'%.3e'%flux_component,integral_BB(val_A,val_kT,Eintegral1,Eintegral2,integration_steps))

        ip = ip + Nparameters[ic]
    # print('Final',flux_array[isim],fluence_array[isim])

# Units/normalisations/etc.
flux_array = flux_array * keVtoErg  # erg/m2/s
flux_BB_array = flux_BB_array * keVtoErg  # erg/m2/s
fluence_bolometric_array = flux_bolometric_array * Tint
E_iso = fluence_bolometric_array * (4. * np.pi * Dl**2) / (1. + z) * keVtoErg  # erg/m2
fluence_array = flux_array * Tint  # erg/m2

# Call code for median, error calculations and plots
# Flux
iplot = iplot + 1
calculate_errors_median_plot(iplot, 'Flux', r'Flux_{tot} [erg/cm^2/s]', flux_array, 'C2')

if 'BB' in models:
    # Flux BB
    iplot = iplot + 1
    calculate_errors_median_plot(iplot, 'FluxBB', r'Flux_{BB} [erg/cm^2/s]', flux_BB_array, 'C2')

    # Flux BB/Flux tot
    fluxBB_o_fluxTot = flux_BB_array / flux_array
    iplot = iplot + 1
    calculate_errors_median_plot(iplot, 'FluxBB/Flux tot', r'\mathrm{Flux}_{BB}/\mathrm{Flux}_{tot} ', fluxBB_o_fluxTot, '#248424')

# Fluence
iplot = iplot + 1
calculate_errors_median_plot(iplot, 'Fluence', r'Fluence [erg/cm^2]', fluence_array, 'darkslateblue')

# CALCULATE Eipeak AND Eiso, PLOT
# ===============================
idxt = np.where(flux_array > 6.072e-4)
print('Epeak ', Epeak_array[idxt])
print(np.where(Epeak_array > 1000.))
# Epeak
if 'SBPL' in models:
    # Delete those simulations for which Epeak is infinite. This happens when beta is large.
    idxnan = np.argwhere(np.isnan(Epeak_array))
    Epeak_array = np.delete(Epeak_array, idxnan)
iplot = iplot + 1
calculate_errors_median_plot(iplot, 'Epeak', r'E_{peak} [keV]', Epeak_array, 'C0')

# Eipeak
iplot = iplot + 1
Eipeak_array = Epeak_array * (1 + z)
calculate_errors_median_plot(iplot, 'Eipeak', r'E_{i,peak} [keV]', Eipeak_array, 'C1')

# Eiso
iplot = iplot + 1
calculate_errors_median_plot(iplot, 'Eiso', r'E_{iso} [erg]', E_iso, 'C1')
plt.savefig(filepng_par, bbox_inches="tight")
# plt.savefig(filepng_par)
# plt.show()


plt.close(fig)

idxt = np.where(flux_array > 6.072e-4)
print('flux ', flux_array[idxt])
# print('flux_BB ',flux_BB_array[idxt])
print('fluence bol ', fluence_bolometric_array[idxt])

# print('Eipeak ',Eipeak_array[idxt])
print('Eiso ', E_iso[idxt])
print('fluence ', fluence_array[idxt])

print('########')

#print(np.where(flux_array > 6.072e-4))
#print(np.where(E_iso > 1.e53))

fileout.close()
