import math as math

import numpy as np
###################################################################################
# OTHER FUNCTIONS
###################################################################################
from matplotlib import pyplot as plt


def get_model_details(model_name):
    # order of models in rmfit
    rmfitorder = np.array(['PL', 'SBPL', 'BAND', 'CPL', 'BB', 'EAC'])
    # number of parameters in each of these models. I am putting 0 for EAC for now, but I will update this later when the fits file is read.
    rmfit_par_numbers = np.array([3, 6, 4, 4, 2, 0])
    # names of all parameters (model-wise)
    rmfit_parnames = np.array([np.array(['A_pl', 'Epiv_pl', 'Index_pl']),
                               np.array(['A_sbpl', 'Epiv_sbpl', 'alpha_sbpl', 'Ebreak_sbpl', 'BreakScale_sbpl', 'beta_sbpl']),
                               np.array(['A_band', 'Epeak_band', 'alpha_band', 'beta_band']),
                               np.array(['A_comp', 'Epeak_comp', 'Index_comp', 'Epiv_comp']),
                               np.array(['A_bb', 'kT_bb']),
                               np.array(['EAC'])], dtype=object)
    rmfit_parfix = np.asarray([np.asarray(['v', 'f', 'v']),
                               np.asarray(['v', 'f', 'v', 'v', 'f', 'v']),
                               np.asarray(['v', 'v', 'v', 'v']),
                               np.asarray(['v', 'v', 'v', 'f']),
                               np.asarray(['v', 'v'])], dtype=object)
    model_array = np.array(model_name.split('_'))  # Get array of models in the original order
    models = np.zeros(model_array.shape[0], dtype='str')  # make new empty array of the same length for rmfit order
    # print model_name
    # print 'model_array',model_array
    indices = np.where(np.in1d(rmfitorder, model_array))[0]  # Get indices of the models in the original order according to rmfit order
    # print indices
    models = rmfitorder[indices]  # new model array
    Nparameters = rmfit_par_numbers[indices]  # Array for number of parameters according to rmfit order
    parnames = np.concatenate(rmfit_parnames[indices])  # Array of parameter names according to rmfit order.
    parfix = np.concatenate(rmfit_parfix[indices])  # Array of parameter fix or vary according to rmfit order.
    # print rmfit_parnames[indices]
    # print 'nparnames',parnames
    return models, Nparameters, parnames, parfix


###################################################################################
# FUNCTIONS WITH MODEL DEFINITIONS, ERROR CALCULATION, OTHERS
###################################################################################


def fSBPL(E, A, Epiv, alpha, Ebreak, delta, beta):
    Epiv = 100.
    delta = 0.3
    r = math.log10(E / Ebreak) / delta
    rp = math.log10(Epiv / Ebreak) / delta
    a = 1. / 2. * delta * (beta - alpha) * np.log((np.exp(r) + np.exp(-r)) / 2.)
    ap = 1. / 2. * delta * (beta - alpha) * np.log((np.exp(rp) + np.exp(-rp)) / 2.)
    N = A * (E / Epiv)**((alpha + beta) / 2.) * 10.**(a - ap)
    return N


def fBAND(E, Amplitude, Epeak, alpha, beta):
    Ec = (alpha - beta) * Epeak / (2.0 + alpha)  #
    if E <= Ec:
        N = Amplitude * (E / 100)**alpha * np.exp(-(alpha + 2.) * E / Epeak)
    else:
        N = Amplitude * (E / 100)**beta * np.exp(beta - alpha) * (((alpha - beta) * Epeak) / (100.0 * (alpha + 2.)))**(alpha - beta)

    return N


def fCPL(E, A, Epeak, index, Epiv):
    Epiv = 100.
    N = A * (E / Epiv)**index * math.exp(-E * (2. + index) / Epeak)
    return N


def fPL(E, A, Epiv, index):
    Epiv = 100.
    N = A * (E / Epiv)**index
    return N


def fBB(E, A, kT):
    # np.exp(E/kT) is very large i. e. greater than 1.e304, then the denominator is infinite. Which means that the numerator is practically zero (~1.e-304), therefore do not
    # calculate N. Just put it equal to zero.
    # exp(E/kT) becomes greater than ~1.e304 when E/kT > 700.
    if E / kT > 700:
        N = 0.
    else:
        N = A * (E**2 / (np.exp(E / kT) - 1.))
    return N


# ================================================================================
# Flux functions
# ================================================================================


def ENE_SBPL(E, A, Epiv, alpha, Ebreak, delta, beta):
    Epiv = 100.
    delta = 0.3
    r = math.log10(E / Ebreak) / delta
    rp = math.log10(Epiv / Ebreak) / delta
    a = 1. / 2. * delta * (beta - alpha) * math.log((np.exp(r) + np.exp(-r)) / 2.)
    ap = 1. / 2. * delta * (beta - alpha) * math.log((np.exp(rp) + np.exp(-rp)) / 2.)
    N = A * (E / Epiv)**((alpha + beta) / 2.) * 10**(a - ap)
    # print(r,rp,a,ap,N)
    return E * N


def ENE_BAND(E, Amplitude, Epeak, alpha, beta):
    if E <= (((alpha - beta) / (alpha + 2)) * Epeak):
        N = Amplitude * (E / 100)**alpha * np.exp(-(alpha + 2) * E / Epeak)
    else:
        N = Amplitude * (E / 100)**beta * np.exp(beta - alpha) * (((alpha - beta) * Epeak) / (100 * (alpha + 2)))**(alpha - beta)
    # E_iso = E*N
    return E * N
    # return E_iso*Tint*(4.*np.pi*D**2)/(1+z)*1.60217657e-9 # Tint is the time interval you used to perform the the joint GBM-LAT spectral analysis


def ENE_CPL(E, A, Epeak, index, Epiv=100.):
    N = A * (E / Epiv)**index * math.exp(-E * (2 + index) / Epeak)
    return E * N


def ENE_PL(E, A, Epiv, index):
    N = A * (E / Epiv)**index
    return E * N


def ENE_BB(E, A, kT):
    # N=A*(E**2/(np.exp(E/kT)-1.))
    if E / kT > 700:
        N = 0.
    else:
        N = A * (E**2 / (np.exp(E / kT) - 1.))
    return E * N


# ================================================================================
# Define integrals for various models. Integrals are calculated between the energy limits E1 and E2.
# ================================================================================

def integral_SBPL(A, Epiv, alpha, Ebreak, delta, beta, E1, E2):
    Epiv = 100.
    delta = 0.3
    # print 'inside integral_SBPL z=',z
    # print A,Epiv,alpha,Ebreak,delta,beta
    return si.quad(ENE_SBPL, E1, E2, args=(A, Epiv, alpha, Ebreak, delta, beta))


def integral_BAND(Amplitude, Epeak, alpha, beta, E1, E2):
    # print 'inside integral_band z=',z
    # print Amplitude,Epeak,alpha,beta
    # test=si.quad(fBand, 1./(1.+z), 10000./(1.+z),args=(Amplitude,Epeak,alpha,beta))
    # print test
    return si.quad(ENE_BAND, E1, E2, args=(Amplitude, Epeak, alpha, beta))


def integral_CPL(A, Epeak, index, Epiv, E1, E2):
    Epiv = 100.
    # print 'inside integral_comp z=',z
    # print A,Epeak,index,Epiv
    return si.quad(ENE_CPL, E1, E2, args=(A, Epeak, index, Epiv))


def integral_PL(A, Epiv, index, E1, E2):
    Epiv = 100.
    # print 'inside integral_PL z=',z
    # print A,Epiv,index
    return si.quad(ENE_PL, E1, E2, args=(A, Epiv, index))


def integral_BB(A, kT, E1, E2, FNsteps):
    # Steps need to be logarithmic
    Nsteps = int(FNsteps)
    E1log = math.log10(E1)
    E2log = math.log10(E2)

    vE = np.logspace(E1log, E2log, Nsteps)
    dE = np.zeros(Nsteps)
    for i in range(Nsteps - 1):
        dE[i + 1] = vE[i + 1] - vE[i]

    integral = 0.
    for i in range(Nsteps - 1):
        Emid = vE[i] + dE[i + 1] / 2.
        ENE_mid = ENE_BB(Emid, A, kT)
        integral = integral + ENE_mid * dE[i + 1]

    return integral
    # return si.quad(ENE_BB, E1, E2,args=(A,kT))


# DEFINE FUNCTIONS

def luminosity_distance(z1):
    return c / h0 * 1 / np.sqrt((1 - omega_L) * (1 + z1)**3 + omega_L)


def integral_norm(z2):
    return si.quad(luminosity_distance, 0, z2)


def calculate_errors_median_plot(iplot, title, axis_label, plotarray, plotcolour):
    print('\n================== ' + title + ' =========================')
    plt.subplot(4, 4, iplot)
    nBins = 250  # number of bins in histogram
    n, bins, patches = plt.hist(plotarray, bins=nBins, color=plotcolour)  # fill histogram with first parameter values
    Med = np.median(plotarray)  # get median (middle value) of the value of the parameter
    gHi = np.where(plotarray >= np.median(plotarray))[0]  # indices of parameter values higher than median
    gLo = np.where(plotarray < np.median(plotarray))[0]  # indices of parameter values lower than median
    vSortLo = np.sort(plotarray[gLo])  # Sort values according to indices (low)
    vSortHi = np.sort(plotarray[gHi])  # Sort values according to indices (high)
    NormLo = vSortLo[int((1.0 - sLim) * np.size(vSortLo))]  # number of values below
    NormHi = vSortHi[int(sLim * np.size(vSortHi))]  # number of values  above

    print("INFO: Lower and upper %i percent ranges are:  %.10f %.10f" % (sLim * 100, Med - NormLo, NormHi - Med))
    print('symetric error = ', ((Med - NormLo) + (NormHi - Med)) / 2.)
    print('Median = ', Med)
    strout = title.ljust(18) + str("%0.3e" % Med).rjust(12) + str("%0.3e" % (((Med - NormLo) + (NormHi - Med)) / 2.)).rjust(20) + str("%0.3e" % (Med - NormLo)).rjust(12) + str(
            "%0.3e" % (NormHi - Med)).rjust(12)
    # fileout.write('\n'+axis_label+'= '+str("%0.3e"%Med))
    # fileout.write('\n'+axis_label+'error= '+str("%0.3e"%(((Med-NormLo)+(NormHi-Med))/2.)))
    # fileout.write('\n'+axis_label+'error low= '+str("%0.3e"%(Med-NormLo)))
    # fileout.write('\n'+axis_label+'error up= '+str("%0.3e"%(NormHi-Med)))
    fileout.write('\n' + strout)
    # plt.text(0.01262,70, '%.2f-%.2e+%.2e' %(Med, Med-NormLo, NormHi-Med))
    # plt.text(0.1,0.9, '%.2f-%.2e+%.2e' %(Med, Med-NormLo, NormHi-Med),transform=ax.transAxes)
    xmin = np.amin(plotarray)
    xmax = np.amin(plotarray)
    dx = (xmin - xmax) / 10.
    ymin = 0.
    ymax = n.max()
    dy = (ymax - ymin) / 10.
    ymax = ymax + dy
    plt.plot([Med, Med], [0, ymax], 'k-', lw=3)  # plot line at position of median
    plt.plot([NormLo, NormLo], [0, ymax], 'k--', lw=3)
    plt.plot([NormHi, NormHi], [0, ymax], 'k--', lw=3)
    # plt.text(xmin+1.*dx,70, '%.3f -%.2e +%.2e' %(Med,Med-NormLo,NormHi-Med))
    plt.text(xmin + 1. * dx, ymax - 3 * dy, '%.3e ' % (Med))
    plt.text(xmin + 1. * dx, ymax - 4 * dy, '+%.2e' % (NormHi - Med))
    plt.text(xmin + 1. * dx, ymax - 5 * dy, '-%.2e ' % (Med - NormLo))
    plt.ylim(ymin, ymax)
    plt.ylabel('count')
    plt.xlabel("'$" + axis_label + "'$")
    plt.grid(True, alpha=.2)
    # exit()
