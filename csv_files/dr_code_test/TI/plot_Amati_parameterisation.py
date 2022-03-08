# !/usr/bin/env python


# S. Sajjad 2022/01/17
# from __future__ import print_function

from pylab import *

# S. Sajjad 2022/01/17
# from __future__ import print_function

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = "True"


# PLOT PRELIMINARIES/AMATI RELATION GRAPH

# Array limits
powxmin = 46
powxmax = 59
powymin = 1.
powymax = 5.

# Plot limits
eisomin = 8.e51
eisomax = 3.e56
eipeakmin = 100.
eipeakmax = 20500.

# https://www.researchgate.net/publication/309893928_A_Brief_Review_of_the_Amati_Relation_for_GRBs

# Relationships from Amati 2008. https://academic.oup.com/mnras/article/391/2/577/1079938
# https://arxiv.org/pdf/0805.0377.pdf
#
# K=94.  #intercept
# m=0.57 #slope
# (Eiso/10**52erg)= K (Eipeak**m

# From F. Dirirsa 2019: https://arxiv.org/pdf/1910.07009.pdf

# (Eiso/10**52erg)= K (Eipeak/950 keV)**m

eiso0 = 1.e52
eipeak0 = 950.

# This is also written as
# y=k +mx

# With y=log10(Eiso/10**52erg)
# x=log10(Eipeak/950 keV)
# log10(K)=k
# or K=10**k

# K=10**(1.67 +/- 0.16)
# m=1.16 +/- 0.37

# notation used by Dirirsa et al. k=logK
logK = 1.67
sigma_logK = 0.16

k = logK
sigma_k = sigma_logK

m = 1.16
sigma_m = 0.37

# To get the 1 sigma interval above and below this, the uncertainty on y is given by Wang et al. 2016, Demianski et al. 2017, Dirirsa et al. 2019.
# sigmay=  sqrt( sigma_k**2 + m**2*sigma_x**2 + x**2 * sigma_m**2 + sigma_ext**2)

# Where sigma_m and sigma_k can be taken from Dirirsa et al. 2019, sigma_x is the error on Eipeak/950 keV.  And sigma_ext is obtained by Dirirsa et al. too by fitting
# different samples. For the F10 Dirirsa et al. sample, sigma_ext=0.47 +/- 0.12.
sigma_ext = 0.47

eip_vect = np.linspace(0.5, 1.7, num=100)

# eis_vect= math.log10(1.8) +0.52*eip_vect
eis_vect = (eip_vect + 26.74 + math.log10(100.) - 0.55 * math.log10(1.e52)) / 0.53
# eis_vect_err=np.sqrt(ei_vect**2 * 2.62797e-01**2 + 2.21451e-01**2  )

# sigma_x can be taken as zero since sigma_ext dominates. Also cf. note in Amati 2002, 2006.
sigma_x = 0.

# logeiso = np.logspace(xmin,xmax,num=100) # Generate 100 logarithmically spaced numbers between xmin and xmax
eipeak = np.logspace(powymin, powymax, num=100, base=10.)

x = np.log10(eipeak / eipeak0)

# print (eipeak)id

# print(x)

sigmay = np.sqrt(sigma_k**2 + m**2 * sigma_x**2 + x**2 * sigma_m**2 + sigma_ext**2)

# We can take the mean to get smooth lines instead.

sigmay = np.mean(np.sqrt(sigma_k**2 + m**2 * sigma_x**2 + x**2 * sigma_m**2 + sigma_ext**2))

# In terms of y and eiso

y = k + m * x

eiso = (10.**y) * eiso0

# 1 and 2-sigma uncertainty limits

ytop = y + sigmay
ybot = y - sigmay

eisotop = (10.**ytop) * eiso0
eisobot = (10.**ybot) * eiso0

ytop2 = y + 2. * sigmay
ybot2 = y - 2. * sigmay

eisotop2 = (10.**ytop2) * eiso0
eisobot2 = (10.**ybot2) * eiso0

fig, ax = plt.subplots(1, 1, figsize=(5.5, 6.))
plt.plot(eipeak, eiso, lw=1, alpha=0.45, color='k')

plt.fill_between(eipeak, eisobot, eisotop, color='k', alpha=.155)
plt.fill_between(eipeak, eisobot2, eisotop2, color='k', alpha=.155)

plt.plot(eipeak, eisobot, color='k', alpha=0.45, linestyle=':')
plt.plot(eipeak, eisotop, color='k', alpha=0.45, linestyle=':')
plt.plot(eipeak, eisobot2, color='k', alpha=0.45, linestyle=':')
plt.plot(eipeak, eisotop2, color='k', alpha=0.45, linestyle=':')

plt.xlabel('E$_\mathrm{i,peak}$   (keV)')
plt.ylabel('E$_\mathrm{iso}$   (erg)')
plt.xscale('log')
plt.yscale('log')
# plt.grid()
# legend = plt.legend(loc="upper left", fontsize='xx-small',fancybox=True,ncol=3,framealpha=0.5)
# legend.get_frame().set_alpha(None)
# legend.get_frame().set_facecolor((1., 1., 1., 0.8))
plt.xlim(eipeakmin, eipeakmax)
plt.ylim(eisomin, eisomax)

plt.savefig('amati.png',bbox_inches='tight')#
plt.savefig('amati.pdf',bbox_inches='tight')#

#return fig, ax
