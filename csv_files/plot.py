"""
Created on Wed Feb 23 12:27:19 2022
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from csv_files.dr_code_test.plot_Amati_parameterisation import amati_plot

fs = 16
mpl.rcParams.update({'font.size': fs})

csv_files = [f for f in os.listdir(os.curdir) if f.endswith('.csv')]
csv_files.sort()

df_list = [pd.read_csv(f'{v}').set_index('Unnamed: 0') for i, v in enumerate(csv_files)]

div = 1

labels = ['Ep1', 'Ep2', 'Ep3']

df_TI_ei = df_list[0].loc['e_isotropic']
df_TI_ep = df_list[0].loc['e_i_peak']

# plt.figure(figsize=(10, 8))

f, ax = amati_plot()


ax.plot(df_TI_ep['median'], df_TI_ei['median']/div, 'ko', ls='--', label=r'T$_\mathrm{90}$')

#redshift = [0.000001, 0.001, 0.01, 0.1, 0.5, 1, 2, 3]

#for i, v in enumerate(redshift[0:-4]):
#    plt.annotate(v, (df_TI_ei['median'][i]/div, df_TI_ep['median'][i]-150))

#plt.annotate(redshift[4], (df_TI_ei['median'][4]/div, df_TI_ep['median'][4]-200))
#plt.annotate(redshift[5], (df_TI_ei['median'][5]/div, df_TI_ep['median'][5]-250))
#plt.annotate(redshift[6], (df_TI_ei['median'][6]/div, df_TI_ep['median'][6]-400))
#plt.annotate(redshift[7], (df_TI_ei['median'][7]/div, df_TI_ep['median'][7]-500))

[ax.plot(v.loc['e_i_peak']['median'], v.loc['e_isotropic']['median']/div, marker='o', ls='--', alpha=0.25, label=labels[i]) for i, v in enumerate(df_list[1:])]

ax.set_ylim(bottom=1e45)

# plt.xscale('log')
# plt.yscale('log')
# plt.ylabel(r'E$_\mathrm{isotropic, 49}$ [erg]')
# plt.xlabel(r'E$_\mathrm{peak}$ (1+z) [keV]')
# #plt.ylim(1.01, 1e4)
# #plt.xlim(1e45, 1e51)
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('eisoep_compare.pdf')
# plt.close()
