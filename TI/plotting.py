"""
Created on Wed Feb 23 11:10:51 2022
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

fs = 16
mpl.rcParams.update({'font.size': fs})

folders = [f for f in os.listdir(os.curdir) if os.path.isdir(f) and '__pycache__' not in f]
folders.sort()

cwd = os.getcwd()


csv_files = []
for i in folders:
    os.chdir(i)
    _f = [f for f in os.listdir(os.curdir) if f.endswith('.csv')][0]
    csv_files.append(_f)
    os.chdir(cwd)

# read all the csv files into a list of data frame
df_list = [pd.read_csv(f'{folders[i]}/{v}')[['e_i_peak', 'e_isotropic']] for i, v in enumerate(csv_files)]

df = df_list[0].T

df = df.append([i.T for i in df_list[1:]])
df.columns = (['median', 'low_err', 'high_err'])

df.to_csv('eip_eis_df_Ep0.csv')

df_e_i_peak = df.loc['e_i_peak']
df_e_isotro = df.loc['e_isotropic']

plt.figure(figsize=(10, 8))
plt.plot(df_e_i_peak['median'], df_e_isotro['median']/1e49, 'go', ls=':')

redshift = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#for i, v in enumerate(redshift):
#    plt.annotate(v, (df_e_isotro['median'][i], df_e_i_peak['median'][i]-200))

#plt.annotate(redshift[4], (df_e_isotro['median'][4], df_e_i_peak['median'][4]-300))
#plt.annotate(redshift[5], (df_e_isotro['median'][5], df_e_i_peak['median'][5]-350))
#plt.annotate(redshift[6], (df_e_isotro['median'][6], df_e_i_peak['median'][6]-500))
#plt.annotate(redshift[7], (df_e_isotro['median'][7], df_e_i_peak['median'][7]-600))

plt.grid('both')

#plt.xscale('log')
#plt.yscale('log')
plt.ylabel(r'E$_\mathrm{isotropic, 49}$ [erg]')
plt.xlabel(r'E$_\mathrm{i,peak}$ [keV]')
#plt.xlim(1.01, 1e4)
plt.tight_layout()
plt.savefig('Ep0__eipeak_eisotropic.pdf')

