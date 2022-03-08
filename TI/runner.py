"""
Created on Feb 23 09:50:29 2022
"""

import os
import subprocess

start, end = -0.128, 27.2

redshift = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

cwd = os.getcwd()

[subprocess.call(f'python {cwd}/isotropic_energy__sys.py SBPL {i} {start} {end} {cwd}', shell=True) for i in redshift]
