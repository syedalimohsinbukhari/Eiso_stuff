"""
Created on Mar 07 11:58:25 2022
"""
import os
import subprocess

redshifts = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

curdir = os.getcwd()

for i in redshifts:
    subprocess.call(f'python {curdir}/flux_parameter_calculations_loop.py {i}', shell=True)
