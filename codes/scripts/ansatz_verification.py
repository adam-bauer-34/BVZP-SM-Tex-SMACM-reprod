"""Ansatz verification: we can ignore the superposition
principle.

Adam Michael Bauer
Univ. of Chicago
7.25.2025
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
from random import randrange 
from scipy.special import lambertw

from scripts.src.getSimulationFluxes import * 
from scripts.src.getAnalyticVerificationFuncs import *
from scripts.src.getSimulate import * 

from fig_locs import figs_path

# setup plot and set random seed for reproducibility
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

cms = mpl.cm 

np.random.seed(5)

# setup 
# define model parameters 
C = 4180 # J/K
F = 300 # W / m^2
alpha = 10 # W / K for now, usually between 5 and 25
v_L = 1.25 * 10**(-2) # kg air / m^2 / s
L = 2.5 * 10**6 # J / kg H20
gamma = 3.7 * 10**(-4) # kg H20 / kg air / K
m0 = 0.1
T_D = 290 # K 
mu = 50 # kg / m^2

param_list = [C, F, alpha, v_L, L, gamma, T_D, mu, m0]

## set up precipitation parameters
# these are rough estimates
dist_size = 10**6
precip_mean = 0.2
precip_scale = 0.5 # no basis for why this is set to one...
precip_amplitudes = np.random.gamma(precip_mean, precip_scale, size=dist_size)

poisson_mean = 3.5 # omega
precip_freqs = np.random.poisson(3.5, size=dist_size)

# simulation details
N_steps = 7776000 # approx one summer's worth of seconds
timestep = 1
time = np.arange(0, N_steps * timestep, timestep)
# 86400 seconds in one day 
omega_timestep = 3.5 * 86400 * timestep**(-1)

# do sim
moisture_full = np.zeros(N_steps)
ansatz_full = np.zeros(N_steps)
ansatz_full_exp = np.zeros(N_steps)

moisture_2d = np.zeros(N_steps)
temperature = np.zeros(N_steps)

tracker = 1

kappa = L * v_L * gamma * alpha**(-1)
timescale = mu * alpha * ( F * v_L * gamma)**(-1)

for i in range(1, N_steps):
    if i == tracker:
        tmp_precip = np.random.gamma(precip_mean, precip_scale)
        moisture_full[i] = moisture_full[i-1] + timestep * getMFlux1D(moisture_full[i-1], param_list, tmp_precip)
        
        moisture_2d[i] = moisture_2d[i-1] + timestep * getMFluxFull(temperature[i-1], moisture_2d[i-1], param_list, tmp_precip)
        temperature[i] = temperature[i-1] + timestep * getTFluxFull(temperature[i-1], moisture_2d[i-1], param_list)
        
        tmp_heaviside = getHeaviside(N_steps, i * timestep)
        tmp_exp = np.exp(- (time - (i * timestep)) / timescale)
        tmp_arg = kappa * tmp_precip * np.exp(kappa * tmp_precip) * tmp_exp
        tmp_ansatz = tmp_heaviside * kappa**(-1) * lambertw(tmp_arg, k=0, tol=1e-8)
        tmp_ansatz_exp = tmp_precip * tmp_exp * tmp_heaviside
        
        ansatz_full = ansatz_full + tmp_ansatz
        ansatz_full_exp = ansatz_full_exp + tmp_ansatz_exp
        tracker += int(np.random.poisson(omega_timestep))
    else:
        moisture_full[i] = moisture_full[i-1] + timestep * getMFlux1D(moisture_full[i-1], param_list, 0)
        moisture_2d[i] = moisture_2d[i-1] + timestep * getMFluxFull(temperature[i-1], moisture_2d[i-1], param_list, 0)
        temperature[i] = temperature[i-1] + timestep * getTFluxFull(temperature[i-1], moisture_2d[i-1], param_list)

fig, ax = plt.subplots(2, sharex=True, figsize=(14,10))

ax[0].plot(time, moisture_2d, color='k', linestyle="-", label="Full dynamics", linewidth=2.5, zorder=1)
ax[0].plot(time, moisture_full, color='#E69F00', linestyle='dotted', label="Hasselmann", linewidth=1.75, zorder=3)
ax[0].plot(time, ansatz_full, color='#56B4E9', linestyle='dashed', alpha=0.8, label='Lambert $W$',linewidth=1, zorder=5)
ax[0].plot(time, ansatz_full_exp, color='#009E73', linestyle="dashdot", label="Exponential", linewidth=1, zorder=7)
ax[0].set_ylabel("Moisture Fraction", fontsize=22)
ax[0].legend(frameon=False, fontsize=20)
ax[0].tick_params(axis="both", labelsize=20)

ax[1].plot(time, abs(moisture_full - ansatz_full), color='#56B4E9', linestyle='dashed')
ax[1].plot(time, abs(moisture_full - ansatz_full_exp), color='#009E73', linestyle='dashdot')
ax[1].set_ylabel("Rel. Diff", fontsize=22)
ax[1].tick_params(axis="both", labelsize=20)
ax[1].set_xlabel(r"Time (days)", fontsize=22)

ax[1].set_xticklabels([0,15,30,45,60,75,90])
ax[1].set_xticks([0 * 86400, int(15 *86400), int(30 *86400), int(45 *86400), int(60 *86400), int(75 *86400), int(90 *86400)])

ax[1].set_xlim([0, int(90 * 86400)])

fig.savefig(figs_path + "fig6_ansatz.png", dpi=400)
print("Figure saved successfullly!")