"""Showing the influence of various soil moisture parameterizations
on near-surface temperatures

Adam Michael Bauer
University of Illinois Urbana-Champaign
5.12.2023
adammb4@illinois.edu
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fig_locs import figs_path


# plotting parameters
color_list = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'] * 2
marker_list = ['o', 's', 'P', '+', 'D', 'v', '3', 'm'] * 2
markersize = 6
linestyle_list = ['solid', 'dashed', 'dashdot', 'dotted'] * 4
linewidth = 2

cdds_params={'axes.linewidth': 3,
 'axes.axisbelow': False,
 'axes.edgecolor': 'black',
 'axes.facecolor': 'None',
 'axes.grid': False,
 'axes.labelcolor': 'black',
 'axes.spines.right': False,
 'axes.spines.top': False,
 'axes.titlesize': 20,
 'axes.labelsize': 20,
 'axes.titlelocation': 'left',
 'figure.facecolor': 'white',
 'figure.figsize': (7, 5),
 'lines.solid_capstyle': 'round',
 'lines.linewidth': 2.5,
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'text.color': 'black',
 'legend.frameon': False,
 'xtick.bottom': True,
 'xtick.major.width': 3,
 'xtick.major.size': 6,
 'xtick.color': 'black',
 'xtick.direction': 'out',
 'xtick.top': False,
 'ytick.color': 'black',
 'ytick.direction': 'out',
 'ytick.left': True,
 'ytick.right': False,
 'ytick.color' : 'black',
 'ytick.major.width': 3,
 'ytick.major.size': 6,
 'axes.prop_cycle': plt.cycler(color=color_list, linestyle=linestyle_list),
 'font.size': 16,
 'font.family': 'serif'}
plt.rcParams.update(cdds_params)

dsm = 0.05
sm = np.arange(0, 5, dsm)

def param_1(sm):
    # linear parameterization of evapotranspiration on sm
    return sm

def param_2(sm):
    # square root parameterization of evapotranspiration on sm
    return np.sqrt(sm)

def param_3(sm, sm_crit):
    # linear until some critical value
    sm_param = np.zeros_like(sm)
    sm_param[sm < sm_crit] = sm[sm < sm_crit]
    sm_param[sm >= sm_crit] = sm_crit
    return sm_param

def param_4(sm):
    # logarithmic 
    return np.log(sm + 1)

t5 = np.zeros_like(sm, dtype=float)
t1 = (1 + param_1(sm))**(-1)
t2 = (1 + param_2(sm))**(-1)
t4 = (1 + param_3(sm, 30 * dsm))**(-1)
t5[sm <= dsm*30] = (t1[sm==30 * dsm] - t1[0]) * (sm[sm==30 * dsm] - sm[0])**(-1) * sm[sm<= dsm * 30] + t1[0]
t5[sm > dsm*30] = (t1[-1] - t1[sm==30*dsm]) * (sm[-1] - sm[sm==30 * dsm])**(-1) * (sm[sm> dsm * 30] - sm[sm==dsm*30]) + t1[sm==dsm*30]
t6 = (1 + param_4(sm))**(-1)

fig, ax = plt.subplots(1)

ax.plot(sm, t1, label=r"$\ell_{\theta}(\theta) = \theta$")
ax.plot(sm, t2, label=r"$\ell_{\theta}(\theta) = \sqrt{\theta}$")
ax.plot(sm, t5, label=str("Breakpoint"))
ax.plot(sm, t4, label=r"$\ell_{\theta}(\theta) =$" + r"piecewise")

ax.plot(sm, t6, label=r"$\ell_{\theta}(\theta) = \ln(\theta + 1)$")

ax.set_ylabel("Temperature")
ax.set_xlabel("Soil moisture")
ax.legend(frameon=False, ncol=1, fontsize=14)

ax.set_xticks([0, 5])
ax.set_yticks([0.2,1])

ax.set_xticklabels([])
ax.set_yticklabels([])

sns.despine(trim=True, offset=10)

plt.subplots_adjust(bottom=0.2)

fig.savefig(figs_path + "fig3_sm_t_parameterizations.png", dpi=300)
print("Fig saved successfully.")
