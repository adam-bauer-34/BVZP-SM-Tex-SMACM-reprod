"""Phase space + nullcline fit figure.

Adam Michael Bauer
Univ. of Chicago
7/25/2025
"""

import os
import sys

anom_type = sys.argv[1]  # what type of anomaly do we apply?
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import matplotlib.transforms as mtransforms
from scipy.optimize import root_scalar

from fig_locs import figs_path
from data_locs import reanal_filename, sims_loc

from src.phase_space_helpers import *

# 20 variables for the lower 48 states: ERA5_gater_regional_data.ipynb
# daily JJA data 1979-2021
ds=xr.open_dataset(reanal_filename).load()

# make the figure! 
panel_list = [['A', 'B', 'C'], ['D', 'E', 'F']]
save_fig = True

fig, ax = plt.subplot_mosaic(panel_list, figsize=[30,16], 
                            gridspec_kw={'height_ratios': [2.5,1], 'width_ratios': [1,1,1]})

make_column(fig, ax, 0, ['A', 'D'], ds, right=False, center=False, left=True, anom_type=anom_type, sim_path=sims_loc)
make_column(fig, ax, 3, ['B', 'E'], ds, False, True, False, anom_type=anom_type, sim_path=sims_loc)
make_column(fig, ax, 2, ['C', 'F'], ds, True, False, False, anom_type=anom_type, sim_path=sims_loc)

ax['A'].set_yticks([15, 20, 25, 30, 35, 40])
ax['A'].set_xlim((0.13, 0.43))
ax['D'].set_xlim((0.13, 0.43))
ax['A'].set_xticks([.13, .18, .23, .28, .33, .38, .43])
ax['D'].set_xticks([.13, .18, .23, .28, .33, .38, .43])

ax['B'].set_yticks([15, 20, 25, 30, 35, 40])
ax['B'].set_xlim((0.26, 0.51))
ax['E'].set_xlim((0.26, 0.51))
ax['B'].set_xticks([.26, .31, .36, .41, .46, .51])
ax['E'].set_xticks([.26, .31, .36, .41, .46, .51])
ax['E'].set_yticklabels([])

ax['C'].set_yticks([15, 20, 25, 30, 35, 40])
ax['C'].set_xlim((0.25, 0.53))
ax['F'].set_xlim((0.25, 0.53))
ax['C'].set_xticks([0.25,0.32,0.39,0.46,0.53])
ax['F'].set_xticks([0.25,0.32,0.39,0.46,0.53])
ax['F'].set_yticklabels([])

for panel in panel_list[0]: 
    sns.despine(ax=ax[panel], offset=10, trim=True)
for panel in panel_list[1]:
    sns.despine(ax=ax[panel], offset=10, trim=True)
    
ax['C'].legend(loc='right', bbox_to_anchor=(0.8, -0.12), ncol=5, fontsize=22)
fig.subplots_adjust(hspace=0.31, wspace=0.2)

if anom_type == 'F':
    fig_name = 's10-phase-space-f.png'
elif anom_type == 'Td':
    fig_name = 'fig5-phase-space-td.png'

fig.savefig(figs_path + fig_name, dpi=400)

print("Figure succeessfully saved.")