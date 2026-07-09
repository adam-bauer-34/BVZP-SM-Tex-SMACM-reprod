# import numpy as np
import os
import sys

anal_type = sys.argv[1]  # which analysis (tmax, t2m, or detrend)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib as mpl 

from fig_locs import figs_path
from data_locs import reanal_filename
from lat_longs import loc_dict

from src.reanal_helper import make_column, make_column_detrend

# 20 variables for the lower 48 states: ERA5_gater_regional_data.ipynb
# daily JJA data 1979-2021
ds=xr.open_dataset(reanal_filename).load()

import matplotlib.transforms as mtransforms

panels = np.array([['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I'], ['J', 'K', 'L']])

fig, ax = plt.subplot_mosaic(panels, gridspec_kw={'height_ratios': [2.2, 1.2, 1.2, 1], 'width_ratios': [1, 1, 1]},
                            figsize=(24,16))

sgp_panels = panels.T[0]
atl_panels = panels.T[1]
nyc_panels = panels.T[2]

if anal_type == 't2m' or anal_type == 't2max':
    make_column(fig, ax, sgp_panels, ds, 0, anal_type, left=True)
    make_column(fig, ax, atl_panels, ds, 3, anal_type)
    make_column(fig, ax, nyc_panels, ds, 2, anal_type, center=True)

elif anal_type == 'detrend':
    make_column_detrend(fig, ax, sgp_panels, ds, 0, left=True)
    make_column_detrend(fig, ax, atl_panels, ds, 3,)
    make_column_detrend(fig, ax, nyc_panels, ds, 2, center=True)    

if anal_type == 't2m':
    ax['D'].set_xlim((15, 45))
    ax['D'].set_xticks([15,25,35, 45])
    ax['D'].set_yticks([0, 0.1, 0.2])
    
    ax['E'].set_xlim((15, 40))
    ax['E'].set_xticks([15, 20, 25, 30, 35, 40])
    ax['E'].set_yticks([0, 0.1, 0.2, 0.3])
    
    ax['F'].set_xlim((10, 40))
    ax['F'].set_xticks([10,20,30,40])
    ax['F'].set_yticks([0, 0.1, 0.2])
    
    ax['G'].set_yticks([0,0.1, 0.2])
    ax['H'].set_yticks([0, 0.1, 0.2, 0.3])
    
    ax['I'].set_yticks([0,0.1,0.2])
    
    ax['K'].set_xticks([0.25, 0.32, 0.39, 0.46, 0.53])
    ax['B'].set_xticks([0.25, 0.32, 0.39, 0.46, 0.53])
    
    ax['C'].set_xticks([0.25,0.35,0.45,0.55])
    ax['L'].set_xticks([0.25,0.35,0.45,0.55])
    
    ax['A'].set_yticks([15,20,25,30,35,40])
    ax['A'].set_xticks([.15,.20,.25,.30,.35,.40])
    ax['J'].set_xticks([.15,.20,.25,.30,.35,.40])
    ax['J'].set_yticks([1.8, 2.2, 2.6, 3.0])
    ax['J'].set_ylim((1.8, 3.2))
    ax['B'].set_yticks([15, 20, 25, 30, 35])
    ax['C'].set_yticks([10,15,20,25,30,35])
    ax['K'].set_yticks([1, 1.5, 2, 2.5])
    ax['L'].set_yticks([2.0, 2.5, 3, 3.5])

sns.despine(offset=10, trim=True)
fig.subplots_adjust(hspace=0.6, wspace=0.2)

upper_left = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
for label in upper_left:
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(4.8, 0.1, fig.dpi_scale_trans)
    ax[label].text(0.0, 1.0, label, transform=ax[label].transAxes + trans, fontsize=22, fontweight='bold',
            verticalalignment='top', bbox=dict(facecolor='1', edgecolor='none', pad=1))
    ax[label].tick_params(axis='both', labelsize=18)

if anal_type == 't2m':
    fig_name = 'fig2-reanalysis-decouple.png'

elif anal_type == 't2max':
    fig_name = 's2-reanalysis-decouple-tmax.png'

elif anal_type == 'detrend':
    fig_name = 's3-reanalysis-decouple-detrend.png'

fig.savefig(figs_path + fig_name, dpi=400)

print("Figure saved successfully...!")