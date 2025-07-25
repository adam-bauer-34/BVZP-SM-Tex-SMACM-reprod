"""Model fit at different locations.

Adam Michael Bauer
Univ. of Chicago
7/25/2025
"""


# import numpy as np
import os
import sys

loc = sys.argv[1]  # which place (sgp, atl, or dc)
length = sys.argv[2]  # short or long?
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.signal import detrend
from scipy import optimize 
import time 

import matplotlib as mpl
import matplotlib.transforms as mtransforms

from scripts.src.locations.ATL import ATL
from scripts.src.locations.DC import DC
from scripts.src.locations.SGP import SGP

from fig_locs import figs_path
from data_locs import reanal_filename
from lat_longs import loc_dict

from scripts.src.time_series_helper import *

plt.rcParams['figure.figsize'] = 12, 6
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
plt.rcParams['text.usetex'] = True
#mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'sans-serif'
color_list = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

# 20 variables for the lower 48 states: ERA5_gater_regional_data.ipynb
# daily JJA data 1979-2021
ds=xr.open_dataset(reanal_filename).load()

# get location index
if loc == 'sgp':
    loc_ind = 0
    Loc = SGP()
    if length == 'short':
        fig_name = 'fig4-model-fit-short-sgp.png'

    else:
        fig_name = 's7-model-fit-long-sgp.png'

elif loc == 'atl':
    loc_ind = 3
    Loc = ATL()
    fig_name = 's8-model-fit-long-atl.png'

elif loc == 'dc':
    loc_ind = 2
    Loc = DC()
    fig_name = 's9-model-fit-long-dc.png'

else:
    raise ValueError("Invalid location passed")

# get info
lat = loc_dict[loc_ind]['lat']
lon = loc_dict[loc_ind]['lon']
location = loc_dict[loc_ind]['location']

#extract data for one location
ds_loc = ds.sel(latitude=lat, longitude=lon, method='nearest')

# translate soil moisture data into vals between 0 and 1
m = (ds_loc.swvl1-ds_loc.swvl1.min()) / (ds_loc.swvl1.max()-ds_loc.swvl1.min())
ds_loc = ds_loc.assign(m=m)

# integreate model
M_smacm, T_smacm, LH_smacm, DH_smacm = get_SMACM_integration_fullcoupled(Loc, ds_loc)

# bin latent and dry heat fluxes from ERA5
LH_era5 = ds_loc.slhf.rolling(time=1, center=True).mean().dropna(dim='time') / 3600
LH_era5 *= -1  # downward is positive in ERA5

DH_era5 = ds_loc.sshf.rolling(time=1,center=True).mean().dropna(dim='time') + ds_loc.str.rolling(time=1,center=True).mean().dropna(dim='time')
DH_era5 *= -1/3600.


###########################
# PLOT A SINGLE SUMME
###########################
if length == 'short':
    # choose what summer you want to show -- this is the N in 1979 + N
    summer_ind = 8
    
    lower_ind = int(92 * summer_ind)
    upper_ind = int(92 * (summer_ind + 1))
    
    date_range = pd.date_range("06-01-{}".format(1979 + summer_ind),
                               "08-31-{}".format(1979 + summer_ind))

    fig, ax = plt.subplots(4, figsize=(20,16), sharex=True)
    
    ax[0].plot(date_range, T_smacm[lower_ind:upper_ind] - 273.15, color='#E69F00', linestyle='solid', label='SMACM', linewidth=2.5)
    ax[0].plot(date_range, ds_loc.t2m.values[lower_ind:upper_ind] - 273.15, color='k', linestyle='dashed', label='ERA5')
    
    ax[1].plot(date_range, M_smacm[lower_ind:upper_ind], label='SMACM', color='#E69F00', linestyle='solid', linewidth=2.5)
    ax[1].plot(date_range, ds_loc.m.values[lower_ind:upper_ind], label='ERA5', color='k', linestyle='dashed')
    
    ax[2].plot(date_range, LH_smacm[lower_ind:upper_ind], color='#E69F00', linestyle='solid', linewidth=2.5)
    ax[2].plot(date_range, LH_era5[lower_ind:upper_ind], color='k', linestyle='dashed')
    
    ax[3].plot(date_range, DH_smacm[lower_ind:upper_ind], color='#E69F00', linestyle='solid', linewidth=2.5)
    ax[3].plot(date_range, DH_era5[lower_ind:upper_ind], color='k', linestyle='dashed')
    
    # aesthetics
    title_fs = 26
    axis_label_fs = 22
    axis_tick_fs = 20
    
    ## legend and title
    ax[0].legend(fontsize=axis_label_fs)
    ax[0].set_title("Southern Great Plains: Summer of {}".format(1979 + summer_ind), fontsize=title_fs)
    
    ## y labels
    ax[0].set_ylabel(r"Temperature ($^\circ$C)", fontsize=axis_label_fs)
    ax[1].set_ylabel("Soil Moisture Fraction (-)", fontsize=axis_label_fs)
    ax[2].set_ylabel(r"$\lambda \mathcal{E}$ (W m$^{-2}$)", fontsize=axis_label_fs)
    ax[3].set_ylabel(r"$\mathcal{H} + F_{LW}$ (W m$^{-2}$)", fontsize=axis_label_fs)
    
    ## x ticks
    ax[3].set_xticks(['1987-06-01', '1987-06-15', '1987-07-01', '1987-07-15', '1987-08-01', '1987-08-15', '1987-08-31'])
    ax[3].set_xticklabels(['June 1', 'June 15', 'July 1', 'July 15', 'August 1', 'August 15', 'August 31'], fontsize=axis_tick_fs)
    
    ## y ticks and x range
    for i in range(4):
        ax[i].tick_params(axis='y', labelsize=axis_tick_fs)
        ax[i].set_xlim((date_range.to_list()[0], date_range.to_list()[-1]))
    
    ## shaded region
    if loc_ind == 0:
        # set highlighted range of soil dessication
        beginning_ind = 47
        end_ind = beginning_ind + 21
        ax[0].axvspan(date_range[beginning_ind], date_range[end_ind], alpha=0.4, color='#009E73')
        ax[0].axvline(date_range[beginning_ind], 0, 1, color='#009E73')
        ax[0].axvline(date_range[end_ind], 0, 1, color='#009E73')
        
        ax[1].axvspan(date_range[beginning_ind], date_range[end_ind], alpha=0.4, color='#009E73')
        ax[1].axvline(date_range[beginning_ind], 0, 1, color='#009E73')
        ax[1].axvline(date_range[end_ind], 0, 1, color='#009E73')
        
        ax[2].axvspan(date_range[beginning_ind], date_range[end_ind], alpha=0.4, color='#009E73')
        ax[2].axvline(date_range[beginning_ind], 0, 1, color='#009E73')
        ax[2].axvline(date_range[end_ind], 0, 1, color='#009E73')
        
        ax[3].axvspan(date_range[beginning_ind], date_range[end_ind], alpha=0.4, color='#009E73')
        ax[3].axvline(date_range[beginning_ind], 0, 1, color='#009E73')
        ax[3].axvline(date_range[end_ind], 0, 1, color='#009E73')
    
    labels = ['A', 'B', 'C', 'D']
    for i in range(4):
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(0, 0.0, fig.dpi_scale_trans)
        ax[i].text(0.98, 0.96, labels[i], transform=ax[i].transAxes + trans, fontsize=title_fs, fontweight='bold',
                verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))
    
    fig.tight_layout()
    fig.savefig(figs_path + fig_name, dpi=400)

###########################
# PLOT MULTIPLE SUMMERS
###########################
else:
    # make the data pretty for multiple summers
    T_0 = 273.15  # conversion to C
    N_nans = 10
    DAYS_PER_SUMMER = 92
    N_days = np.shape(M_smacm)[0]
    N_total = int(N_nans * (N_days / DAYS_PER_SUMMER) + N_days)  # N_summers * N_nans + N_days
    
    M_model_aes = np.zeros(N_total)
    T_model_aes = np.zeros(N_total)
    LH_model_aes = np.zeros(N_total)
    DH_model_aes = np.zeros(N_total)
    
    M_raw_aes = np.zeros(N_total)
    T_raw_aes = np.zeros(N_total)
    LH_raw_aes = np.zeros(N_total)
    DH_raw_aes = np.zeros(N_total)
    
    i = 0
    old_ind = 0
    for n in range(N_total):
        # if i < days per summer, then we are in an interval where i want
        # the plotted data to be the model output
        if i < DAYS_PER_SUMMER:
            M_model_aes[n] = M_smacm[old_ind]
            T_model_aes[n] = T_smacm[old_ind] - T_0
            LH_model_aes[n] = LH_smacm[old_ind]
            DH_model_aes[n] = DH_smacm[old_ind]
    
            M_raw_aes[n] = ds_loc.m.values[old_ind]
            T_raw_aes[n] = ds_loc.t2m.values[old_ind] - T_0
            LH_raw_aes[n] = LH_era5[old_ind]
            DH_raw_aes[n] = DH_era5[old_ind]
            
            old_ind += 1
            i += 1
    
        # if we're in a place where we've exceeded the number of days in a summer,
        # but not yet exceeded days per summer plus the nans i want in between summers,
        # patch in nans
        elif i >= DAYS_PER_SUMMER and i < (N_nans + DAYS_PER_SUMMER):
            M_model_aes[n] = np.nan
            T_model_aes[n] = np.nan
            LH_model_aes[n] = np.nan
            DH_model_aes[n] = np.nan
    
            M_raw_aes[n] = np.nan
            T_raw_aes[n] = np.nan
            LH_raw_aes[n] = np.nan
            DH_raw_aes[n] = np.nan
            i += 1
    
            # once we violate both of the above (i.e., i > (N_nans + DAYS PER SUMMER)
            # reset i and keep going
    
            # doing it here is required to avoid a zero at the beginning of each summer
            if i == N_nans + DAYS_PER_SUMMER:
                i = 0


    # plot that shiz
    fig, ax = plt.subplots(4,1,figsize=[20,16], sharex=True)

    # aesthetics
    title_fs = 26
    axis_label_fs = 22
    axis_tick_fs = 20

    # interval to plot
    N_sums = 10
    low_sum = 3
    xlims = [low_sum * (92 + N_nans), (low_sum + N_sums) * (92 + N_nans) - N_nans]  # the -N_nans at the end trims off the last bit of grey
    
    ax[0].plot(T_raw_aes, label='ERA5', color='k')
    ax[0].plot(T_model_aes, label="SMACM", color='#E69F00')
    
    ax[1].plot(M_raw_aes, label='ERA5', color='k')
    ax[1].plot(M_model_aes, label='SMACM', color='#E69F00')
    
    ax[2].plot(LH_raw_aes, color='k')
    ax[2].plot(LH_model_aes, color='#E69F00')
    
    ax[3].plot(DH_raw_aes, color='k')
    ax[3].plot(DH_model_aes, color='#E69F00')
    
    # x tick locations and labels
    labels = [(1979 + low_sum + i) for i in range(N_sums)]
    ax[3].set_xticks([46 + (92 + N_nans) * (i + low_sum) for i in range(N_sums)])
    ax[3].set_xticklabels(labels)
    ax[3].tick_params(axis='x', labelsize=axis_tick_fs)
    
    ax[1].set_ylabel('m (--)', fontsize=axis_label_fs)
    ax[1].set_xlim(xlims)
    ax[1].set_ylim(0, 1)
    
    ax[0].set_ylabel('Temperature ($^{\circ}$C)', fontsize=axis_label_fs)
    ax[0].set_xlim(xlims)
    ax[0].legend(fontsize=axis_tick_fs)
    
    ax[2].set_xlim(xlims)
    ax[2].set_ylabel('$\lambda \mathcal{E}$ (W m$^{-2}$)', fontsize=axis_label_fs)
    ax[2].set_ylim(-25,200)
    
    ax[3].set_xlim(xlims)
    ax[3].set_ylim(-25,200)
    ax[3].set_ylabel('$\mathcal{H} + F_{LW}$ (W m$^{-2}$)', fontsize=axis_label_fs)
    
    # put black spaces in between summers
    for i in range(1, N_sums):
        i += low_sum
        ax[0].axvspan(92 * i + N_nans * (i - 1), 92 * i + N_nans * i, color='k', alpha=0.5)
        ax[1].axvspan(92 * i + N_nans * (i - 1), 92 * i + N_nans * i, color='k', alpha=0.5)
        ax[2].axvspan(92 * i + N_nans * (i - 1), 92 * i + N_nans * i, color='k', alpha=0.5)
        ax[3].axvspan(92 * i + N_nans * (i - 1), 92 * i + N_nans * i, color='k', alpha=0.5)
    
    if loc_ind == 0:
        ax[0].set_title("Southern Great Plains", fontsize=26)
    
    if loc_ind == 2:
        ax[0].set_title("Washington DC", fontsize=26)
    
    if loc_ind == 3:
        ax[0].set_title("Atlanta, GA", fontsize=26)
        
    labels = ['A', 'B', 'C', 'D']
    for i in range(4):
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(0, 0.0, fig.dpi_scale_trans)
        ax[i].text(0.98, 0.96, labels[i], transform=ax[i].transAxes + trans, fontsize=title_fs, fontweight='bold',
                verticalalignment='top', bbox=dict(facecolor='none', edgecolor='none', pad=1))
    
    fig.tight_layout()
    fig.savefig(figs_path + fig_name, dpi=400)

print("Figure saved successfully!")