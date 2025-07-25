"""Compute lagged correlation coefficient.

Adam Michael Bauer
Univ. of Chicago
7.25.2025
"""

import os
import sys

loc = sys.argv[1]  # which location (sgp, atl, or dc)
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

# These are some parameters to make figures nice (and big)
plt.rcParams['figure.figsize'] = 12, 6
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

# 20 variables for the lower 48 states: ERA5_gater_regional_data.ipynb
# daily JJA data 1979-2021
ds=xr.open_dataset(reanal_filename).load()

# get location index
if loc == 'sgp':
    ind = 0
    fig_name = 's04-lag_coeff_sgp.png'

elif loc == 'atl':
    ind = 3
    fig_name = 's05-lag_coeff_atl.png'

elif loc == 'dc':
    ind = 2
    fig_name = 's06-lag_coeff_dc.png'

# extract location
lat=loc_dict[ind]['lat']
lon=loc_dict[ind]['lon']
location=loc_dict[ind]['location']

# get that location's data
ds_loc=ds.sel(latitude=lat,longitude=lon,method='nearest')

# select vars
dew = ds_loc.d2m
temp = ds_loc.t2m

# define helper function
from scipy.signal import detrend
def lag_regress(x,y,N):
    z=np.zeros(2*N+1)
    for j in range(-N,N+1):
        x1=x.shift({'time':j})
        y1=y.sel(time=x.time[~x1.isnull()])
        x1=x1.sel(time=x1.time[~x1.isnull()])
        r = np.polyfit(detrend(x1), detrend(y1), 1)
        z[j]=r[0]
    lags=np.arange(-N,N+1)    
    z=np.concatenate([z[N+1:],z[0:N+1]])
    return z,lags

# do lagged regression
reg, lag=lag_regress(dew, temp, 30)

# make figure
fig, ax = plt.subplots()

# add error line
sigma = np.sqrt(91 * (2021 - 1979))**(-1)  # 1/sqrt(N_samples) is sample variance
ax.hlines(2 * sigma, -30, 30, color='k', linestyle='dashed')
ax.hlines(-2 * sigma, -30, 30, color='k', linestyle='dashed', label=r'$\pm 2\sigma$ confidence interval')

ax.plot(lag, reg, '.-', color='blue', label='Lag correlation')
ax.set_ylabel("Lag coefficient")
ax.set_xlabel("Lag (days)")
ax.grid()
ax.set_xlim((-30, 30))
ax.legend()

fig.tight_layout()

fig.savefig(figs_path + fig_name, dpi=400)
print("Figure saved successfully!")