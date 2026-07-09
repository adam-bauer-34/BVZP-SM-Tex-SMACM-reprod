"""Model fit at different locations.

Adam Michael Bauer
Univ. of Chicago
7/25/2025
"""

# import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import xarray as xr
import pandas as pd

from scripts.src.locations.ATL import ATL
from scripts.src.locations.DC import DC
from scripts.src.locations.SGP import SGP
from scripts.src.locations.DAL import DAL
from scripts.src.locations.SEA import SEA
from scripts.src.locations.WIT import WIT
from scripts.src.locations.MIN import MIN
from scripts.src.var_exp_helper import calc_variance_ratio, calc_ext_ratio

from data_locs import reanal_filename
from lat_longs import loc_dict

# 20 variables for the lower 48 states: ERA5_gater_regional_data.ipynb
# daily JJA data 1979-2021
ds=xr.open_dataset(reanal_filename).load()

final_tab_values = []
tab_labels = ['eta', 't0', 'mbar', 'var_m', 'var_t_m/var_t', 't0/tx']
inds = [0,1,3,4,2,7,8] # location indexes
locs = []

for ind in inds:
    lat=loc_dict[ind]['lat']
    lon=loc_dict[ind]['lon']
    location=loc_dict[ind]['location']
    locs.append(location)

    if ind == 0:
        LOC = SGP()

    elif ind == 1:
        LOC = WIT()

    elif ind == 2:
        LOC = DC()

    elif ind == 3:
        LOC = ATL()

    elif ind == 4:
        LOC = DAL()

    elif ind == 7:
        LOC = MIN()

    elif ind == 8:
        LOC = SEA()

    ds_loc = ds.sel(latitude=lat, longitude=lon, method='nearest')
    sm = ds_loc.swvl1
    t = ds_loc.t2m

    eta, T0, mbar, m_var, t_var_m = calc_variance_ratio(sm, LOC)

    T0_TX = calc_ext_ratio(t, LOC)

    final_tab_values.append([eta, T0, mbar, m_var, t_var_m/np.var(t.values), T0_TX])

df = pd.DataFrame(final_tab_values, index=locs, columns=tab_labels)
print(df)

