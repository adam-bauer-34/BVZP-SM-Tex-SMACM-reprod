# import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.linear_model import LinearRegression
from scipy import stats
import numpy as np
import xarray as xr
import pandas as pd

from fig_locs import figs_path
from data_locs import reanal_filename
from lat_longs import loc_dict

# 20 variables for the lower 48 states: ERA5_gater_regional_data.ipynb
# daily JJA data 1979-2021
ds=xr.open_dataset(reanal_filename).load()

inds = [0,1,3,4,2,7,8] # location indexes
ndayss = [1,3,5,7,14] # days to average over

rs = np.zeros((len(inds), len(ndayss)))
a_s = np.zeros((len(inds), len(ndayss)))
int_s = np.zeros((len(inds), len(ndayss)))
loc_for_df = []

ind_i = 0
days_i = 0

for ind in inds:
    lat=loc_dict[ind]['lat']
    lon=loc_dict[ind]['lon']
    location=loc_dict[ind]['location']
    loc_for_df.append(location)

    #extract data for one location
    ds_loc=ds.sel(latitude=lat,longitude=lon,method='nearest')
    m=(ds_loc.swvl1-ds_loc.swvl1.min())/(ds_loc.swvl1.max()-ds_loc.swvl1.min())
    ds_loc=ds_loc.assign(m=m)
    for ndays in ndayss:
        lw=-ds_loc.sshf.rolling(time=ndays,center=True).mean().dropna(dim='time')/3600
        sens=-ds_loc.str.rolling(time=ndays,center=True).mean().dropna(dim='time')/3600

        total = (lw + sens).values
        T = ds_loc.skt.rolling(time=ndays,center=True).mean().dropna(dim='time') - 273.15
        Td = ds_loc.d2m.rolling(time=ndays,center=True).mean().dropna(dim='time')- 273.15
        
        temp_data = (T-Td).values
        
        temp_data_reshaped = temp_data.reshape(-1,1)

        reg = LinearRegression().fit(temp_data_reshaped, total)
        
        rs[ind_i, days_i] = reg.score(temp_data_reshaped, total)
        a_s[ind_i, days_i] = reg.coef_[0]
        int_s[ind_i, days_i] = reg.intercept_
        days_i +=1
    
    ind_i += 1
    days_i = 0

df = pd.DataFrame(rs, index=loc_for_df, columns=ndayss)

print(df)