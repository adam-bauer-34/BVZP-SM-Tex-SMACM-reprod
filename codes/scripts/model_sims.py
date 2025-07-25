# import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import xarray as xr

from scripts.src.locations.ATL import ATL
from scripts.src.locations.DC import DC
from scripts.src.locations.SGP import SGP

from data_locs import reanal_filename, sims_loc
from lat_longs import loc_dict

from scripts.src.time_series_helper import *

# 20 variables for the lower 48 states: ERA5_gater_regional_data.ipynb
# daily JJA data 1979-2021
ds=xr.open_dataset(reanal_filename).load()

# do sgp
loc_ind = 0
SGP = SGP()

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
M_smacm_sgp, T_smacm, LH_smacm, DH_smacm = get_SMACM_integration_fullcoupled(SGP, ds_loc)

m_ds = xr.Dataset(data_vars={"m": (["time"], M_smacm_sgp),
                                     },
                           coords={"time": (["time"], ds.time.data),})

m_filename = ''.join([sims_loc, "m-hasselmann-sgp.nc"])

m_ds.to_netcdf(path=m_filename, mode="w", format="NETCDF4", engine="netcdf4")
print("SGP data saved.")

# do atl
loc_ind = 3
ATL = ATL()
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
M_smacm_atl, T_smacm, LH_smacm, DH_smacm = get_SMACM_integration_fullcoupled(ATL, ds_loc)
m_ds = xr.Dataset(data_vars={"m": (["time"], M_smacm_atl),
                                     },
                           coords={"time": (["time"], ds.time.data),})

m_filename = ''.join([sims_loc, "m-hasselmann-atl.nc"])
m_ds.to_netcdf(path=m_filename, mode="w", format="NETCDF4", engine="netcdf4")
print("ATL data saved.")

# do dc
loc_ind = 2
DC = DC()
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
M_smacm_dc, T_smacm, LH_smacm, DH_smacm = get_SMACM_integration_fullcoupled(DC, ds_loc)
m_ds = xr.Dataset(data_vars={"m": (["time"], M_smacm_dc),
                                     },
                           coords={"time": (["time"], ds.time.data),})

m_filename = ''.join([sims_loc, "m-hasselmann-dc.nc"])

m_ds.to_netcdf(path=m_filename, mode="w", format="NETCDF4", engine="netcdf4")
print("DC data saved.")


print("Sims successful, data written to {}".format(sims_loc))