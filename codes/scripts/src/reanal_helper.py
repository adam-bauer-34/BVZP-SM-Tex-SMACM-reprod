import os
import sys

anal_type = sys.argv[1]  # which analysis (tmax, t2m, or detrended)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib as mpl 
from scipy.signal import detrend

from fig_locs import figs_path
from data_locs import reanal_filename
from lat_longs import loc_dict

# plotting lists
color_list = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'] * 4
marker_list = ['o', 's', 'P', '*', '<', 'v', '>', 'p'] * 4
linestyle_list = ['solid'] * 8 + ['dashdot'] * 8 + ['dashed'] * 8 + ['dotted'] * 8

# dictionaries of params
cdds_params={'axes.linewidth': 3,
 'axes.axisbelow': False,
 'axes.edgecolor': 'black',
 'axes.facecolor': 'None',
 'axes.grid': False,
 'axes.labelcolor': 'black',
 'axes.spines.right': False,
 'axes.spines.top': False,
 'axes.titlesize': 24,
 'axes.labelsize': 20,
 'axes.titlelocation': 'left',
 'figure.facecolor': 'white',
 'figure.figsize': (18, 10),
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

def make_column(fig, ax, panels, ds, ind, anal_type, center=False, left=False):
    # unpack panels
    scatter_panel, kde_panel, kde_shift_panel, std_panel = panels 
    
    # get location
    lat=loc_dict[ind]['lat']
    lon=loc_dict[ind]['lon']
    location=loc_dict[ind]['location']
    ds_loc = ds.sel(latitude=lat,longitude=lon,method='nearest')
    
    # extract interesting variables
    if anal_type == 't2m':
        temp=ds_loc.t2m

    elif anal_type == 't2max':
        temp=ds_loc.t2max

    else:
        raise ValueError("Invalid analysis type parameter.")
        
    soil=ds_loc.swvl1
    dew = ds_loc.d2m

    solar=ds_loc.ssr/3600
    
    # Temperature percentiles
    t99= np.percentile(temp,99)
    t95= np.percentile(temp,95)

    # Compute 10 soil moisture quantiles
    ql=np.percentile(soil,np.arange(0,100,10))
    qu=np.percentile(soil,np.arange(5,105,10))

    # compute the soil moisture mean, temperature mean and temperature standard deviation in each quantile
    temp_mean=np.zeros(len(ql))
    temp_sd  =np.zeros(len(ql))
    soil_mean=np.zeros(len(ql))

    for j in range(len(ql)):
        temp_mean[j]=temp.where(soil>ql[j]).where(soil<qu[j]).mean()
        temp_sd  [j]=temp.where(soil>ql[j]).where(soil<qu[j]).std()
        soil_mean[j]=soil.where(soil>ql[j]).where(soil<qu[j]).mean()
        
    # soil moisture calculations
    soil20 = np.percentile(soil, 20)
    total_sm_vals = np.shape(soil[temp>t95])[0]
    total_low_sm_vals = np.shape(np.where(soil[temp>t95] < soil20))[1]

    # perent shifts
    perc = (total_sm_vals - total_low_sm_vals)/total_sm_vals
    perc = 1 - perc

    # kde panels
    xx = np.linspace(280, 320, 1000)
    kde_all = stats.gaussian_kde(temp)

    # select 10 soil-moisture quantiles. 
    ql=np.percentile(soil, np.arange(0,100,10))
    qu=np.percentile(soil, np.arange(5,105,10))

    # estimate the pdf of temperature in each soil moisture quantile
    for j in range(len(ql)):
        #select temperatures in soil-moisture quantile
        y=np.asarray(temp.where(soil>ql[j]).where(soil<qu[j]))
        y=y[~np.isnan(y)]
        # estimate pdf using gaussian kernel density estimation
        kde = stats.gaussian_kde(y)

        # plot 
        xx = np.linspace(280, 320, 1000)
        shade=j/len(ql)    
        ax[kde_panel].plot(xx - 273.15,kde(xx),color=[0.5,shade,shade], linestyle='solid')
        ax[kde_panel].set_xlim(np.median(temp)- 15 - 273.15,np.median(temp) + 15 - 273.15)

    ax[kde_panel].plot(xx- 273.15,kde_all(xx),color='k',linewidth=5)
    ax[kde_panel].set_xlabel("Temperature ($^{\circ}$C)", fontsize=22)
    ax[scatter_panel].set_title(location)

    # redo the calculation, but now remove the sample mean    
    for j in range(len(ql)):
        y=np.asarray(temp.where(soil>ql[j]).where(soil<qu[j]))
        y=y[~np.isnan(y)]
        kde = stats.gaussian_kde((y-np.mean(y)),bw_method=0.5)
        xx = np.linspace(-20, 20, 1000)
    
        g=j/len(ql)    
        ax[kde_shift_panel].plot(xx,kde(xx),color=[0.5,g,g], linestyle='solid')
        ax[kde_shift_panel].set_xlabel('$T - \mu_{T}$ ($^{\circ}$C)', fontsize=22)
        ax[kde_shift_panel].set_xlim(-15,+15)
        
    # plot soil-moisture and temperature
    #ax[0].set_title(location, fontsize=22)
    ax[scatter_panel].scatter(soil[temp<t95], temp[temp<t95]-273.15, marker='o', color='k')
    tmps = soil[temp>t95]
    tmpt = temp[temp>t95]
    ax[scatter_panel].scatter(tmps[t99>tmpt], tmpt[t99>tmpt]-273.15, marker='+', 
                              color='#D55E00',label='$T > 95$%')
    ax[scatter_panel].scatter(soil[temp>t99], temp[temp>t99]-273.15, marker='x', 
                              color='#009E73',label='$T > 99$%')
    ax[scatter_panel].plot(soil_mean,temp_mean-273.15,
                           label='$\mu_T$ by soil moisture decile', color='#E69F00', 
                           linewidth=3, marker='o', linestyle='solid')
    # plot soil moisture quintiles
    soil_qs= np.percentile(soil,[20,40,60,80])
    ax[scatter_panel].vlines(soil_qs, np.min(temp)-273.15,np.max(temp)-273.15,
                             label='Soil moisture quintiles',color='#CC79A7')

    # plot temperature standard deviation
    ax[std_panel].plot(soil_mean, temp_sd, color=color_list[7], marker='o', 
                       label= '$\sigma_T$ by soil moisture decile')

    if anal_type == 't2m':
        print("Quoted numbers:")
        print("Relative temperature standard deviations to average:")
        print((temp_sd / np.mean(temp_sd)) - 1)
        
    ax[std_panel].set_xlim(np.min(soil), np.max(soil))
    # ax[std_panel].set_ylim(1,4)
    ax[scatter_panel].set_xlabel('Vol. Soil Moisture (cm$^3$/cm$^3$)', fontsize=22)
    ax[std_panel].set_xlabel('Vol. Soil Moisture (cm$^3$/cm$^3$)', fontsize=22)
    if left:
        ax[scatter_panel].set_ylabel('Temperature ($^{\circ}$C)', fontsize=22)
        ax[std_panel].set_ylabel('$\sigma_{T}$ ($^{\circ}$C)', fontsize=22)
        ax[kde_panel].set_ylabel('PDF', fontsize=22)
        ax[kde_shift_panel].set_ylabel('PDF', fontsize=22)

    if center:
        #ax[std_panel].legend(loc='upper right', bbox_to_anchor=(-0.21, -.09), fontsize=20)
        ax[scatter_panel].legend(loc='center', bbox_to_anchor=(-0.7, 1.2), ncol=4, fontsize=20)


def make_column_detrend(fig, ax, panels, ds, ind, center=False, left=False):
    
    # unpack panels
    scatter_panel, kde_panel, kde_shift_panel, std_panel = panels 
    
    # get location
    lat=loc_dict[ind]['lat']
    lon=loc_dict[ind]['lon']
    location=loc_dict[ind]['location']
    ds_loc = ds.sel(latitude=lat,longitude=lon,method='nearest')
    
    # extract interesting variables
    temp_trend = ds_loc.t2m.values
    temp_dt = detrend(temp_trend, type='linear')  # detrend temperature data
    ds_temp = xr.Dataset(data_vars={'temp_dt': (['time'], temp_dt)},
                         coords={'time': (['time'], ds_loc.time.data)})
    temp_dt = ds_temp.temp_dt
    soil=ds_loc.swvl1

    solar=ds_loc.ssr/3600
    
    # Temperature percentiles
    t99= np.percentile(temp_dt,99)
    t95= np.percentile(temp_dt,95)

    # Compute 10 soil moisture quantiles
    ql=np.percentile(soil,np.arange(0,100,10))
    qu=np.percentile(soil,np.arange(5,105,10))

    # compute the soil moisture mean, temperature mean and temperature standard deviation in each quantile
    temp_mean=np.zeros(len(ql))
    temp_sd  =np.zeros(len(ql))
    soil_mean=np.zeros(len(ql))

    for j in range(len(ql)):
        temp_mean[j]=temp_dt.where(soil>ql[j]).where(soil<qu[j]).mean()
        temp_sd  [j]=temp_dt.where(soil>ql[j]).where(soil<qu[j]).std()
        soil_mean[j]=soil.where(soil>ql[j]).where(soil<qu[j]).mean()
        
    # soil moisture calculations
    soil20 = np.percentile(soil, 20)
    total_sm_vals = np.shape(soil[temp_dt>t95])[0]
    total_low_sm_vals = np.shape(np.where(soil[temp_dt>t95] < soil20))[1]
    
    # perent shifts
    perc = (total_sm_vals - total_low_sm_vals)/total_sm_vals
    perc = 1 - perc

    # kde panels
    xx = np.linspace(-10, 10, 1000)
    kde_all = stats.gaussian_kde(temp_dt)

    # select 10 soil-moisture quantiles. 
    ql=np.percentile(soil,np.arange(0,100,10))
    qu=np.percentile(soil,np.arange(5,105,10))

    # estimate the pdf of temperature in each soil moisture quantile
    for j in range(len(ql)):
        #select temperatures in soil-moisture quantile
        y=np.asarray(temp_dt.where(soil>ql[j]).where(soil<qu[j]))
        y=y[~np.isnan(y)]
        # estimate pdf using gaussian kernel density estimation
        kde = stats.gaussian_kde(y)

        # plot 
        xx = np.linspace(-10, 10, 1000)
        shade=j/len(ql)    
        ax[kde_panel].plot(xx, kde(xx),color=[0.5,shade,shade], linestyle='solid')
        # ax[kde_panel].set_xlim(np.median(temp)- 15 - 273.15,np.median(temp) + 15 - 273.15)

    ax[kde_panel].plot(xx, kde_all(xx),color='k',linewidth=5)
    ax[kde_panel].set_xlabel("Temperature Anomaly ($^{\circ}$C)", fontsize=22)
    ax[scatter_panel].set_title(location)

    # redo the calculation, but now remove the sample mean    
    for j in range(len(ql)):
        y=np.asarray(temp_dt.where(soil>ql[j]).where(soil<qu[j]))
        y=y[~np.isnan(y)]
        kde = stats.gaussian_kde((y-np.mean(y)),bw_method=0.5)
        xx = np.linspace(-20, 20, 1000)
    
        g=j/len(ql)    
        ax[kde_shift_panel].plot(xx, kde(xx), color=[0.5,g,g], linestyle='solid')
        ax[kde_shift_panel].set_xlabel('$T_{anom} - \mu_{T_{anom}}$ ($^{\circ}$C)', fontsize=22)
        ax[kde_shift_panel].set_xlim(-15,+15)
        
    # plot soil-moisture and temperature
    #ax[0].set_title(location, fontsize=22)
    ax[scatter_panel].scatter(soil[temp_dt<t95], temp_dt[temp_dt<t95], marker='o', color='k')
    tmps = soil[temp_dt>t95]
    tmpt = temp_dt[temp_dt>t95]
    ax[scatter_panel].scatter(tmps[t99>tmpt], tmpt[t99>tmpt], marker='+', 
                              color='#D55E00',label='$T > 95$%')
    ax[scatter_panel].scatter(soil[temp_dt>t99], temp_dt[temp_dt>t99], marker='x', 
                              color='#009E73',label='$T > 99$%')
    ax[scatter_panel].plot(soil_mean,temp_mean,
                           label='$\mu_T$ by soil moisture decile', color='#E69F00', 
                           linewidth=3, marker='o', linestyle='solid')
    # plot soil moisture quintiles
    soil_qs= np.percentile(soil,[20,40,60,80])
    ax[scatter_panel].vlines(soil_qs, np.min(temp_dt),np.max(temp_dt),
                             label='Soil moisture quintiles',color='#CC79A7')

    # plot temperature standard deviation
    ax[std_panel].plot(soil_mean, temp_sd, color=color_list[7], marker='o', 
                       label= '$\sigma_T$ by soil moisture decile')

    ax[std_panel].set_xlim(np.min(soil), np.max(soil))
    # ax[std_panel].set_ylim(1,4)
    ax[scatter_panel].set_xlabel('Vol. Soil Moisture (cm$^3$/cm$^3$)', fontsize=22)
    ax[std_panel].set_xlabel('Vol. Soil Moisture (cm$^3$/cm$^3$)', fontsize=22)
    if left:
        ax[scatter_panel].set_ylabel('Temperature Anomaly ($^{\circ}$C)', fontsize=22)
        ax[std_panel].set_ylabel('$\sigma_{T}$ ($^{\circ}$C)', fontsize=22)
        ax[kde_panel].set_ylabel('PDF', fontsize=22)
        ax[kde_shift_panel].set_ylabel('PDF', fontsize=22)
        

    if center:
        #ax[std_panel].legend(loc='upper right', bbox_to_anchor=(-0.21, -.09), fontsize=20)
        ax[scatter_panel].legend(loc='center', bbox_to_anchor=(-0.7, 1.2), ncol=4, fontsize=20)