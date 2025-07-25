"""Helper functions for Figure 5, phase space.

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

import matplotlib as mpl 
from lat_longs import loc_dict


def get_Td_difference_Td(anom_size, t99, m, place):
    
    """
    Of use constants.
    """
    L = 2.5e6
    C = 4180 
    P_surf = 101325
    R_w = 461.52
    T_0 = 273.15 

    if place == "SGP":
        """
        Model parameters. (SGP.)
        """
        alpha_s = 6.74
        alpha_r = 0
        nu = 0.0058
        mu = 70 * 0.75
        m_0 = 0.
        F_mean = 226.96
        Td_mean = 291.55
        Td_std = 2.75
        off = 39.1 + 79.2 + F_mean * 0.093
        #gamma = 0.00084
    
    if place == "DC":
        alpha_s=11.3
        alpha_r=0
        nu   =0.0078
        mu   =70 * 0.7
        V    =3.61
        m_0   =0.0
        F_mean = 205.1
        Td_mean = 290.7
        Td_std = 3.51
        off = 23.8 + 45.1 + 0.05 * F_mean
        #gamma = 0.00081
        
    if place == "ATL":
        alpha_s = 0
        alpha_r = 11.46
        nu = 0.0089
        mu = 70 * 0.7
        m_0 = 0.
        F_mean = 204.61
        Td_mean = 292.72
        Td_std = 2.67
        off = 24.9 + 60.6 + F_mean * 0.059
        #gamma = 0.0009
    
    factor1 = 611 # Pa
    factor2 = 0.622 # -
    
    Td_1 = anom_size * Td_std + Td_mean
    
    exp_arg1 = L * R_w**(-1) * (T_0**(-1) - Td_1**(-1))
    prefactor1 = factor1 * factor2 * L * (R_w * P_surf * Td_1**2)**(-1)
    gamma1 = prefactor1 * np.exp(exp_arg1)
    
    null1 = Td_1 + (F_mean - off) * (alpha_s + alpha_r + L * nu * gamma1 * (m_0 + m))**(-1)
    
    diff = t99 - null1 
    return diff

def find_anomaly_size_Td(loc_ind, t99, m):
    if loc_ind == 0:
        anom_size = [root_scalar(get_Td_difference_Td, args=(t99, m[i], "SGP"), bracket=[0,10], method='brentq').root for i in range(len(m))]
    
    elif loc_ind == 3:
        anom_size = [root_scalar(get_Td_difference_Td, args=(t99, m[i], "ATL"), bracket=[0,10], method='brentq').root for i in range(len(m))]
        
    elif loc_ind == 8:
        anom_size = [root_scalar(get_Td_difference_Td, args=(t99, m[i], "SEA"), bracket=[0,10], method='brentq').root for i in range(len(m))]
    
    elif loc_ind == 2:
        anom_size = [root_scalar(get_Td_difference_Td, args=(t99, m[i], "DC"), bracket=[0,10], method='brentq').root for i in range(len(m))]
        
    return anom_size

def get_Td_difference_F(anom_size, t99, m, place):
    
    """
    Of use constants.
    """
    L = 2.5e6
    C = 4180 
    P_surf = 101325
    R_w = 461.52
    T_0 = 273.15 

    if place == "SGP":
        """
        Model parameters. (SGP.)
        """
        alpha_s = 6.74
        alpha_r = 0
        nu = 0.0058
        mu = 70 * 0.75
        F_std = 44.53
        m_0 = 0.
        F_mean = 226.96
        Td_mean = 291.55
        Td_std = 2.75
        off = 39.1 + 79.2 + F_mean * 0.093
        #gamma = 0.00084
    
    if place == "DC":
        alpha_s=11.3
        alpha_r=0
        nu   =0.0078
        mu   =70 * 0.7
        V    =3.61
        m_0   =0.0
        F_mean = 205.1
        Td_mean = 290.7
        Td_std = 3.51
        F_std = 48.95
        off = 23.8 + 45.1 + 0.05 * F_mean
        #gamma = 0.00081
        
    if place == "ATL":
        alpha_s = 0
        F_std = 49.87
        alpha_r = 11.46
        nu = 0.0089
        mu = 70 * 0.7
        m_0 = 0.
        F_mean = 204.61
        Td_mean = 292.72
        Td_std = 2.67
        off = 24.9 + 60.6 + F_mean * 0.059
        #gamma = 0.0009
    
    factor1 = 611 # Pa
    factor2 = 0.622 # -
    
    F_1 = anom_size * F_std + F_mean
    
    exp_arg1 = L * R_w**(-1) * (T_0**(-1) - Td_mean**(-1))
    prefactor1 = factor1 * factor2 * L * (R_w * P_surf * Td_mean**2)**(-1)
    gamma1 = prefactor1 * np.exp(exp_arg1)
    
    null1 = Td_mean + (F_1-off) * (alpha_s + alpha_r + L * nu * gamma1 * (m_0 + m))**(-1)
    
    diff = t99 - null1 
    return diff

def find_anomaly_size_F(loc_ind, t99, m):
    if loc_ind == 0:
        anom_size = [root_scalar(get_Td_difference_F, args=(t99, m[i], "SGP"), bracket=[0,40], method='brentq').root for i in range(len(m))]
    
    elif loc_ind == 3:
        anom_size = [root_scalar(get_Td_difference_F, args=(t99, m[i], "ATL"), bracket=[0,40], method='brentq').root for i in range(len(m))]
        
    elif loc_ind == 2:
        anom_size = [root_scalar(get_Td_difference_F, args=(t99, m[i], "DC"), bracket=[0,40], method='brentq').root for i in range(len(m))]
    
    elif loc_ind == 11:
        anom_size = [root_scalar(get_Td_difference_F, args=(t99, m[i], "NY"), bracket=[0,40], method='brentq').root for i in range(len(m))]
        
    return anom_size

def make_column(fig, ax, loc_ind, panels, data, right, center, left, anom_type, sim_path):
    ### color list ###
    cl = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    
    ### separate out panels ###
    ps_panel, prob_panel = panels

    ### select data ###
    lat=loc_dict[loc_ind]['lat']
    lon=loc_dict[loc_ind]['lon']
    location=loc_dict[loc_ind]['location']

    #extract locations' data
    ds_loc=data.sel(latitude=lat,longitude=lon,method='nearest')
    
    ### Extract a few variables: soil moisture, temperature, dew point temperature, solar radiation ###
    temp=ds_loc.t2m
    soil=ds_loc.swvl1
    dew = ds_loc.d2m
    Td_std = np.std(dew)
    #print(Td_std)

    solar=ds_loc.ssr/3600
    
    ### Compute temperature percentiles ###
    t99= np.percentile(temp,99)
    t95= np.percentile(temp,95)
    
    ### Compute 10 soil moisture quantiles ###
    ql=np.percentile(soil,np.arange(0,100,10))
    qu=np.percentile(soil,np.arange(5,105,10))
    
    ### Compute the soil moisture mean, temperature mean and temperature standard deviation in each quantile ###
    temp_mean=np.zeros(len(ql))
    temp_sd  =np.zeros(len(ql))
    soil_mean=np.zeros(len(ql))

    for j in range(len(ql)):
        temp_mean[j]=temp.where(soil>ql[j]).where(soil<qu[j]).mean()
        temp_sd  [j]=temp.where(soil>ql[j]).where(soil<qu[j]).std()
        soil_mean[j]=soil.where(soil>ql[j]).where(soil<qu[j]).mean()
        
    ### Make nullcline ###
    # make sw dist and soil range
    Theta = np.linspace(np.min(soil.values), np.max(soil.values), 10**4)
    m = (Theta - np.min(Theta)) / (np.max(Theta) - np.min(Theta))
    L = 2.5e6
    P_surf = 101325 # Pa
    R_w = 461.52 #J/kg
    T_0 = 273.15 # K
    solar_mean = np.mean(solar).values
    F_std = np.std(solar).values
    Td_mean = np.mean(dew).values
    Td_std = np.std(dew).values
    
    # load model paramters
    if loc_ind == 0:
        alpha_s = 6.74
        alpha_r = 0
        nu = 0.0058
        mu = 70 * 0.75
        m0 = 0.
        F_mean = 226.96
        Td_mean = 291.55
        #Td_std = 2.75
        off = 39.1 + 79.2 + F_mean * 0.093
        gamma = 0.00084
        
        # impot simulated Hasselmann model to get soil moisture dist
        m_ds = xr.open_dataset(sim_path + 'm-hasselmann-sgp.nc')
        
    if loc_ind == 3:
        # fitted parameters for Atlanta, GA
        alpha_s = 0
        alpha_r = 11.46
        nu = 0.0089
        mu = 70 * 0.7
        m0 = 0.
        F_mean = 204.61
        Td_mean = 292.72
        #Td_std = 2.67
        off = 24.9 + 60.6 + F_mean * 0.059
        gamma = 0.0009
        
        # impot simulated Hasselmann model to get soil moisture dist
        m_ds = xr.open_dataset(sim_path + 'm-hasselmann-atl.nc')
    
    if loc_ind == 2:
        # fitted parameters for DC (calculated from model_fit.ipynb) 
        alpha_s=11.3
        alpha_r=0
        nu   =0.0078
        mu   =70 * 0.7
        V    =3.61
        m0   =0.0
        F_mean = 205.1
        Td_mean = 290.7
        #Td_std = 3.51
        off = 23.8 + 45.1 + 0.05 * F_mean
        gamma = 0.00081
        
        # impot simulated Hasselmann model to get soil moisture dist
        m_ds = xr.open_dataset(sim_path + 'm-hasselmann-dc.nc')
      
    # make nullcline
    T_nullcline = Td_mean + (F_mean-off) * (alpha_s + alpha_r + nu * m0 * L * gamma + L * gamma * nu * m)**(-1)
    
    # make offset nullclines
    if anom_type == 'F':
        N_anoms = 3
        F_dist = np.asarray([F_mean - off + i * F_std for i in range(1, N_anoms)]) 

        # offset nullclines
        nullcline_offset = Td_mean + (F_dist[:, None]) * (alpha_s + alpha_r + nu * m0 * L * gamma  + L * gamma * nu * m)**(-1)

    elif anom_type == 'Td':   
        N_anoms = 4
        T_d_dist = np.asarray([Td_mean + i * Td_std for i in range(1, N_anoms)])

        # corresponding offset gammas
        prefactor_dist = 0.622 * L * (P_surf * R_w * T_d_dist**2)**(-1) * 611
        exp_term_dist = np.exp(L * R_w**(-1) * (T_0**(-1) - T_d_dist**(-1) ))
        gamma_dist = prefactor_dist * exp_term_dist
        
        # offset nullclines
        nullcline_offset = T_d_dist[:, None] + (F_mean-off) * (alpha_s + alpha_r + nu * m0 * L * gamma_dist[:, None]  + L * gamma_dist[:, None] * nu * m)**(-1)

    ### Phase space panel ###
    ax[ps_panel].plot(soil          ,temp-273.15          ,'.', color='k', alpha=0.075)
    ax[ps_panel].plot(soil[temp>t95],temp[temp>t95]-273.15,'.', color='k', alpha=0.075)
    ax[ps_panel].plot(soil[temp>t99],temp[temp>t99]-273.15,'.', color='#D55E00', alpha=0.075)
    ax[ps_panel].plot(soil_mean,temp_mean-273.15,label='$\mu_T$ by soil moisture quantile', 
               color='#E69F00', linewidth=4, marker='o', linestyle='solid')
    ax[ps_panel].plot(Theta, T_nullcline-273.15, color=cl[3], linewidth=4, label='SMACM Mean')
    ax[ps_panel].hlines(t99-273.15, min(Theta), max(Theta), color=cl[2], 
                 linestyle='dashdot', linewidth=3, label="$T_{99}$", zorder=40)

    ### put histo on bottom
    ylow = np.min(temp).values -1 - 273.15
    #yhigh = 42
    #ax[ps_panel].set_ylim((ylow, yhigh))
    ax[ps_panel].hist(soil, bins=20, weights=np.ones_like(soil.values) * 0.015, alpha=0.5, color=cl[6], 
                      bottom=ylow, label='ERA5 Soil Moisture')
    ax[ps_panel].hist(m_ds.m * (max(soil) - min(soil)) + min(soil), bins=20, weights=np.ones_like(m_ds.m) * 0.015, 
               alpha=0.5, color=cl[5], bottom=ylow, label='Hasselmann Model')

    fillc = cl[5]
    alphas = np.arange(0,0.6,0.125)[::-1]
    for i in range(N_anoms-1):
        ax[ps_panel].plot(Theta, nullcline_offset[i]-273.15, color=cl[3], linestyle='dashed', linewidth=2.5)
    #ax[ps_panel].legend()

    ### Get anomaly size needed to cross t99 ###
    if anom_type == 'F': 
        anom_size = find_anomaly_size_F(loc_ind, t99, m)

    if anom_type == 'Td': 
        anom_size = find_anomaly_size_Td(loc_ind, t99, m)
    
    ### make anomaly size/probability panel ###
    ax[prob_panel].plot(Theta, anom_size, color='k', linestyle='solid', linewidth=2.5)
    
    # second axis to plot probabilities on
    ax12 = ax[prob_panel].twinx()
    ax12.plot(Theta, 1-norm.cdf(anom_size), linestyle='dashed', color='k', linewidth=2.5)
    #ax12.set_ylim((0,max(1-norm.cdf(anom_size))+0.02))
    
    # add labels
    if anom_type == 'F':
        sig_label = '$\sigma_{\mathcal{F}}$'

    elif anom_type == 'Td':
        sig_label = '$\sigma_{T_{d}}$'
        
    if right:
        ax12.set_ylabel("Probability of\n occurrence", fontsize=22, rotation=270, labelpad=50)
        ax[ps_panel].set_title("Washington DC", fontsize=24)
        ax[ps_panel].text(0.513,23.9, '+' + sig_label, fontsize=18)
        ax[ps_panel].text(0.513,27,'+2' + sig_label, fontsize=18)
        if anom_type == 'Td':
            ax[ps_panel].text(0.513,30.26, '+3' + sig_label, fontsize=18)

    if left:
        ax[prob_panel].set_ylabel(sig_label + " required\nto exceed $T_{99}$", fontsize=22)
        ax12.set_yticklabels([])
        ax[ps_panel].set_ylabel('Surface Temperature ($^{\circ}$C)', fontsize=22)
        ax[ps_panel].text(0.428,24.75,'+' + sig_label, fontsize=18)
        ax[ps_panel].text(0.428,27,'+2' + sig_label, fontsize=18)
        if anom_type == 'Td':
            ax[ps_panel].text(0.428,29.26,'+3' + sig_label, fontsize=18)
        ax[ps_panel].set_title("Southern Great Plains", fontsize=24)
        
    if center:
        ax12.set_yticklabels([])
        ax[ps_panel].set_title("Atlanta, GA", fontsize=24)
        ax[ps_panel].text(0.507,24.05+0.44,'+' + sig_label, fontsize=18)
        ax[ps_panel].text(0.507,26.3+0.44,'+2' + sig_label, fontsize=18)
        
        if anom_type == 'Td':
            ax[ps_panel].text(0.507,28.66+0.44,'+3' + sig_label, fontsize=18)
    
    ax[prob_panel].set_xlabel("Volumetric soil moisture (cm$^{3}$/cm$^{3}$)", fontsize=22)
    ax[prob_panel].set_yticks([0, 1, 2, 3, 4, 5])
    ax12.set_yticks([0, .1, .2, .3, .4, .5])

    # decorate
    sns.despine(ax=ax12, right=False, bottom=True, offset=10, trim=True)

    for label in [prob_panel]:
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(-0.78, 0.1, fig.dpi_scale_trans)
        ax[label].text(0.0, 1.0, label, transform=ax[label].transAxes + trans, fontsize=24, fontweight='bold',
                verticalalignment='top', bbox=dict(facecolor='1', edgecolor='none', pad=1))

    for label in [ps_panel]:
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(-0.9, -0.3, fig.dpi_scale_trans)
        ax[label].text(0.0, 1.0, label, transform=ax[label].transAxes + trans, fontsize=24, fontweight='bold',
                verticalalignment='top', bbox=dict(facecolor='1', edgecolor='none', pad=1))