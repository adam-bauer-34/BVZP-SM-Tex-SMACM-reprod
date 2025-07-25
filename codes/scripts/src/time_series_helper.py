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

def get_SMACM_integration_fullcoupled(LOC, ds_loc):
    """Integrate SMACM model equations.

    Parameters
    ----------
    LOC: `Location` class
        object with all the model parameters for the location we're
        interested in

    ds_loc: `xarray.Dataset`
        ERA5 data for the location we're interested in
        Needed variables:
            - time
            - t2m
            - m (translated soil moisture, between 0 and 1)
            - ssr
            - tp
            - d2m

    Returns
    -------
    M: (N_times,) numpy array
        soil moisture time series from SMACM integration

    T: (N_times,) numpy array
        surface temperature time series from SMACM integration

    LH: (N_times,) numpy array
        latent heat time series from SMACM integration

    DH: (N_times,) numpy array
        dry heat fluxes (sensible + longwave) time series from SMACM integration
    """
    
    # Define a higher resolution time mesh
    dn = 1.0  # fraction of days for timestep 
    dt = dn * 24 * 60 * 60  # timestep for integration, in seconds
    to = np.arange(0, len(ds_loc.time))  # former time
    ti = np.arange(0, len(ds_loc.time), dn)  # new time

    # set up empty arrays of our model variables
    M = np.zeros(len(ti))
    T = np.zeros(len(ti))
    LH = np.zeros(len(ti))
    DH = np.zeros(len(ti))
    
    # Interpolate hourly forcing terms to new (finer) grid
    ## divide shortwave by 3600 to get W instead of J
    Fsw = np.interp(ti, to, ds_loc.ssr.values / 3600.)

    ## multiply P by 1000 (density of liquid water) to get a mass
    ## divide by 3600 to get a mass flux 
    ## (original P data comes in height of the "puddle" in the gridbox)
    P = np.interp(ti, to, ds_loc.tp.values * 1000. / 3600.)

    ## interpolate Td as normal 
    Td = np.interp(ti, to, ds_loc.d2m.values)

    # define effective forcing (offset by ground heat flux and
    # intercepts from latent and sensible heat), see just below Eqn. (11)
    G_adj = LOC.G_adj
    F_eff = Fsw - LOC.phi_L - LOC.phi_D - np.mean(Fsw) * G_adj
    F_eff[F_eff <= 0] = 0.0

    # set initial conditions for the first summer 
    T[0]= ds_loc.t2m[0]
    M[0] = ds_loc.m[0]
    LH[0] = LOC.L * LOC.nu * LOC.gamma * M[0] * (T[0] - Td[0]) + LOC.phi_L  # Eqn (11) in the paper
    DH[0] = (LOC.alpha_s + LOC.alpha_r) * (T[0] - Td[0]) + LOC.phi_D  # Eqn (7) in the paper

    # print some checks or quantities of interest
    # print(Td, Fsw)
    # print(np.std(ds_loc.d2m.values), np.std(Fswi))

    # we are only interested in summers, so every time we elapse 92 days, we need
    # to reset the boundary conditions 
    summer_tracker = 0
    mu = LOC.mu 

    # integrate model equations 
    for j in range(1, len(ti)):
        # if we've gone 91 days, we've gone a whole summer
        # so now we set the ICs as the values from the data
        if summer_tracker == int(91 / dn):
            # set to ICs
            M[j] = ds_loc.m.values[int(j * dn)]

            # reset summer tracker
            summer_tracker = 0

        # else, we're in the clear, and we can simulate the model
        else:
            M[j] = M[j-1] + (dt / mu) * (P[j]
                                             - (Fsw[j] - G_adj * np.mean(Fsw)) * LOC.nu * LOC.gamma * M[j-1]
                                             / (LOC.alpha_s + LOC.alpha_r + LOC.L * LOC.gamma * LOC.nu * M[j-1]))

            # if condition sets upper and lower bounds on soil moisture
            # assumed to be established by runoff and infiltration
            if M[j] > 1:
                M[j] = 1
            
            if M[j] < 0:
                M[j] = 0

            summer_tracker += 1

        T[j] = F_eff[j] / (LOC.alpha_s + LOC.alpha_r + LOC.L * LOC.gamma * LOC.nu * M[j]) + Td[j]
        LH[j] = LOC.L * LOC.nu * LOC.gamma * M[j] * F_eff[j] / (LOC.alpha_s + LOC.alpha_r + LOC.L * LOC.gamma * LOC.nu * M[j]) + LOC.phi_L
        DH[j] = (LOC.alpha_s + LOC.alpha_r) * F_eff[j] / (LOC.alpha_s + LOC.alpha_r + LOC.L * LOC.gamma * LOC.nu * M[j]) + LOC.phi_D

    return M, T, LH, DH