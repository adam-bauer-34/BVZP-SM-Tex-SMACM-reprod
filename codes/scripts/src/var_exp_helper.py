import numpy as np

def calc_variance_ratio(sm, loc):
    """Calculate the variance ratio between T_{SM} and T_{d}.
    
    Parameters
    ----------
    sm: `xarray.DataArray` object
        DataArray of soil moisture for location of interest.
        Also works with numpy array. Must be in volumetric units.
    loc: `Location` object
        Location object containing all the model parameters for a
        given place.
        
    Returns
    -------
    eta: float
        ratio of soil moisture feedbacks to other feedbacks
    T0: float
        temperature when the soil is completely dry along the 
        diagnostic temperature equation (i.e., the nullcline)
    mbar: float
        mean soil moisture
    m_var: float
        soil moisture variance
    t_var: float
        soil moisture-induced temperature variance
    """
    
    eta = (loc.L * loc.gamma * loc.nu) * (loc.alpha_s + loc.alpha_r + loc.L * loc.gamma * loc.nu * loc.m_0)**(-1)
    T0 = (loc.F_mean * (1 - loc.G_adj) - loc.phi_D - loc.phi_L) * (loc.alpha_s + loc.alpha_r + loc.L * loc.gamma * loc.nu * loc.m_0)**(-1)
    
    sm_frac = (sm - min(sm)) * (max(sm) - min(sm))**(-1) 
    mbar = np.mean(sm_frac.values)
    m_var = np.var(sm_frac.values)
    
    t_diag = T0 * (1 + eta * sm_frac.values)**(-1)
    t_var = np.var(t_diag)
    return eta, T0, mbar, m_var, t_var

def calc_ext_ratio(t, loc):
    """Calculate the ratio between the dry-out temperature 
    and the most extreme temperature in the reanalysis product.
    
    Parameters
    ----------
    t: `xarray.DataArray` object
        DataArray of temperature for location of interest.
        Also works with numpy array.
    loc: `Location` object
        Location object containing all the model parameters for a
        given place.
        
    Returns
    -------
    ratio: float
        T_0 / T_x
    """
    
    T0 = (loc.F_mean * (1 - loc.G_adj) - loc.phi_D - loc.phi_L) * (loc.alpha_s + loc.alpha_r + loc.L * loc.gamma * loc.nu * loc.m_0)**(-1)
    TX = np.max(t.values) - loc.Td_mean
    
    return T0/TX

