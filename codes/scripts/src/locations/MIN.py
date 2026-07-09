"""
Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
2.26.2022

This code contains the location class for Minneapolis, MN,
or "MN." MN is a subclass of the Location abstract class. 
"""

import numpy as np 

from .Location import Location

class MIN(Location):
    """
    Class object for Dallas, TX. Subclass of the Location
    abstract class.
    """
    def __init__(self):
        """
        Of use constants.
        """
        self.L = 2.5e6
        self.C = 4180 
        self.P_surf = 101325
        self.R_w = 461.52
        self.T_0 = 273.15 

        """
        Model parameters.
        """
        self.alpha_s = 11.3
        self.alpha_r = 0
        self.nu = 0.007
        self.gamma = 0.0007
        self.mu = 70 * 0.8
        self.m_0 = 0.0
        self.F_mean = 212.3
        self.F_std = 51.079645207299464
        self.Td_mean = 288.2
        self.Td_std = 3.8579948
        self.omega = 4.5
        self.omega_s = self.omega * 86400 # days -> seconds
        self.p_0 = 0.08
        self.p_scale = 1
        self.F_warming_max = None
        self.m_mean = 0.0617308
        self.phi_D = 19.2
        self.phi_L = 58.3
        self.G_adj = 0.07

    def calibrate_warming_simulations(self, max_warming):
        """
        Calculate the maximum increase in F needed to warm the mean temperature
        by max_warming.

        Arguments
        ---------
        max_warming: float
            maximum amount of warming to allow in the location (K)
        """
        
        """
        Using nullcline equation, we can write the mean temperature as:
        """
        mean_T = self.Td_mean + self.F_mean * (self.alpha_s + self.alpha_r +
                                               self.L * self.gamma * self.nu *
                                               (self.m_0 + self.m_mean))**(-1)
        """
        Then add a forcing anomaly such that the temperature is equal to the
        mean_T + some amount. Solving for the forcing anomaly, we have:
        """
        diff_T = mean_T - self.Td_mean + max_warming 
        feedback = self.alpha_r + self.alpha_s + self.gamma * self.L * self.nu * (self.m_0 + self.m_mean)
        self.F_warming_max = diff_T * feedback - self.F_mean
