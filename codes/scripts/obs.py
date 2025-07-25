import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from netCDF4 import Dataset

from fig_locs import figs_path
from data_locs import obs_T_filename, obs_M_filename

def hor_interp(array,X_initial,Y_initial,X_final,Y_final):
	from scipy.interpolate import griddata
	x,y = np.meshgrid(X_initial,Y_initial)
	x_final,y_final = np.meshgrid(X_final,Y_final)
	final_array = griddata((x.ravel(),y.ravel()),array.ravel(),(x_final,y_final))
	return(final_array)

def add_time_dim(xda):
	xda = xda.expand_dims(time=[datetime.datetime.now()])
	return(xda)

def plotting(X,Y,Z):
    import matplotlib as mpl
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

###############################################################################################
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None,midpoint=None,clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self,vmin,vmax,clip)
        def __call__(self,value,clip=None):
            x, y = [self.vmin,self.midpoint,self.vmax], [0,0.5,1]
            return np.ma.masked_array(np.interp(value,x,y)) 

################################################################################################

# Some plotting information required to make color maps
    vmin = 0
    vmax = 100
    midpoint = 50	
    ext = 'both'
    bounds = np.linspace(vmin,vmax,11)
    #bounds = vmax*np.array([-1,-.9,-.75,-.60,-.45,-.30,-.15,-0.05,.05,.15,.3,.45,.6,.75,.9,1])

#    cmap = cm.copper
#    cmap = cm.BrBG
#    cmap = cm.gist_earth
    cmap = cm.gist_heat
#    cmap = cm.RdBu_r

##################################################
    
    states_provinces = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',scale='50m',facecolor='none')
    fig = plt.figure(figsize=(8,8))


    # FOR USA central lon = 265
    #proj = ccrs.Orthographic(central_longitude=240,central_latitude=35)

    proj = ccrs.Robinson()
    ax1 = fig.add_axes([0.1,0.2,0.8,0.65],projection=proj)
    #ax1.add_feature(cfeature.BORDERS)
    ax1.coastlines()
    #ax1.set_extent([225,300,25,50])
    #ax1.add_feature(states_provinces)
    ax1.contourf(X,Y,Z,norm=MidpointNormalize(vmin=vmin,vmax=vmax,midpoint=midpoint),transform=ccrs.PlateCarree(),cmap=cmap,levels=bounds,extend=ext)
    #ax1.contour(X,Y,C,levels=np.array([30,35,40]),transform=ccrs.PlateCarree(),colors='C3',linewidth=3)	
    ax1.set_global()

#####################################################################################################
    ax1_cbar = fig.add_axes([0.1,0.12,0.8,.02])
    cb1 = mpl.colorbar.ColorbarBase(ax1_cbar,norm=MidpointNormalize(vmin=vmin,vmax=vmax,midpoint=midpoint),spacing='uniform',
        cmap=cmap,boundaries=bounds,ticks=bounds,orientation='horizontal')

    plt.savefig(figs_path + 'fig1_obs.pdf')

def heat_fraction():

    years = np.linspace(2012,2021,10)
    nyears = 10
    i = 0
    mset = Dataset(obs_M_filename)

    m = mset['m'][:,:,:]
    mlon = mset['lon'][:]
    mlat = mset['lat'][:]
    mset.close()

    Tset = Dataset(obs_T_filename)
    Tlon = Tset['lon'][:]
    Tlat = Tset['lat'][:]
    T = Tset['T'][:,:,:]
    
    NX = 360
    NY = 125
    X = np.linspace(0,360,NX)
    Y = np.linspace(-55,70,NY)
    
    Z = np.zeros(shape=(NY,NX))
    C = np.zeros(shape=(NY,NX))
    i = 0

    while i < NY:
        j = 0
        while j < NX:
            #dryest,dry,wet,wetest = fraction(X[j],Y[i],Tlon,Tlat,mlon,mlat)
            
            diff = fraction(X[j],Y[i],Tlon,Tlat,mlon,mlat,T)
            Z[i,j] = diff*100
            #C[i,j] = thresh

            j+=1
        i+=1

    #write_2d('mmin_less_mbar.nc','diff',X,Y,Z)
    plotting(X,Y,Z)

def fraction(Blon,Blat,Tlon,Tlat,mlon,mlat,T):

    years = np.linspace(2012,2021,10)
    nyears = 10
    
    Tlatdex = np.argmin(abs(Tlat - Blat))
    Tlondex = np.argmin(abs(Tlon - Blon))
    mlatdex = np.argmin(abs(mlat - Blat))
    if Blon > 180:
        mlondex = np.argmin(abs(mlon - (Blon - 360)))
    else:
        mlondex = np.argmin(abs(mlon - Blon))

    Tyear = np.reshape(T[:,Tlatdex,Tlondex],(10,92))
    thresh = np.nanpercentile(Tyear,q=99)
    
    i = 0
    mseason = np.zeros(shape=(10,92))
    while i < 10:
        mset = Dataset('mcci_jja_'+str(int(years[i]))+'.nc')
        m = mset['m'][:,mlatdex,mlondex]
        mset.close()
        mseason[i,:] = m[:]
        i+=1


    i = 0
    dryest = 0
    total = 0
    m25 = np.nanpercentile(mseason,q=25)

    while i < 10:
        Tdex_dryest = np.where(np.logical_and(Tyear[i,:] > thresh,mseason[i,:]<m25))[0]
        dryest+=len(Tdex_dryest)
        
        Tdex_hottest = np.where(Tyear[i,:] > thresh)[0]
        total +=len(Tdex_hottest)
        i+=1

    if total > 0:
        return(dryest/total)
    else:
        return(np.nan)


heat_fraction()

