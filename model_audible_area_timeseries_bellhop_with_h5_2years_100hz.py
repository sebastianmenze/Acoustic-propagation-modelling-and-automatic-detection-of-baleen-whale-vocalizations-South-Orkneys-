# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 13:42:25 2022

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:51:24 2022

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:11:19 2022

@author: Administrator
"""


import h5py

# from pyram.PyRAM import PyRAM
from scipy import interpolate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os
import sys

os.chdir(r'D:\passive_acoustics\propagation_modelling')

import gsw

from netCDF4 import Dataset
import pandas as pd
import cartopy
import cartopy.crs as ccrs
from scipy.ndimage import gaussian_filter

import arlpy.uwapm as pm
from pyproj import Geod
geod = Geod("+ellps=WGS84")

modelfrec=75

# load data and slice out region of interest

# read mapdata

latlim=[-62,-56]
lonlim=[-(46+5),-(46-5)]

spacer=1
gebcofile=r"C:\Users\a5278\Documents\gebco_2020_netcdf\GEBCO_2020.nc"
gebco = Dataset(gebcofile, mode='r')
g_lons = gebco.variables['lon'][:]
g_lon_inds = np.where((g_lons>=lonlim[0]) & (g_lons<=lonlim[1]))[0]
# jump over entries to reduce data
g_lon_inds=g_lon_inds[::spacer]

g_lons = g_lons[g_lon_inds]
g_lats = gebco.variables['lat'][:]
g_lat_inds = np.where((g_lats>=latlim[0]) & (g_lats<=latlim[1]))[0]
# jump over entries to reduce data
g_lat_inds=g_lat_inds[::spacer]

g_lats = g_lats[g_lat_inds]
d = gebco.variables['elevation'][g_lat_inds, g_lon_inds]
gebco.close()

#%%

h5files=glob.glob('tl_ray_'+str(modelfrec)+'hz_*.h5')
h5file=h5files[0]

timevec=[]
tl_map_dict={}

for h5file in h5files:
    
    tl_circle_mat=pd.read_hdf(h5file).values
    
    datestr=pd.to_datetime(h5file,format='tl_ray_'+str(modelfrec)+'hz_%Y_%m_%d.h5')
    timevec.append(datestr )


   
    tl_map_dict[datestr.strftime('%Y_%m_%d') ] = tl_circle_mat.copy()
    

# timevec = pd.Series( pd.date_range(start=pd.Timestamp('2017-01-01'),end=pd.Timestamp('2018-01-01'),freq='M') )
timevec=pd.Series(timevec)
#%%

# specmat2016=pd.read_csv('longternspec2016.csv',index_col=0)
# specmat2016.index=pd.to_datetime(specmat2016.index)
# specmat2016.to_hdf('longtermspectrogram_2016.h5', key='df')


# specmat2017=pd.read_csv('longternspec2017.csv',index_col=0)
# specmat2017.index=pd.to_datetime(specmat2017.index)
# specmat2017.to_hdf('longtermspectrogram_2017.h5', key='df')


specmat2016=pd.read_hdf('longtermspectrogram_2016.h5')
specmat2017=pd.read_hdf('longtermspectrogram_2017.h5')

specmat=10*np.log10( pd.concat([specmat2016,specmat2017]) )

#%%

#%% get audible area

sl_db=180 

az12,az21,dist1 = geod.inv(g_lons[0],g_lats[0],g_lons[0],g_lats[-1])
az12,az21,dist2 = geod.inv(g_lons[0],g_lats[0],g_lons[-1],g_lats[0])

pixel_area= dist1 / g_lats.shape[0] * dist2 / g_lons.shape[0] # sqarementers

ix_f=np.argmin( np.abs( specmat.columns.astype(float) - modelfrec ) )

t=specmat.index[0]
i=0
area_audible=[]
for t in specmat.index:
    noise_db=specmat.iloc[i,ix_f]
    
    ix_tltime= np.argmin( np.abs(timevec - t) ) 
    tl_circle_mat_filt=gaussian_filter(tl_map_dict[timevec[ix_tltime].strftime('%Y_%m_%d')], sigma=3)
    
    snr=sl_db + tl_circle_mat_filt - noise_db
    
    aa=np.sum( snr>5 ) * pixel_area
    area_audible.append(aa)
    i=i+1
area_audible=pd.DataFrame(area_audible)    
area_audible.index=specmat.index
# area_audible.to_csv('audible_area_' + str(modelfrec) +  'hz_sl180_ray_bathandssp.csv')

#%%

# area_audible=pd.read_csv('audible_area_500hz_sl180_ray_bathandssp.csv',index_col=0)
# area_audible.index=pd.to_datetime(area_audible.index)

plt.figure(num=6)
plt.clf()     
plt.plot(area_audible,'.k')
plt.plot(area_audible.resample('1d').mean(),'-r')
plt.ylabel('Area in m$^2$')
plt.title('Area where 180 dB calls can be detected (SNR>5 dB @ '+str(modelfrec)+' Hz)')
plt.grid()
    
    # plt.savefig('detection_area_timeseries_500hz_bellhop.jpg',dpi=300)
      