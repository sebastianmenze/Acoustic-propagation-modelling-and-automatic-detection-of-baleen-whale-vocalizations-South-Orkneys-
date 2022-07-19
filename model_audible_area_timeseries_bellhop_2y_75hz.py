# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:23:09 2022

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

#%% get bathymetry slices
import pyresample


lo,la=np.meshgrid(g_lons, g_lats)
grid = pyresample.geometry.GridDefinition(lats=la, lons=lo)

m_loc=[-( 45+57.548/60) , -(60+24.297/60)]




from pyproj import Geod
geod = Geod("+ellps=WGS84")

bearings=np.arange(360)

bathy_dict={}
points_lat=pd.DataFrame()
points_lon=pd.DataFrame()

for b in bearings:
    print(b)

    points = geod.fwd_intermediate(lon1=m_loc[0],lat1=m_loc[1],azi1=b,npts=500,del_s=1000 )
    p_lon=points[3]
    p_lat=points[4]
    points_lat=pd.concat( [points_lat,pd.DataFrame(p_lat)],ignore_index=True,axis=1 )
    points_lon=pd.concat( [points_lon,pd.DataFrame(p_lon)],ignore_index=True,axis=1  )
    
    swath = pyresample.geometry.SwathDefinition(lons=p_lon, lats=p_lat)
    
    
    # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
    _, _, index_array, distance_array = pyresample.kd_tree.get_neighbour_info(
        source_geo_def=grid, target_geo_def=swath, radius_of_influence=500000,
        neighbours=1)
    # get_neighbour_info() returns indices in the flattened lat/lon grid. Compute
    # the 2D grid indices:
    index_array_2d = np.unravel_index(index_array, grid.shape)
    
    value = d[index_array_2d[0],index_array_2d[1]] 
    
    dvec=np.arange(0,1000*500,1000)
    bb=np.transpose(np.array([dvec,-value.data]))    
    bathy_dict[b]=    bb.tolist()

timevec = pd.Series( pd.date_range(start=pd.Timestamp('2016-01-01'),end=pd.Timestamp('2018-01-01'),freq='M') )

#%%
datestr=timevec[7]

tl_mat_dict={}
tl_map_dict={}


for datestr in timevec:

    ncfile=r"D:\copernicus_data\\"  + datestr.strftime('%Y-%m-%d')  + r"_ocean_reanalysis.nc"
    nc = Dataset(ncfile)
    
    
    la,lo=np.meshgrid(nc['latitude'][:].data, nc['longitude'][:].data)
    grid = pyresample.geometry.GridDefinition(lats=la, lons=lo)
    
    m_loc=[-( 45+57.548/60) , -(60+24.297/60)]
    
    geod = Geod("+ellps=WGS84")
    
    bearings=np.arange(360)
    
    z_ss_dict={}
    rp_ss_dict={}
    cw_dict={}
    points_lat=pd.DataFrame()
    points_lon=pd.DataFrame()
    
    tl_mat_ray=pd.DataFrame()
    lat_mat_ray=pd.DataFrame()
    lon_mat_ray=pd.DataFrame()
        
    for b in bearings:
        print(b)
    
        points = geod.fwd_intermediate(lon1=m_loc[0],lat1=m_loc[1],azi1=b,npts=500,del_s=1000 )
        p_lon=points[3]
        p_lat=points[4]
        points_lat=pd.concat( [points_lat,pd.DataFrame(p_lat)],ignore_index=True,axis=1 )
        points_lon=pd.concat( [points_lon,pd.DataFrame(p_lon)],ignore_index=True,axis=1  )
        
        swath = pyresample.geometry.SwathDefinition(lons=p_lon, lats=p_lat)
        
        
        # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
        _, _, index_array, distance_array = pyresample.kd_tree.get_neighbour_info(
            source_geo_def=grid, target_geo_def=swath, radius_of_influence=500000,
            neighbours=1)
        # get_neighbour_info() returns indices in the flattened lat/lon grid. Compute
        # the 2D grid indices:
        index_array_2d = np.unravel_index(index_array, grid.shape)
        
        
        temp = nc['thetao'][:][0,:,index_array_2d[1],index_array_2d[0]]
        sal = nc['so'][:][0,:,index_array_2d[1],index_array_2d[0] ]
        depth=nc['depth'][:]
        depth_mat=np.tile( depth, [sal.shape[0],1] )
        # depth.shape
        sound_speed = gsw.sound_speed(sal,temp,depth_mat) 
        sound_speed = pd.DataFrame( sound_speed.data )
        sound_speed=sound_speed.fillna(axis=1,method='ffill')
       
        # fig=plt.figure(num=6)
        # plt.clf()   
        # plt.imshow(np.transpose(sound_speed.values[:,:]),aspect='auto')
        # plt.pcolormesh(dvec,-depth,np.transpose(sound_speed.values))
        # plt.boxplot((sound_speed.values))
     
        
        dvec=np.arange(0,1000*500,1000)
    
    
      # ssp2 = sound_speed.astype('int')
        sspdic={}
        i=0
        dd=dvec.copy()
        dd[-1]=dvec[-1]*10
        for rang in dd:
            sspdic[rang]= sound_speed.iloc[i,:].values
            i=i+1
        ssp2=pd.DataFrame(sspdic)   
        
        depth=nc['depth'][:]
        dd=np.array(bathy_dict[b])[:,1].max()
        ixx=depth<dd
        
        ssp3=ssp2.iloc[ixx,:]
        
        dssp=depth.data.astype('int').copy()[ixx]
        dssp[0]=0
        dssp[-1]=dd
        ssp3.index=dssp
       # ssp2 = pd.DataFrame({
       #    0: [1540, 1530, 1532, 1533],     # profile at 0 m range
       #  100: [1540, 1535, 1530, 1533],     # profile at 100 m range
       #  200: [1530, 1520, 1522, 1525] },   # profile at 200 m range
       #  index=[0, 10, 20, 30])             # depths of the profile entries in m
     
        env = pm.create_env2d(
            depth= bathy_dict[b],
            soundspeed=ssp3,
            bottom_soundspeed=1450,
            bottom_density=1200,
            bottom_absorption=1.0,
            tx_depth=200,
            frequency=modelfrec,
            min_angle = -45,
            max_angle=  45)     
        ddarr=np.array(bathy_dict[b])
       
        env['rx_range'] = ddarr[:,0]
        # env['rx_range'] = np.linspace(0, 1000*299, 1000)
        
        env['rx_depth'] = 15
                
        # tloss = pm.compute_transmission_loss(env,mode='incoherent',debug=True)
        tloss = pm.compute_transmission_loss(env,mode='incoherent')

        tloss_dB= 20*np.log10( tloss.abs() )

   
        lats=points_lat.iloc[:,b]
        lons=points_lon.iloc[:,b]
        lat_mat_ray=pd.concat( [lat_mat_ray,lats],axis=1,ignore_index=True )
        lon_mat_ray=pd.concat( [lon_mat_ray,lons],axis=1,ignore_index=True )
    
        tl_mat_ray=pd.concat( [tl_mat_ray,pd.DataFrame(tloss_dB.values[0,:])],axis=1,ignore_index=True )
        
    z=tl_mat_ray.values.flatten()
    y=lat_mat_ray.values.flatten()
    x=lon_mat_ray.values.flatten()
    grid_x, grid_y = np.meshgrid(g_lons, g_lats)      
    pp=np.transpose(np.array([x , y ]))
    tl_circle_mat=interpolate.griddata(pp,z, (grid_x, grid_y),method='linear' )
    tldf=pd.DataFrame(tl_circle_mat)
    tldf.to_hdf('tl_ray_' + str(modelfrec)+ 'hz_'+datestr.strftime('%Y_%m_%d')+'.h5', key='df', mode='w')

   
    tl_mat_dict[datestr.strftime('%Y_%m_%d') ] = tl_mat_ray.copy()
    tl_map_dict[datestr.strftime('%Y_%m_%d') ] = tl_circle_mat.copy()
#%%    