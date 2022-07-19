# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:20:59 2022

@author: Administrator
"""



from functools import partial
import multiprocessing  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os
from matplotlib.path import Path

from scipy.io import wavfile
from scipy import signal
from skimage.transform import rescale, resize, downscale_local_mean

from skimage import data, filters, measure, morphology
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import disk  # noqa

import pickle

from scipy.signal import find_peaks
from skimage.feature import match_template


    
def automatic_detector_specgram_corr(f,t,z,shape_label,df_shape,corrscore_threshold):
             
        offset_f=10
        offset_t=0.5                  
        shape_f=df_shape['Frequency_in_Hz'].values
        shape_t=df_shape['Time_in_s'].values
        shape_t=shape_t-shape_t.min()
       
        f_lim=[ shape_f.min() - offset_f ,  shape_f.max() + offset_f ]
        k_length_seconds=shape_t.max()+offset_t*2

        # generate kernel  
        time_step=np.diff(t)[0]  
        k_t=np.linspace(0,k_length_seconds,int(k_length_seconds/time_step) )
        ix_f=np.where((f>=f_lim[0]) & (f<=f_lim[1]))[0]
        k_f=f[ix_f[0]:ix_f[-1]]
       
        kk_t,kk_f=np.meshgrid(k_t,k_f)   
        kernel_background_db=0
        kernel_signal_db=1
        kernel=np.ones( [ k_f.shape[0] ,k_t.shape[0] ] ) * kernel_background_db
        # find wich grid points are inside the shape
        x, y = kk_t.flatten(), kk_f.flatten()
        points = np.vstack((x,y)).T 
        p = Path(list(zip(shape_t, shape_f))) # make a polygon
        grid = p.contains_points(points)
        mask = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon  
        kernel[mask]=kernel_signal_db
       
        ix_f=np.where((f>=f_lim[0]) & (f<=f_lim[1]))[0]
        spectrog = z[ ix_f[0]:ix_f[-1],: ] 
   
        result = match_template(spectrog, kernel)
        corr_score=result[0,:]
        t_score=np.linspace( t[int(kernel.shape[1]/2)] , t[-int(kernel.shape[1]/2)], corr_score.shape[0] )

        peaks_indices = find_peaks(corr_score, height=corrscore_threshold)[0]
    
       
        t1=[]
        t2=[]
        f1=[]
        f2=[]
        score=[]
        df=pd.DataFrame(columns=['t-1','t-2','f-1','f-2',shape_label+'_score'])       
        if len(peaks_indices)>0: 
            t2_old=0
            for ixpeak in peaks_indices:     
                tstar=t_score[ixpeak] - k_length_seconds/2 - offset_t
                tend=t_score[ixpeak] + k_length_seconds/2 - offset_t
                if tstar>t2_old:
                    t1.append(tstar)
                    t2.append(tend)
                    f1.append(f_lim[0]+offset_f)
                    f2.append(f_lim[1]-offset_f)
                    score.append(corr_score[ixpeak])
                t2_old=tend
            df['t-1']=t1
            df['t-2']=t2
            df['f-1']=f1
            df['f-2']=f2
            df[shape_label+'_score']=score           
        return df


def parafunc( audiopath_list,fileindex ):
    
    audiopath=audiopath_list[fileindex]
    
    # load sgram
    datekey='aural_%Y_%m_%d_%H_%M_%S.wav'
    starttime= dt.datetime.strptime( audiopath.split('\\')[-1], datekey )
  
    fs, x = wavfile.read(audiopath)
    fft_size=2**14
    f, t, Sxx = signal.spectrogram(x, fs, window='hamming',nperseg=fft_size,noverlap=0.9*fft_size)
    
    # filter out background
    spectrog = 10*np.log10(Sxx )     
    rectime= pd.to_timedelta( t ,'s')
    spg=pd.DataFrame(np.transpose(spectrog),index=rectime)
    bg=spg.resample('3min').mean().copy()
    bg=bg.resample('1s').interpolate(method='time')
    bg=    bg.reindex(rectime,method='nearest')
    background=np.transpose(bg.values)   
    z=spectrog-background
    
    df=pd.DataFrame()
    
    kernel_csv=r"D:\passive_acoustics\detector_delevopment\final_detections\templates\kernel_minke_bioduck_single.csv"
    df_shape=pd.read_csv(kernel_csv,index_col=0)
    df_corr= automatic_detector_specgram_corr(f,t,z,'minke_bioduck',df_shape,0.3)
    df=pd.concat([df,df_corr])
    
    kernel_csv=r"D:\passive_acoustics\detector_delevopment\final_detections\templates\kernel_minke_downsweep.csv"
    df_shape=pd.read_csv(kernel_csv,index_col=0)
    df_corr= automatic_detector_specgram_corr(f,t,z,'minke_downsweep',df_shape,0.3)
    df=pd.concat([df,df_corr])
    
    kernel_csv=r"D:\passive_acoustics\detector_delevopment\final_detections\templates\kernel_dcall_fine.csv"
    df_shape=pd.read_csv(kernel_csv,index_col=0)
    df_corr= automatic_detector_specgram_corr(f,t,z,'dcall',df_shape,0.3)
    df=pd.concat([df,df_corr])    
    
    kernel_csv=r"D:\passive_acoustics\detector_delevopment\final_detections\templates\kernel_fw_downsweep_1.csv"
    df_shape=pd.read_csv(kernel_csv,index_col=0)
    df_corr= automatic_detector_specgram_corr(f,t,z,'fw_downsweep',df_shape,0.3)
    df=pd.concat([df,df_corr])
    
    kernel_csv=r"D:\passive_acoustics\detector_delevopment\final_detections\templates\kernel_srw_1.csv"
    df_shape=pd.read_csv(kernel_csv,index_col=0)
    df_corr= automatic_detector_specgram_corr(f,t,z,'srw',df_shape,0.3)
    df=pd.concat([df,df_corr])
    
    df['realtime']=starttime + pd.to_timedelta( df['t-1'] ,'s')
    df['duration']=df['t-2']-df['t-1']
    df['f-width']=df['f-2']-df['f-1']
    df['filename']=audiopath 
    df=df.reset_index(drop=True) 
       
    if not os.path.exists('sgramcorr'):
        os.mkdir('sgramcorr')
    txt=audiopath.split('\\')
    
    targetname='sgramcorr\\' + txt[-1][0:-4]+'_sgramcorr_mf.csv' 
    df.to_csv(targetname)

    fft_size=2**15  
    f, t, Sxx = signal.spectrogram(x, fs, window='hamming',nperseg=fft_size,noverlap=0.9*fft_size)  
    # filter out background
    spectrog = 10*np.log10(Sxx )     
    rectime= pd.to_timedelta( t ,'s')
    spg=pd.DataFrame(np.transpose(spectrog),index=rectime)
    bg=spg.resample('3min').mean().copy()
    bg=bg.resample('1s').interpolate(method='time')
    bg=    bg.reindex(rectime,method='nearest')
    background=np.transpose(bg.values)   
    z=spectrog-background
 
    df=pd.DataFrame()
   
    kernel_csv=r"D:\passive_acoustics\detector_delevopment\final_detections\templates\kernel_fw_20hz.csv"
    df_shape=pd.read_csv(kernel_csv,index_col=0)
    df_corr= automatic_detector_specgram_corr(f,t,z,'fw20',df_shape,0.3)
    df=pd.concat([df,df_corr])

    kernel_csv=r"D:\passive_acoustics\detector_delevopment\final_detections\templates\kernel_zcall.csv"
    df_shape=pd.read_csv(kernel_csv,index_col=0)
    df_corr= automatic_detector_specgram_corr(f,t,z,'zcall',df_shape,0.3)
    df=pd.concat([df,df_corr])
    
    df['realtime']=starttime + pd.to_timedelta( df['t-1'] ,'s')
    df['duration']=df['t-2']-df['t-1']
    df['f-width']=df['f-2']-df['f-1']
    df['filename']=audiopath 
    df=df.reset_index(drop=True) 
    
    targetname='sgramcorr\\' + txt[-1][0:-4]+'_sgramcorr_lf.csv' 
    df.to_csv(targetname)    

#########
    return 


#%%


os.chdir(r'D:\passive_acoustics\detector_delevopment\final_detections')

if __name__ == '__main__':


    # audio_folder=r'I:\postdoc_krill\pam\2016_aural\**'    

    # audiopath_list=glob.glob(audio_folder+'\*.wav',recursive=True)
    
    # cpucounts=multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=cpucounts)
    # index_list=range(len( audiopath_list ))
    # para_result=pool.map( partial( parafunc,audiopath_list), index_list)
    # pool.close  
    
    audio_folder=r'I:\postdoc_krill\pam\2016_aural\**'    

    audiopath_list=glob.glob(audio_folder+'\*.wav',recursive=True)
    
    cpucounts=multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpucounts)
    index_list=range(len( audiopath_list ))
    para_result=pool.map( partial( parafunc,audiopath_list), index_list)
    pool.close      
    