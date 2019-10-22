#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:18:12 2019

function to get the photodiode(PD) timing for the movie and the car task


@author: nkozhemi
"""
import appnope
appnope.nope()
from mne.io.pick import pick_channels_regexp
from mne import find_events
import numpy as np

import matplotlib.pyplot as plt
from mne import Epochs
from mne.io import read_raw_ctf

def get_photodiode_events(raw):
    
    raw = raw.copy()
    
    # pick PD channel time series from CTF data
    raw.info['comps'] = []
    PD = pick_channels_regexp(raw.info['ch_names'],'UADC015-3007')
    PD_ts = raw.get_data(picks = PD)
    
    #find all 'ON' states of PD
    n, bins, patches = plt.hist(PD_ts[0,:], bins=50, 
                                    range=(np.percentile(PD_ts[0,:], 1),
                                    np.percentile(PD_ts[0,:], 99)),
                                    color='#0504aa', alpha=0.7, rwidth=0.85)
    thr = bins[np.nonzero(n == np.min(n))]# set a treshold
    if len(thr) > 1:
        thr = thr[-1]
    T_PD = PD_ts > thr
    Ind_PD_ON = []
    Ind_PD_OFF = []
    for ind, n in enumerate(T_PD[0,:]):
        if n == True and T_PD[0,ind-1] == False:
            Ind_PD_ON.append(ind)
        elif  n == False and T_PD[0,ind-1] == True:
            Ind_PD_OFF.append(ind)
    return PD_ts, Ind_PD_ON, Ind_PD_OFF, T_PD

def plot_events(PD_ts, Ind_PD_ON, T_PD, Ind_PD_OFF,Trig_ts, events):
    
    #plt.plot((PD_ts[0,:]-np.mean(PD_ts[0,:]))*100)
    plt.plot((PD_ts[0,:]))
    plt.plot(T_PD[0,:]*5)
    y = np.ones((1,len(Ind_PD_ON)))
    y = y = 0.5*y        
    plt.plot(np.array(Ind_PD_ON), y[0,:], 'o', color='black');
    y = np.ones((1,len(Ind_PD_OFF)))
    y = y = 0.5*y
    plt.plot(np.array(Ind_PD_OFF), y[0,:], 'o', color='red');
    plt.plot(events[:,0], events[:,2], 'o', color='green');
    plt.plot((Trig_ts[0,:]))
    plt.ylabel('a.u.')
    plt.xlabel('samples')
    plt.title('Photodiode and triggers timing for ' + task + ' task')
   

def get_triger_names_PD(event_id, Ind_PD_ON, events_trig):
    #create events
    events = np.zeros((len(Ind_PD_ON),3))
    events[:,0] = Ind_PD_ON;
        
        #get trigger names for PD ON states
    for key, value in event_id.items():
        ind = events_trig[:,2] == value
        time = events_trig[ind,0]
        for n in time:
            inx = (Ind_PD_ON-n)
            m = min(inx[inx>0])
            events[inx == m,2] = value;
    return events

def get_events(raw, task, plot=1):
    #general description of the data
    fs = raw.info['sfreq']
    time = raw.buffer_size_sec
    N_samples = raw.n_times;
    print ('Data is composed from')
    print(str(N_samples) + ' samples with ' + str(fs) + ' Hz sampling rate')
    print('Total time = ' + str(time) +'s')
    
    #get photodiode events from CTF data
    PD_ts, Ind_PD_ON, Ind_PD_OFF, T_PD = get_photodiode_events(raw)
                
    #pick Trigger channel time series from CTF data
    Trig = pick_channels_regexp(raw.info['ch_names'],'UPPT001')
    Trig_ts = raw.get_data(picks = Trig)
    
    #get events from trigger channel
    events_trig = find_events(raw, stim_channel='UPPT001')
    
    if task == 'Car':
        event_id = {'Transp/H2L': 10, 'Transp/L2H': 20, 
                        'NotTransp/H2L': 30, 'NotTransp/L2H': 40}
        #get trigger names for PD ON states
        events = get_triger_names_PD(event_id, Ind_PD_ON, events_trig)
    
    elif task == 'Movie':
        if len(Ind_PD_ON) and len(Ind_PD_OFF) != 196:
            print('NOT ALL OF THE PD STIM PRESENT!!!')
        event_id = []
        events = np.zeros((len(Ind_PD_ON),3))
        events[:,0] = Ind_PD_ON;
        events[:,2] = 1;
            
    elif task == 'Flanker':
        event_id = {'Incongruent/Left': 3, 'Incongruent/Right': 5, 
                    'Congruent/Left': 4, 'Congruent/Right': 6,
                    'Reward/Face': 7,'Reward/Coin': 8,
                    'Reward/Neut_FaceTRL': 9,'Reward/Neut_CoinTRL': 10}
                #get trigger names for PD ON states
        events = get_triger_names_PD(event_id, Ind_PD_ON, events_trig)
    
    events = events.astype(np.int64)
    #plot trig and PD
    if plot:
        plt.figure()
        plot_events(PD_ts, Ind_PD_ON, T_PD, Ind_PD_OFF,Trig_ts, events)    
    return event_id, events

data_path = '/Users/nkozhemi/Documents/MATLAB/18000P/18000P_CarTask_20180801_04.ds'
data_path = '/Users/nkozhemi/Documents/MATLAB/18000P/18000P_Flanker_20180801_02.ds'
data_path = '/Users/nkozhemi/Documents/MATLAB/18000P/18000P_Movie_20180801_03.ds'
data_path = '/Users/nkozhemi/Documents/MATLAB/18069P/18069P_CarTask_20180816_04.ds'
data_path = '/Users/nkozhemi/Documents/MATLAB/18069P/18069P_Movie_20180816_03.ds'
data_path = '/Users/nkozhemi/Documents/MATLAB/18069P/18069P_Flanker_20180816_02.ds'

# Load a dataset that contains events
raw = read_raw_ctf(data_path, preload=True)
task = 'Flanker'

event_id, events = get_events(raw, task, plot=1)

epochs = Epochs(raw, events, event_id, tmin=-0.1, tmax=1,
                baseline=(None, 0), preload=True)
   
   