#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:11:35 2020

@author: andrea
"""

import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from scipy import stats
import random as rnd


EQD_X = np.loadtxt('/home/andrea/DEMO/NNET/DEMO_S1_HNC_13ch_EQD_X.txt')
EQD_Yn = np.loadtxt('/home/andrea/DEMO/NNET/DEMO_S1_HNC_13ch_EQD_Yn.txt')
DEMO_total_counts = np.loadtxt('/home/andrea/DEMO/NNET/LIE_counts_13.txt')
a = np.linspace(EQD_X[0], 1, 21)
print(DEMO_total_counts)
 
NNETA = "Model_DEMO_13ch_everything_shuffled.h5"
NNETB = "Model_DEMO_13ch.h5"
NNETA = load_model(NNETA)
NNETB = load_model(NNETB)

rnd.seed()
ir1 = [rnd.randint(0, 12)]
ir2 = [rnd.randint(0, 12)]
DEMO_total_counts[ir1]=0
DEMO_total_counts[ir2]=0
print(DEMO_total_counts)
DEMO_total_counts[ir1] = (DEMO_total_counts[ir1[0]+1] + DEMO_total_counts[ir1[0]-1])/2
DEMO_total_counts[ir2] = (DEMO_total_counts[ir2[0]+1] + DEMO_total_counts[ir2[0]-1])/2
#print(DEMO_total_counts)

DEMO_targets = np.interp(a,EQD_X, EQD_Yn)

DEMO_target_norm = stats.zscore(DEMO_targets)
DEMO_target_mean= np.mean(DEMO_targets)
DEMO_target_std = np.std(DEMO_targets, ddof=1)

DEMO_total_counts_norm = stats.zscore(DEMO_total_counts)
DEMO_total_counts_mean= np.mean(DEMO_total_counts)
DEMO_total_counts_std = np.std(DEMO_total_counts, ddof=1)


NET_target_norm_NNETA = NNETA.predict(DEMO_total_counts_norm.reshape(-1, 13))
NET_target_norm_NNETB = NNETB.predict(DEMO_total_counts_norm.reshape(-1, 13))

NET_target_NNETA = NET_target_norm_NNETA*DEMO_target_std + DEMO_target_mean
NET_target_NNETB = NET_target_norm_NNETB*DEMO_target_std + DEMO_target_mean

plt.figure(1000,figsize = (6, 8))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])  
ax0 = plt.subplot(gs[0])
plt.subplot(gs[0])
plt.plot(a,DEMO_target_norm.T,'r')
plt.plot(a,NET_target_norm_NNETA.T,'b')
plt.plot(a,NET_target_norm_NNETB.T,'k')
plt.ylabel('Normalized emissivity', fontsize=14)
plt.legend(['DEMO','NNETA','NNETB'],fontsize=14,frameon=False)
plt.setp(ax0.get_xticklabels(), visible=False)

ax1 = plt.subplot(gs[1], sharex = ax0)
plt.plot(a,DEMO_targets.T,'r')
plt.plot(a,NET_target_NNETA.T,'b')
plt.plot(a,NET_target_NNETB.T,'k')

plt.ylabel('emissivity (1/m3/s)', fontsize=14)
plt.legend(['DEMO','NNETA','NNETB'],fontsize=14,frameon=False)
plt.setp(ax1.get_xticklabels(), visible=False)


plt.subplot(gs[2], sharex = ax0)
plt.plot(a,100*(1-NET_target_NNETA.T/DEMO_targets.reshape(21, 1)),'b--')
plt.plot(a,100*(1-NET_target_NNETB.T/DEMO_targets.reshape(21, 1)),'k--')
plt.ylabel('(%)', fontsize=14)
plt.ylim(-20, 20)
plt.xlabel('normalized minor radius', fontsize=14)
plt.legend(['rel diff NNETA','rel diff NNETB'],fontsize=14,frameon=False)
plt.subplots_adjust(hspace=.0)

plt.savefig('NNETA_VS_NNETB_2ch_interp.png')





