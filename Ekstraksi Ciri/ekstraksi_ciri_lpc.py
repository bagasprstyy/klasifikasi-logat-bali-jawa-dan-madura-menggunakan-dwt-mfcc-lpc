# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 19:05:20 2021

@author: BAGAS
"""

import csv
import pandas as pd
import librosa as lr
import matplotlib.pyplot as plt
import numpy as np
import pywt
from glob import glob
from pywt import wavedec
from python_speech_features import mfcc
from sklearn.decomposition import PCA

#Mengakses Dataset
data_dir = 'All_data'
audio_files = glob(data_dir + '/*.wav')

x = len(audio_files)

for file in range(0, len(audio_files), 1):
    np.set_printoptions(suppress=True)
    audio, sfreq = lr.load(audio_files[file])
    time = np.arange(0, len(audio))/sfreq
    
    #Pra pengolahan
    logat = []
    audio = lr.util.normalize(audio)
    logat = lr.effects.preemphasis(audio)

    #Plot Audio
    #fig, ax = plt.subplots()
    #ax.plot(time, logat)
    #ax.set(xlabel = 'Time', ylabel = 'Amplitudo')
    #plt.show()
    
    #Array Hasil
    final = []
    #print ('Logat = ',logat)
    
    #Proses DWT
    coeffs = wavedec(logat, 'db2', level=2)
    cA2, cD2, cD1 = coeffs
    
    #print("cA2 = ", cA2)
    #print("cD2 = ", cD2)
    
    #Proses MFCC
    LPC= lr.lpc(cA2, 12)
    
    final =  LPC
    
    #Export Ke CSV
    f =  open('feature_lpc_all_data.csv', 'a')
    with f :
        w = csv.writer(f)
        w.writerow(final)