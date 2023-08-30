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
    audio_nr = lr.util.normalize(audio)
    logat = lr.effects.preemphasis(audio_nr)

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
    MFCC= mfcc(cA2, sfreq, winlen=0.02, winstep=0.01, numcep=13, winfunc=np.hamming, nfft=1024)
    mfcc_t = MFCC.transpose()
    #print("MFCC = ", MFCC)
    
    #Statistik
    #min
    min_mfcc = np.amin(mfcc_t,1)
    stat_mfcc = min_mfcc
    #max
    max_mfcc = np.amax(mfcc_t,1)
    stat_mfcc = np.vstack((stat_mfcc, max_mfcc))
    #mean
    mean_mfcc = np.mean(mfcc_t,1)
    stat_mfcc = np.vstack((stat_mfcc, mean_mfcc))
    #median
    median_mfcc = np.median(mfcc_t,1)
    stat_mfcc = np.vstack((stat_mfcc, median_mfcc))
    #standar deviasi
    stddev_mfcc = np.std(mfcc_t,1)
    stat_mfcc = np.vstack((stat_mfcc, stddev_mfcc))
    #print(finalstat_mfcc.shape)
    finalstat = stat_mfcc.transpose()
    statistik = finalstat
    #print(statistik)
    #print("Statistika = ",statistik)
    #print ("Statistik = ", statistik)
    
    #PCA
    df = pd.DataFrame (statistik)
    data5 = df.head()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df)
    
    final_1 = np.reshape(pca_result[:,:2], 26, order='F')
    final =  final_1
    
    #Export Ke CSV
    f =  open('feature_mfcc13_all_data.csv', 'a')
    with f :
        w = csv.writer(f)
        w.writerow(final)