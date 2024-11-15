"""
Implementation of time domain features, extracted from EMG signals

Dependency : numpy, sklearn

Author : Martin Colot
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, normalize

def MAV(x):
    """
    Mean Absolute Value
    """
    return np.mean(np.abs(x), axis=2)

def RMS(x):
    """
    Root Mean Square
    """
    return np.sqrt(np.mean(x**2, axis=2))

def MAA(x):
    """
    Maximum Absolute Amplitude
    """
    return np.max(np.abs(x), axis=2)

def WL(x):
    """
    Waveform Length
    """
    return np.sum(np.abs(x[:, :, 1:] - x[:, :, :-1]), axis=2)

def ZC(x):
    """
    Zero-Crossings
    """
    return np.sum(np.sign(x[:, :, 1:] * x[:, :, :-1]), axis=2)

def SSC(x):
    """
    Slope Sign changes
    """
    return np.sum(np.sign((x[:, :, 2:] - x[:, :, 1:-1])*(x[:, :, 1:-1] - x[:, :, :-2])), axis=2)

def WA(x):
    """
    Wilson Amplitude
    """
    return np.sum(np.sign(np.abs(x[:, :, 1:] - x[:, :, :-1]) - np.std(x, axis=2).reshape((len(x), len(x[0]), 1))), axis=2)

def MFL(x):
    """
    Maximum Fractal Length
    """
    return np.log(np.sqrt(np.sum((x[:, :, 1:] - x[:, :, :-1])**2, axis=2)))

def KRT(x):
    """
    Kurtosis
    """
    return np.mean((x - np.mean(x, axis=2).reshape((len(x), len(x[0]), 1)))**4 / np.std(x, axis=2).reshape((len(x), len(x[0]), 1))**4, axis=2)


def computeTDF(X):
    """
    compute 9 Time Domain Features for each channel of each sample of subject in the dataset X
    :param X: dataset of shape (n_subjects, n_samples, n_channels, n_frames)
    :return: features vectors of shape (n_subjects, n_samples, n_channels * 9)
    """
    TDF = [MAV, RMS, MAA, WL, ZC, SSC, WA, MFL, KRT]
    tdfX = [np.concatenate(np.array([f(x) for f in TDF]), axis=1) for x in X]
    tdfX = np.array(tdfX, dtype="object")
    return tdfX

def normalizeTDF(tdfX):
    """
    normalize time domain features as defined in the paper, applying l2 normalization and standardization
    :param tdfX: features vectors to normalize of shape (n_subjects, n_samples, n_channels * 9)
    :return: normalized features vectors per subject
    """
    tdfX_n = tdfX.copy()
    for i in range(len(tdfX)):
      tdfX_n[i] = normalize(tdfX_n[i])
      tdfX_n[i] = StandardScaler().fit_transform(tdfX_n[i])
    return tdfX_n