"""
Functions to load and preprocess the public datasets used in the paper, and shape it correctly for testing (NinaproDB4, EMG-EPN-612, Toro-Ossaba et al. 2022)
You can contact the authors of the paper to access the custom dataset evaluated in the paper

dependency : scipy, numpy, json

Author : Martin Colot
"""

import scipy
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt
import json


def lowpassFilter(X, lowPass, sfreq=2000):
    lowPass = lowPass / sfreq
    b2, a2 = butter(4, lowPass, btype='lowpass')
    return filtfilt(b2, a2, X)

def highpassFilter(X, highpass, sfreq=2000):
    highpass = highpass / sfreq
    b2, a2 = butter(4, highpass, btype='highpass')
    return filtfilt(b2, a2, X)

def bandPassFilter(X, low, high, sfreq):
  X = lowpassFilter(X, high, sfreq)
  X = highpassFilter(X, low, sfreq)
  return X

def comb_filter(data, f0, Q, fs):
    w0 = f0/(0.5*fs)
    b, a = iirnotch(w0, Q)
    y = filtfilt(b, a, data)
    return y


def load_ninaproDB4(folderPath):
    """
    loads samples from the ninaproDB4 using same preprocessing as described in the paper
    :param folderPath: path to the folder where the subject's folder, containing their .mat files, of the ninapro db4 dataset should be loaded (available on https://ninapro.hevs.ch/instructions/DB4.html)
    :return: features vectors and labels for all the subjects
    """
    GesturesE1 = {12: 0}
    GesturesE2 = {2: 1, 5: 2, 6: 3, 10: 4, 12: 5, 13: 6, 14: 7, 15: 8, 16: 9}
    sfreq = 2000
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    allDataE1 = []
    allDataE2 = []
    for subject in subjects:
        allDataE1.append(scipy.io.loadmat(f"{folderPath}/S{subject}_E1_A1.mat"))
        allDataE2.append(scipy.io.loadmat(f"{folderPath}/S{subject}_E2_A1.mat"))
    Q = 30  # Quality factor for notch filter
    f0 = 50  # Frequency to be removed (Hz)
    harmonic_freqs = np.arange(f0, 401, f0)
    for s in range(len(allDataE1)):
        for channel in range(12):
            allDataE1[s]["emg"][:, channel] = bandPassFilter(allDataE1[s]["emg"][:, channel], 1, 150, sfreq)
            allDataE2[s]["emg"][:, channel] = bandPassFilter(allDataE2[s]["emg"][:, channel], 1, 150, sfreq)
            for harmonic in harmonic_freqs:
                allDataE1[s]["emg"][:, channel] = comb_filter(allDataE1[s]["emg"][:, channel], harmonic, Q, sfreq)
                allDataE2[s]["emg"][:, channel] = comb_filter(allDataE2[s]["emg"][:, channel], harmonic, Q, sfreq)
            # envelope
            allDataE1[s]["emg"][:, channel] = np.abs(allDataE1[s]["emg"][:, channel])
            allDataE1[s]["emg"][:, channel] = lowpassFilter(allDataE1[s]["emg"][:, channel], 75, sfreq)
            allDataE2[s]["emg"][:, channel] = np.abs(allDataE2[s]["emg"][:, channel])
            allDataE2[s]["emg"][:, channel] = lowpassFilter(allDataE2[s]["emg"][:, channel], 75, sfreq)
    X = []
    Y = []
    windowSize = 5
    windowSize *= sfreq
    windowSize = int(windowSize)
    for s in range(len(allDataE1)):
        x = []
        y = []
        for i in range(1, len(allDataE1[s]["emg"]) - windowSize, 1):
            if allDataE1[s]["stimulus"][i - 1][0] == 0 and allDataE1[s]["stimulus"][i][0] in GesturesE1:
                x.append(allDataE1[s]["emg"][i:i + windowSize].T)
                y.append(GesturesE1[allDataE1[s]["stimulus"][i][0]])
        for i in range(1, len(allDataE2[s]["emg"]) - windowSize, 1):
            if allDataE2[s]["stimulus"][i - 1][0] == 0 and allDataE2[s]["stimulus"][i][0] in GesturesE2:
                x.append(allDataE2[s]["emg"][i:i + windowSize].T)
                y.append(GesturesE2[allDataE2[s]["stimulus"][i][0]])
        X.append(np.array(x))
        Y.append(np.array(y))
    X = np.array(X, dtype="float")
    Y = np.array(Y, dtype="float")
    return X, Y


def load_ToroOssaba(folderPath):
    """
    loads samples from the Toro-Ossaba et al. 2022 using same preprocessing as described in the paper
    :param folderPath: path to the folder where the subject's folder, containing their .txt files, of the Toro-Ossaba et al. 2022 dataset should be loaded (available on https://zenodo.org/records/7668251)
    :return: features vectors and labels for all the subjects
    """
    def csvToArray(path):
        res = []
        with open(path, "r") as f:
            for line in f:
                res.append(list(map(float, line.split("\t"))))
        return np.array(res)

    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    allData = []
    for subject in subjects:
        allData.append([])
        for i in range(5):
            allData[-1].append(csvToArray(f"{folderPath}/subject {subject}/{i}.txt"))
    Q = 30  # Quality factor for notch filter
    f0 = 50  # Frequency to be removed (Hz)
    harmonic_freqs = np.arange(f0, 401, f0)
    sfreq = 1024
    for s in range(len(allData)):
        for c in range(5):
            for channel in range(4):
                allData[s][c][:, channel] = bandPassFilter(allData[s][c][:, channel], 1, 150, sfreq)
                for harmonic in harmonic_freqs:
                    allData[s][c][:, channel] = comb_filter(allData[s][c][:, channel], harmonic, Q, sfreq)
                # envelope
                allData[s][c][:, channel] = np.abs(allData[s][c][:, channel])
                allData[s][c][:, channel] = lowpassFilter(allData[s][c][:, channel], 75, sfreq)
    X = []
    Y = []
    windowSize = 0.5
    windowSize *= sfreq
    windowSize = int(windowSize)
    for s in range(len(allData)):
        X.append([])
        Y.append([])
        for c in range(5):
            for i in range(0, len(allData[s][c]) - windowSize, windowSize):
                X[-1].append(allData[s][c][i:i + windowSize, :].T)
                Y[-1].append(c)
        X[-1] = np.array(X[-1])
        Y[-1] = np.array(Y[-1])
    X = np.array(X, dtype="object")
    Y = np.array(Y, dtype="object")
    return X, Y

def load_EMG_EPN_612(folderPath):
    """
    loads samples from the EMG-EPN-612 dataset using same preprocessing as described in the paper
    :param folderPath: path to the folder where the dataset should be loaded (available on https://zenodo.org/records/4421500)
    :return: features vectors and labels for all the subjects
    """
    X = []
    Y = []
    for user in range(1, 105):  # we only keep the first 100 clean subjects from the 307 available in the training dataset
        file = f"{folderPath}/trainingJSON/user{user}/user{user}.json"
        try:
            f = open(file)
            data = json.load(f)

            x = []
            y = []
            for idx in range(1, 151):
                sample = []
                if len(data["trainingSamples"][f"idx_{idx}"]["emg"][f"ch1"]) >= 900:
                    for ch in range(1, 9):
                        sample.append(data["trainingSamples"][f"idx_{idx}"]["emg"][f"ch{ch}"][:900])
                    x.append(sample)
                    y.append(
                        data["generalInfo"]["myoPredictionLabel"][data["trainingSamples"][f"idx_{idx}"]["gestureName"]])
            x = np.array(x, dtype="object")
            y = np.array(y, dtype="object")
            X.append(x)
            Y.append(y)
        except Exception as e:
            print("error :", e)
    X = np.array(X, dtype="object")
    Y = np.array(Y, dtype="object")
    # subject 54 is empty
    X = np.delete(X, 54)
    Y = np.delete(Y, 54)
    X = X[:100]
    Y = Y[:100]

    sFreq = 200
    newSFreq = 1000
    X_2 = []
    for s in range(len(X)):
        print(s)
        x2Shape = list(X[s].shape)
        x2Shape[2] = int(900 * (newSFreq / sFreq))
        x = np.zeros(x2Shape)
        for sample in range(len(X[s])):
            for channel in range(8):
                x[sample][channel] = scipy.signal.resample(X[s][sample][channel], int(900 * (newSFreq / sFreq)))
        X_2.append(x)
    X = np.array(X_2, dtype="object")
    Q = 30  # Quality factor for notch filter
    f0 = 50  # Frequency to be removed (Hz)
    harmonic_freqs = np.arange(f0, 101, f0)
    sfreq = 1000  # 200
    for s in range(len(X)):
        for sample in range(len(X[s])):
            for channel in range(8):
                X[s][sample][channel] = bandPassFilter(X[s][sample][channel], 1, 150, sfreq)
                for harmonic in harmonic_freqs:
                    X[s][sample][channel] = comb_filter(X[s][sample][channel], harmonic, Q, sfreq)
                # envelope
                X[s][sample][channel] = np.abs(X[s][sample][channel])
                X[s][sample][channel] = lowpassFilter(X[s][sample][channel], 75, sfreq)
    return X, Y
