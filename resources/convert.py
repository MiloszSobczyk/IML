import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

""" Short-time Fourier Transform of audio signal """
def STFT(sig, frameSize, overlapFac = 0.5, window = np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # Zeros at beginning to center the first window at sample 0
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
    # Calculate columns for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # Zeros at end for full frame coverage
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape = (int(cols), frameSize), strides = (samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

""" Logarithmic scaling of the frequency axis """
def LogarithmicScale(spec, sr = 44100, factor = 20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    # Create spectrogram with new frequency bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis = 1)
        else:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])], axis = 1)

    # Calculate center frequencies for bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = [np.mean(allfreqs[int(scale[i]):int(scale[i + 1])]) if i < len(scale) - 1 else np.mean(allfreqs[int(scale[i]):]) for i in range(len(scale))]

    return newspec, freqs

""" Plot spectrogram without axes and save to file """
def PlotSTFT(inputFilePath, outputFilePath, binsize = 2 ** 10, colormap = "jet"):
    if os.path.exists(outputFilePath):
        return

    samplerate, samples = wav.read(inputFilePath)

    s = STFT(samples, binsize)
    sshow, _ = LogarithmicScale(s, factor = 1.0, sr = samplerate)

    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # Amplitude to decibel

    plt.figure(figsize = (15, 7.5))
    plt.imshow(np.transpose(ims), origin = "lower", aspect = "auto", cmap = colormap, interpolation = "none")
    plt.axis('off')

    # Save spectrogram without axes
    plt.savefig(outputFilePath, bbox_inches = "tight", pad_inches = 0)
    plt.close()

def processDirectory(rootDirPath, binsize = 2 ** 10, colormap = "jet"):
    for dirPath, _, fileNames in os.walk(rootDirPath):
        if 'spectrograms' in dirPath or 'daps' not in dirPath:
            continue
        outputDirPath = (rootDirPath + 'spectrograms/' + dirPath[7:]).replace('\\', '/')
        os.makedirs(outputDirPath, exist_ok = True)
        for fileName in fileNames:
            if not fileName.endswith('.wav'):
                continue

            inputFilePath = os.path.join(dirPath, fileName).replace('\\', '/')
            outputFilePath = os.path.join(outputDirPath, fileName[:-4] + '_spectrogram.png').replace('\\', '/')
            PlotSTFT(inputFilePath, outputFilePath, binsize=binsize, colormap = colormap)

# Generate and save the spectrogram
processDirectory('./')
