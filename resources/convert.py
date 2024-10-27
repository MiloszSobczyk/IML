import os
import numpy as np
import noisereduce as nr
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks

def STFT(sig, frameSize, overlapFactor=0.75, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFactor * frameSize))
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

def LogarithmicScale(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])], axis=1)

    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = [np.mean(allfreqs[int(scale[i]):int(scale[i + 1])]) if i < len(scale) - 1 else np.mean(allfreqs[int(scale[i]):]) for i in range(len(scale))]

    return newspec, freqs

def Normalize(samples):
    """Normalize audio samples to be between -1 and 1."""
    maxVal = np.max(np.abs(samples))
    if maxVal > 0:
        return samples / maxVal
    return samples

def RemoveSilence(samples, threshold=0.01):
    """Remove silent passages from the audio signal."""
    non_silent_indices = np.where(np.abs(samples) > threshold)[0]
    return samples[non_silent_indices] if non_silent_indices.size > 0 else np.array([])

def PlotSTFT(samples, sampleRate, outputFilePath, binSize, colormap, minDB=-80):
    s = STFT(samples, binSize)
    sShow, _ = LogarithmicScale(s, factor=1.0, sr=sampleRate)

    ims = 20. * np.log10(np.abs(sShow) / 10e-6)
    ims = np.maximum(ims, minDB)

    plt.figure(figsize=(20, 10))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="nearest")
    plt.axis("off")

    plt.savefig(outputFilePath, bbox_inches="tight", pad_inches=0)
    plt.close()

def ProcessAndDenoiseDirectory(rootDirPath, denoise=True, binSize=512, colormap="gray", segmentDuration=2, 
                                noiseSampleDuration=0.5, propDecrease=0.5):
    for dirPath, _, fileNames in os.walk(rootDirPath):
        if 'spectrograms' in dirPath or 'daps' not in dirPath:
            continue
        outputDirPath = (rootDirPath + 'spectrograms/' + dirPath[7:]).replace('\\', '/')
        os.makedirs(outputDirPath, exist_ok=True)

        for fileName in fileNames:
            if not fileName.endswith('.wav'):
                continue

            inputFilePath = os.path.join(dirPath, fileName).replace('\\', '/')
            sampleRate, samples = wav.read(inputFilePath)

            samples = Normalize(samples)
            samples = RemoveSilence(samples)

            if denoise:
                noiseSamples = int(noiseSampleDuration * sampleRate)
                noiseSample = samples[:noiseSamples]

                reducedNoiseSamples = nr.reduce_noise(y=samples, sr=sampleRate, y_noise=noiseSample, prop_decrease=propDecrease)
            else:
                reducedNoiseSamples = samples

            segmentSamples = segmentDuration * sampleRate
            segmentsCount = int(np.ceil(len(reducedNoiseSamples) / segmentSamples))

            for i in range(segmentsCount):
                start = i * segmentSamples
                end = min(start + segmentSamples, len(reducedNoiseSamples))
                segment = reducedNoiseSamples[start:end]

                outputFilePath = os.path.join(outputDirPath, f"{fileName[:-4]}_spectrogram_{i + 1}.png")
                PlotSTFT(segment, sampleRate, outputFilePath, binSize=binSize, colormap=colormap)

ProcessAndDenoiseDirectory('./', denoise=True)
