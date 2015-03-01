import essentia
import essentia.standard
import essentia.streaming
import matplotlib.pyplot as plt
from essentia.standard import *
from essentia.streaming import *
import numpy as np
import os

def hzToCents(freq, tuningFreq = 440.0):
    # freq[freq < eps] = eps
    # return 1200.0 * np.log2(freq / tuningFreq)
    cents = freq.copy()
    for i in range(len(freq)):
        if freq[i] > 0:
            cents[i] = 1200.0 * np.log2(freq[i] / tuningFreq)
    return cents

def diffAvg(x, n, aRange = 16):
    df = np.zeros(n)
    for i in range(n):
        rb = min(n, i+aRange/2)
        re = min(n, i+aRange/2+aRange)
        lb = max(0, i-aRange/2-aRange)
        le = max(0, i-aRange/2)
        rmean = np.mean(x[rb : re])
        lmean = np.mean(x[lb : le])
        df[i] = rmean - lmean
    return df

def peakDetection(x, n, minDur = 5):
    minDurH = minDur / 2
    peaks = np.zeros(n)
    for i in range(n):
        if abs(x[i]) == np.max(abs(x[max(0, i - minDurH) : min(n, i + minDurH + 1)])):
            peaks[i] = abs(x[i])
            peaks[max(0, i - minDur) : i] = 0  # smooth
    return peaks


eps = np.finfo(float).eps
fSize = 2048
hSize = 128
sr = 44100
vt = 1.4
minDur = 5  # frame range for peak detection, odd
minDurH = minDur / 2
pThr = 60  # pitch threshhold
eThr = 0.08  # energy threshold
wavfile = 'melody12.wav'

loader = essentia.standard.MonoLoader(filename = './melodies/' + wavfile)
audio = loader()

predominantMelody = essentia.standard.PredominantMelody(frameSize=fSize, hopSize=hSize, sampleRate=sr, voicingTolerance=vt)
pitchH, pitchConfidence = predominantMelody(audio)

n = len(pitchH)
t = np.arange(n, dtype=float) * hSize / sr
t_p = np.empty([n, 2])
for i in range(n):
    t_p[i][0] = t[i]
    t_p[i][1] = pitchH[i]
np.savetxt(wavfile[:-4]+ '.csv', t_p, delimiter=",")
# subprocess.call(['python melosynth.py', '--fs 44100 ' + wavfile[-4] + '.csv'])
command = 'python melosynth.py --fs 44100 ' + wavfile[:-4] + '.csv'
os.system(command)

pitch = hzToCents(pitchH)
onsets = np.zeros(n)

# f0 variation
df0 = diffAvg(pitch, n)
peaks = peakDetection(df0, n)

f, plots = plt.subplots(3)
plots[0].set_title('sample')
plots[0].plot(np.arange(len(audio), dtype=float) / sr, audio)
plots[1].set_title('pitch')
plots[1].plot(t, pitch)
plots[2].set_title('absolute pitch vatiation')
plots[2].plot(t, abs(df0))  # pitch values are non-negative

segP = np.zeros(peaks.shape[0])
melP = []  # list of [onset, duration, frequency]
melOn = -1
melDur = -1
melF = -1
onsetMark = 0
for i in range(peaks.shape[0]):
    if peaks[i] > pThr:
        onsets[i] = 1
        melF = np.median(pitchH[onsetMark : i])
        segP[onsetMark : i] = melF
        if melF > eps:
            melOn = t[onsetMark]
            melDur = t[i] - t[onsetMark]
            melP.append([melOn, melDur, melF])
        onsetMark = i
        plots[0].axvline(t[i], color='g')
        plots[1].axvline(t[i], color='g')
        plots[2].axvline(t[i], color='g')

annoFile = open(wavfile[:-4] + '_anno_pitch.txt', 'w')
for i in range(len(melP)):
    annoFile.write(str(melP[i][0]) + ',' + str(melP[i][1]) + ',' + str(melP[i][2]) + '\n')
annoFile.close()
t_pS = np.empty([n, 2])
for i in range(n):
    t_pS[i][0] = t[i]
    t_pS[i][1] = segP[i]
np.savetxt(wavfile[:-4]+ '_pitchSeg.csv', t_pS, delimiter=",")
# subprocess.call(['python melosynth.py', '--fs 44100 ' + wavfile[-4] + '.csv'])
command = 'python melosynth.py --fs 44100 ' + wavfile[:-4] + '_pitchSeg.csv'
os.system(command)



# energy variation
rms = essentia.standard.RMS()
w = essentia.standard.Windowing(type = 'hann')
energy = []
for frame in FrameGenerator(audio, frameSize = fSize, hopSize = hSize):
    energy.append(rms(w(frame)))
# n = len(energy)
energy = np.asarray(energy)
# dEng = diffAvg(energy, n)
# dEng[dEng < eps] = eps
# logdEng = np.log10(dEng)
energy[energy < eps] = eps
logEng = np.log10(energy)
dEng = diffAvg(logEng, n)
# peaks = peakDetection(logdEng, n)
peaks = peakDetection(dEng, n)

f2, plots2 = plt.subplots(4)
plots2[0].set_title('sample')
plots2[0].plot(np.arange(len(audio), dtype=float) / sr, audio)
plots2[1].set_title('energy')
plots2[1].plot(t, energy)
plots2[2].set_title('log energy')
plots2[2].plot(t, logEng)
plots2[3].set_title('log absolute energy vatiation')
# plots2[3].plot(t, abs(logdEng))
plots2[3].plot(t, abs(dEng))

segE = np.zeros(peaks.shape[0])
onsetMark = 0
for i in range(peaks.shape[0]):
    if peaks[i] > eThr:
        onsets[i] = 1
        segE[onsetMark : i] = np.median(pitchH[onsetMark : i])
        onsetMark = i
        plots2[0].axvline(t[i], color='g')
        plots2[1].axvline(t[i], color='g')
        plots2[2].axvline(t[i], color='g')
        plots2[3].axvline(t[i], color='g')

t_pE = np.empty([n, 2])
for i in range(n):
    t_pE[i][0] = t[i]
    t_pE[i][1] = segE[i]
np.savetxt(wavfile[:-4]+ '_energySeg.csv', t_pE, delimiter=",")
# subprocess.call(['python melosynth.py', '--fs 44100 ' + wavfile[-4] + '.csv'])
command = 'python melosynth.py --fs 44100 ' + wavfile[:-4] + '_energySeg.csv'
os.system(command)

# remove onsets that are too close
for i in range(len(onsets)):
    if onsets[i] == 1:
        onsets[i-2 : i] = 0

t_pMerge = np.empty([n, 2])
onsetMark = 0
for i in range(n):
    t_pMerge[i, 0] = t[i]
    if onsets[i] == 1:
        t_pMerge[onsetMark : i, 1] = np.median(pitchH[onsetMark : i])
        onsetMark = i

np.savetxt(wavfile[:-4]+ '_mergeSeg.csv', t_pMerge, delimiter=",")
# subprocess.call(['python melosynth.py', '--fs 44100 ' + wavfile[-4] + '.csv'])
command = 'python melosynth.py --fs 44100 ' + wavfile[:-4] + '_mergeSeg.csv'
os.system(command)

plt.show()
