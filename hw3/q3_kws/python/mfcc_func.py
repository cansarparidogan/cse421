import os
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig

def _read_wav_8k(path, target_sr=8000):
    sr, x = wav.read(path)
    if x.ndim > 1:
        x = x[:,0]
    x = x.astype(np.float32)
    if sr != target_sr:
        n = int(len(x) * (target_sr / sr))
        x = sig.resample(x, n).astype(np.float32)
        sr = target_sr
    x = np.clip(x, -32768, 32767)
    return sr, x

def _mfcc_13_per_frame(x, sr, n_fft, n_mels, n_mfcc, window):
    import librosa
    x = x / 32768.0
    m = librosa.feature.mfcc(
        y=x,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=n_fft//2,
        win_length=n_fft,
        window=window,
        n_mels=n_mels,
        fmin=20,
        fmax=4000
    )
    return m

def create_mfcc_features(record_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window):
    feats=[]
    labels=[]
    for root, fname in record_list:
        path=os.path.join(root, fname) if root is not None else fname
        sr, x = _read_wav_8k(path, sample_rate)
        m = _mfcc_13_per_frame(x, sr, FFTSize, numOfMelFilters, numOfDctOutputs, "hamming")
        mu = np.mean(m, axis=1)
        sd = np.std(m, axis=1)
        f = np.concatenate([mu, sd], axis=0).astype(np.float32)
        feats.append(f)
        digit = int(fname.split("_")[0])
        labels.append(digit)
    return np.stack(feats, axis=0), np.array(labels, dtype=np.int32)
