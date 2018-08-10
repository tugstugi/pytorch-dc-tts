"""These methods are copied from https://github.com/Kyubyong/dc_tts/"""

import os
import copy
import librosa
import scipy.io.wavfile
import numpy as np

from tqdm import tqdm
from scipy import signal
from hparams import HParams as hp


def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag ** hp.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")


def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def save_to_wav(mag, filename):
    """Generate and save an audio file from the given linear spectrogram using Griffin-Lim."""
    wav = spectrogram2wav(mag)
    scipy.io.wavfile.write(filename, hp.sr, wav)


def preprocess(dataset_path, speech_dataset):
    """Preprocess the given dataset."""
    wavs_path = os.path.join(dataset_path, 'wavs')
    mels_path = os.path.join(dataset_path, 'mels')
    if not os.path.isdir(mels_path):
        os.mkdir(mels_path)
    mags_path = os.path.join(dataset_path, 'mags')
    if not os.path.isdir(mags_path):
        os.mkdir(mags_path)

    for fname in tqdm(speech_dataset.fnames):
        mel, mag = get_spectrograms(os.path.join(wavs_path, '%s.wav' % fname))

        t = mel.shape[0]
        # Marginal padding for reduction shape sync.
        num_paddings = hp.reduction_rate - (t % hp.reduction_rate) if t % hp.reduction_rate != 0 else 0
        mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
        mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
        # Reduction
        mel = mel[::hp.reduction_rate, :]

        np.save(os.path.join(mels_path, '%s.npy' % fname), mel)
        np.save(os.path.join(mags_path, '%s.npy' % fname), mag)
