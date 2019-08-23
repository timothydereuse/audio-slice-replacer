import librosa as rosa
import numpy as np


def extract_features_from_audio(inp, sr, len_segments=2**10, n_segments=4, normalize=True):
    n_fft = len_segments * 2
    hop = len_segments

    # normalize sound before feature extraction
    if max(inp) > 0 and normalize:
        inp = np.clip(inp / max(inp), 0, 1)

    # extend with 0s if we are trying to get more feature frames than this is long
    if len(inp) < len_segments * n_segments:
        extend = (len_segments * n_segments) - len(inp)
        inp = np.concatenate([inp, np.zeros(extend)])


    feats = {}
    # comment out lines to exclude items from the featureset
    feats['spec_centroid'] = rosa.feature.spectral_centroid(inp, sr, n_fft=n_fft, hop_length=hop)
    feats['spec_contrast'] = rosa.feature.spectral_contrast(inp, sr)
    feats['spec_bandwidth'] = rosa.feature.spectral_bandwidth(inp, sr, n_fft=n_fft, hop_length=hop)
    feats['spec_flatness'] = rosa.feature.spectral_flatness(inp, n_fft=n_fft, hop_length=hop)
    feats['spec_rolloff'] = rosa.feature.spectral_rolloff(inp, sr, n_fft=n_fft, hop_length=hop)
    feats['rms'] = rosa.feature.rms(inp, frame_length=n_fft, hop_length=hop)
    feats['zero_crossing'] = rosa.feature.zero_crossing_rate(inp, frame_length=n_fft, hop_length=hop)

    flat_feats = {}
    for key in feats.keys():
        feat_seq = feats[key].ravel()

        for i in range(n_segments):
            try:
                flat_feats[f'{key}_{i}'] = feat_seq[i]
            except IndexError:
                flat_feats[f'{key}_{i}'] = 0

    return flat_feats