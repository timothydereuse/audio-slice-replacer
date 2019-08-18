import librosa as rosa
import numpy as np


def extract_features_from_audio(inp, sr, len_segments=1024, n_segments=4):
    n_fft = len_segments * 2
    hop = len_segments

    if max(inp) > 0:
        inp = np.clip(inp / max(inp), 0, 1)

    if len(inp) < len_segments * n_segments:
        extend = (len_segments * n_segments) - len(inp)
        inp = np.concatenate([inp, np.zeros(extend)])

    # the built-ins, first
    feats = {}
    feats['spec_centroid'] = rosa.feature.spectral_centroid(inp, sr, n_fft=n_fft, hop_length=hop)
    # feats['spec_contrast'] = rosa.feature.spectral_contrast(inp, sr)
    feats['spec_bandwidth'] = rosa.feature.spectral_bandwidth(inp, sr, n_fft=n_fft, hop_length=hop)
    feats['spec_flatness'] = rosa.feature.spectral_flatness(inp, n_fft=n_fft, hop_length=hop)
    feats['spec_rolloff'] = rosa.feature.spectral_rolloff(inp, sr, n_fft=n_fft, hop_length=hop)
    feats['rms'] = rosa.feature.rms(inp, frame_length=n_fft, hop_length=hop)

    # all_poly = rosa.feature.poly_features(inp, sr, n_fft=n_fft, hop_length=hop, order=2)
    # for i, x in enumerate(all_poly):
    #     feats[f'polyfit_{i}'] = x

    flat_feats = {}
    for key in feats.keys():
        feat_seq = feats[key].ravel()

        for i in range(n_segments):
            try:
                flat_feats[f'{key}_{i}'] = feat_seq[i]
            except IndexError:
                flat_feats[f'{key}_{i}'] = 0

    return flat_feats