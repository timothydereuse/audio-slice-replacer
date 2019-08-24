import feature_extraction as fe
from importlib import reload
import numpy as np
reload(fe)

class AudioSlice(object):

    def __init__(self, audio, sr, fname=None, normalize=True):
        self.feats = fe.extract_features_from_audio(audio, sr)
        self.fname = fname
        self.audio = audio
        if normalize:
            self.audio = audio / max(np.abs(audio))
        self.sr = sr

    def get_rev_slice(self):
        new_slice = AudioSlice(self.audio[::-1], self.sr, self.fname)
        return new_slice