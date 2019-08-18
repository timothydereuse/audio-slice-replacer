import feature_extraction as fe
from importlib import reload
reload(fe)

class AudioSlice(object):

    def __init__(self, audio, sr, fname=None):
        self.feats = fe.extract_features_from_audio(audio, sr)
        self.fname = fname
        self.audio = audio
        self.sr = sr
