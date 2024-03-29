# audio-slice-replacer

Takes a collection of audio files (_sources_) and a single longer audio file (_target_) and slices them all up. Then it replaces each slice in the target with the most similar slice from the sources, where "most similar" is determined by feature extraction and a 1-nearest-neighbor search.

Run from `slicer.py`. Edit variables at the top of the file. Change stuff around in `feature_extraction.py` if you want to change what features are used, but be careful to make sure that every signal input ends up with a feature vector of the same size.

N.B. The source should be a path to a folder, and all sound files in that folder (and all subdirectories) will be included in the sources. If a source is sufficiently long (8 seconds by default, editable by parameter) it will be sliced up, but otherwise it will be considered a source slice on its own. The path to the target should go directly to one file.

Requires scikit-learn (for PCA and k-nearest-neighbors) and librosa (for onset detection / feature extraction).
