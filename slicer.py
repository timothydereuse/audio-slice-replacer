import librosa as rosa
import numpy as np
import soundfile
import os
import feature_extraction as fe
import audio_slice as ausl
import nn_ind_mapping as nim
from importlib import reload
reload(ausl)
reload(fe)
reload(nim)


music_data_path = r'C:\Users\Tim\Documents\MUSIC DATA'
targets_path = rf'{music_data_path}\raw materials'
sources_path = rf'{music_data_path}\short weird perx'
ftypes = ['wav', 'aiff', 'aif']

sources_fnames = []
for path, folder, files in os.walk(sources_path):
    for file in files:
        if not file.split('.')[-1] in ftypes:
            continue
        sources_fnames.append(rf'{path}\{file}')

targets_fnames = []
for path, folder, files in os.walk(targets_path):
    for file in files:
        if not file.split('.')[-1] in ftypes:
            continue
        targets_fnames.append(rf'{path}\{file}')

# get features for every target file
sources = []
for i, fname in enumerate(sources_fnames):
    if not i % 20:
        print(f'calculating features for source {i} of {len(sources_fnames)}...')

    s, sr = soundfile.read(fname)
    if len(s.shape) > 1:
        s = np.mean(s, 1)
    sources.append(ausl.AudioSlice(s, sr, fname))

# perform onset detection and get features for each detected slice
choose_loop = rf"{targets_path}/terrible table vamp SHUREMIC1.wav"
y, sr = soundfile.read(choose_loop)
if len(y.shape) > 1:
    y = np.mean(y, 1)

onsets = rosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
onset_times = rosa.frames_to_samples(onsets)
onset_times = np.concatenate([onset_times, [len(y)]])

segmented = [y[onset_times[n]:onset_times[n+1]] for n in range(len(onset_times) - 1)]

targets = []
for i, s in enumerate(segmented):
    if not i % 20:
        print(f'calculating features for target {i} of {len(segmented)}...')
    targets.append(ausl.AudioSlice(s, sr, None))

# gather features and perform index mapping between source and target
keys_sorted = sorted(targets[0].feats.keys())
Y = [[s.feats[k] for k in keys_sorted] for s in sources]
X = [[s.feats[k] for k in keys_sorted] for s in targets]
ind_mapping = nim.nn_ind_mapping(X, Y)

output = np.zeros(y.shape)
for ind in range(len(ind_mapping)):
    replace_audio = sources[ind_mapping[ind]].audio
    start_time = onset_times[ind]
    end_time = onset_times[ind+1]
    dur = end_time - start_time

    if len(replace_audio) < end_time - start_time:
        end_time = start_time + len(replace_audio)

    output[start_time:end_time] = replace_audio[:dur]

soundfile.write('output.wav', output, sr)


