import librosa as rosa
import numpy as np
import soundfile
import os
import datetime
from pathlib import Path
import feature_extraction as fe
import audio_slice as ausl
import nn_ind_mapping as nim
from importlib import reload
reload(ausl)
reload(fe)
reload(nim)

ftypes = ['wav', 'aiff', 'aif']     # only bother with files of these types

# path to folder of sound sources. ALL sound files in here will be added to the possible sources.
sources_path = r"C:\Users\Tim\Documents\MUSIC DATA\short misc samples,fx\BoulangerTamTamReplacements22"

# path to target file - this will be sliced, and sound slices from the sources will be matched to it
target_file = r"C:\Users\Tim\Documents\MUSIC DATA\Loops\Drumdays-Vol-02-Part1-b\primus_tommythecat_live3solo.wav"

match_volume = True         # attempt to match the energy between each source slice and the slice its replacing
include_reverses = False    # include the reverse of each slice. gonna be honest w/u. its not that fun
pca_reduce_amt = 3          # strength of dimensionality reduction on extracted features. higher = messier
                            # categorization by the knn but more information included
slice_threshold_secs = 8    # if a source is longer than this number of seconds, then slice it up before
                            # adding it to the pool of source audio clips
length_limit_secs = 30      # if a source is longer than this number of seconds, then discard anything
                            # past this point so you don't accidentally slice up a 20 min file
declick_amt = 15            # use a linear envelope of this many samples to declick slices.

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
out_fname = f'output_{timestamp}.wav'   # output filename with timestamp

poll_every = 50             # controls how often console output is produced when calculating features


def slice_long_sample(y, sr, declick_samples=15, length_limit=None, fname=''):

    if length_limit and len(y) / sr > length_limit:
        y = y[0:length_limit * sr]

    onsets = rosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = rosa.frames_to_samples(onsets)
    onset_times = np.concatenate([onset_times, [len(y)]])
    segmented = [y[onset_times[n]:onset_times[n + 1]] for n in range(len(onset_times) - 1)]

    segmented = [s for s in segmented if len(s) >= declick_samples]

    if declick_samples > 1:
        declick_envelope = np.linspace(1 / declick_samples, 1 - (1 / declick_samples), declick_samples)
        for i in range(len(segmented)):
            segmented[i][0:declick_samples] *= declick_envelope

    slices = []
    for i, s in enumerate(segmented):
        if not i % poll_every and i > 1:
            print(rf'calculating features for slice {i}/{len(segmented)} of {fname}...')
        slices.append(ausl.AudioSlice(s, sr, fname))

    return slices, onset_times


def get_fnames_from_directory(path):
    fnames = []
    for directory, folder, files in os.walk(sources_path):
        for file in files:
            if not file.split('.')[-1] in ftypes:
                continue
            fnames.append(Path(directory) / file)

    if not fnames:
        raise FileNotFoundError(f'directory {path} not found!')

    return fnames


sources_fnames = get_fnames_from_directory(sources_path)

# get features for every target file
sources = []
for i, fname in enumerate(sources_fnames):
    if not i % poll_every:
        print(f'calculating features for source {i} of {len(sources_fnames)}...')

    s, sr = soundfile.read(fname)
    if len(s.shape) > 1:
        s = np.mean(s, 1)

    if len(s) / sr < slice_threshold_secs:
        sources.append(ausl.AudioSlice(s, sr, fname))
        continue

    print(f'slicing source {fname}...')
    slices, _ = slice_long_sample(s, sr, fname=fname, declick_samples=declick_amt, length_limit=length_limit_secs)
    sources += slices

if include_reverses:
    print('calculating features for reverse slices...')
    sources += [x.get_rev_slice() for x in sources]

# perform onset detection and get features for each detected slice
y, sr = soundfile.read(target_file)
if len(y.shape) > 1:
    y = np.mean(y, 1)
targets, onset_times = slice_long_sample(y, sr, fname=target_file, declick_samples=declick_amt)

print('performing assignment with knn...')
# gather features and perform index mapping between source and target
keys_sorted = sorted(targets[0].feats.keys())
Y = [[s.feats[k] for k in keys_sorted] for s in sources]
X = [[s.feats[k] for k in keys_sorted] for s in targets]
ind_mapping = nim.nn_ind_mapping(X, Y, pca_reduce=pca_reduce_amt)

print('replacing slices in target with slices from source...')
output = np.zeros(y.shape)
for ind in range(len(ind_mapping)):

    # slice from target audio file
    orig_audio = targets[ind].audio

    # identified matching slice from source audio file(s)
    replace_audio = sources[ind_mapping[ind]].audio

    # match volume of orig and replace slices. this uses RMS for volume... kinda works
    if match_volume:
        orig_nrg = np.mean(orig_audio * orig_audio)
        repl_nrg = np.mean(replace_audio * replace_audio)
        nrg_ratio = np.sqrt(orig_nrg / repl_nrg)
        replace_audio *= nrg_ratio

    # use the original start and end time of the target slice. cut it down if it's too long
    start_time = onset_times[ind]
    end_time = onset_times[ind+1]
    dur = end_time - start_time

    if len(replace_audio) < end_time - start_time:
        end_time = start_time + len(replace_audio)

    # slot it ni
    output[start_time:end_time] = replace_audio[:dur]

print('writing to file...')
soundfile.write(out_fname, output, sr)

