import os
from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm
import pickle
import pandas as pd
import jams
import numpy as np
import muda

audio_path = os.path.join('../../..', 'marl_audio')
jams_path = os.path.join('../../..', 'marl_jams')
output_dir = os.path.join('../../..', 'marl_audio_jams_augmentation')

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

def root(x):
    return os.path.splitext(os.path.basename(x))[0]

AUDIO = [audio_path + '/' + file for file in os.listdir(audio_path)]
ANNOS = [jams_path + '/' + file for file in os.listdir(jams_path)]

AUDIO.sort()
ANNOS.sort()

# Make sure there are the same number of files
assert len(ANNOS) == len(AUDIO)
# And that they're in agreement
assert all([root(_1) == root(_2) for (_1, _2) in zip(AUDIO, ANNOS)])

def augment(afile, jfile, deformer, outpath, is_stretch):
    jam = muda.load_jam_audio(jfile, afile)

    base = root(afile)
    outfile = os.path.join(outpath, base)

    marker = ""
    if is_stretch:
        marker = "time_"

    for i, jam_out in enumerate(deformer.transform(jam)):
        muda.save('{}.{}{}.flac'.format(outfile, marker, i),
                  '{}.{}{}.jams'.format(outfile, marker, i),
                  jam_out, strict=False)

# Create the augmentation engine
pitcher = muda.deformers.PitchShift(n_semitones=[-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6])
# Pitch-shifting data augmentation
Parallel(n_jobs=10, verbose=10)(delayed(augment)(aud, jam, pitcher, output_dir, 0) for (aud, jam) in zip(AUDIO, ANNOS))

print("Done.")

