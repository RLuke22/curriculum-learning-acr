import os
import numpy as np 
from tqdm import tqdm
import pumpp 
import jams
import pickle
from glob import glob
from joblib import Parallel, delayed

def root(x):
    return os.path.splitext(os.path.basename(x))[0]

# All audio tracks and jams files for the MARL dataset (output of audio_augmentation.py)
PATH_TO_AUDIO_JAMS = '../../../marl_all_audio_jams'
OUTPUT_DIR = "../../../marl_pump"

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

AUDIO0 = [PATH_TO_AUDIO_JAMS + '/' + str(i) +'.mp3' for i in range(1217)]
ANNOS0 = [PATH_TO_AUDIO_JAMS + '/' + str(i) +'.jams' for i in range(1217)]

AUDIO1 = [PATH_TO_AUDIO_JAMS + '/' + file for file in os.listdir(PATH_TO_AUDIO_JAMS) if file.endswith(".flac")]
ANNOS1 = [PATH_TO_AUDIO_JAMS + '/' + file for file in os.listdir(PATH_TO_AUDIO_JAMS) if file.endswith(".jams")]

ANNOS1 = [x for x in ANNOS1 if x not in ANNOS0]

AUDIO0.sort()
ANNOS0.sort()

AUDIO1.sort()
ANNOS1.sort()

AUDIO = AUDIO0 + AUDIO1
ANNOS = ANNOS0 + ANNOS1

print(len(AUDIO), len(ANNOS))

# Make sure there are the same number of files
assert len(AUDIO) == len(ANNOS)
# And that they're in agreement
assert all([root(_1) == root(_2) for (_1, _2) in zip(AUDIO, ANNOS)])

sr = 44100
hop_length = 4096
p_feature = pumpp.feature.CQTMag(name='cqt', sr=sr, hop_length=hop_length, log=True, conv='tf', n_octaves=6)
p_chord_tag = pumpp.task.ChordTagTransformer(name='chord_tag', sr=sr, hop_length=hop_length, sparse=True)
# Pitch, Bass, Root structured components from McFee, 2017
p_chord_struct = pumpp.task.ChordTransformer(name='chord_struct', sr=sr, hop_length=hop_length, sparse=True)
pump = pumpp.Pump(p_feature, p_chord_tag, p_chord_struct)

with open(os.path.join(OUTPUT_DIR, './pump.pkl'), 'wb') as fd:
    pickle.dump(pump, fd)

def convert(aud, jam, pump, outdir):
    data = pump.transform(aud, jam)
    fname = os.path.extsep.join([root(aud), 'npz'])
    np.savez(os.path.join(outdir, fname), **data)

Parallel(n_jobs=10, verbose=10)(delayed(convert)(aud, jam, pump, OUTPUT_DIR) for (aud, jam) in zip(AUDIO, ANNOS))

print("Done.")