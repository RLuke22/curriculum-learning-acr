import os 
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True) 
from tqdm import tqdm 

INPUT_PATH = "../../MARL/marl_data/pump"

# First, we get track-level statistics
track_lists = []
tot_li = [0] * 14
for name in tqdm([os.path.join(INPUT_PATH, "{}.npz".format(i)) for i in range(1217)]):
    npz = np.load(name)

    li = [0] * 14
    chord_tag = npz['chord_tag/chord'][0,:,0]
    chord_tag[chord_tag < 168] = chord_tag[chord_tag < 168] % 14

    for j in range(14):
        li[j] += np.sum(chord_tag == j) 
        tot_li[j] += np.sum(chord_tag == j)

    track_lists.append(li)

tot_li = np.array(tot_li).reshape((1,14))
track_lists = np.array(track_lists)

# Normalize all dimensions
track_lists = track_lists / tot_li

# Convert back into list of lists, where sorting can be done easier
track_lists = [list(track_lists[i]) for i in range(1217)]
for i in range(1217):
    track_lists[i].append(i)

folds = {0: [], 1: [], 2: [], 3: [], 4: []}
fold_profiles = {0: np.array([0]*14), 1: np.array([0]*14), 2: np.array([0]*14), 3: np.array([0]*14), 4: np.array([0]*14)}

# Now perform stratification algorithm
# The first index we "balance" is index 11 (minmaj7) as this is the rarest chord class in the dataset.

exhausted_idxs = []
for i in tqdm(range(1217 // 5)):
    if i == 0:
        sorted_list = sorted(track_lists, reverse=True, key=lambda x: (x[11]))

        for j in range(5):
            folds[j].append(sorted_list[j][-1])
            fold_profiles[j] = fold_profiles[j] + np.array(sorted_list[j][0:14])

            track_lists = list(filter(lambda x: x[-1] != sorted_list[j][-1], track_lists))

    else:
        # Check for exhausted indices
        track_lists_np = np.array(track_lists)
        remaining_props = np.sum(track_lists_np, axis=0)
        
        for i in range(14):
            if i not in exhausted_idxs and remaining_props[i] == 0:
                exhausted_idxs.append(i)

        # calculate maximum variance chord quality
        max_var = 0
        max_var_idx = -1
        for j in range(14):
            if j in exhausted_idxs:
                continue
            
            quality_proportions = np.array([fold_profiles[0][j], fold_profiles[1][j], fold_profiles[2][j], fold_profiles[3][j], fold_profiles[4][j]])
            var = np.var(quality_proportions)

            if var > max_var:
                max_var = var 
                max_var_idx = j 
        
        # Now that max_var index is found, sort track_lists by max_var chord quality in ASCENDING order
        sorted_list = sorted(track_lists, reverse=False, key=lambda x: (x[max_var_idx]))

        fold_props = np.array([fold_profiles[i][max_var_idx] for i in range(5)])
        sorted_idxs = np.argsort(fold_props)

        max_fold = sorted_idxs[-1]

        # Add minimum track to maximum fold
        fold_profiles[max_fold] = fold_profiles[max_fold] + np.array(sorted_list[0][0:14])
        folds[max_fold].append(sorted_list[0][-1])
        track_lists = list(filter(lambda x: x[-1] != sorted_list[0][-1], track_lists))

        # Now must conduct linear search to find "best" track to add to each remaining fold.
        folds_remaining = [i for i in sorted_idxs if i != max_fold]
        for fold in folds_remaining:
            current_prop = fold_profiles[fold][max_var_idx]
            target_prop = fold_profiles[max_fold][max_var_idx] - current_prop 

            best_diff = float("inf")
            best_track = 0
            for track in track_lists:
                track_diff = abs(target_prop - track[max_var_idx])
                if track_diff < best_diff:
                    best_diff = track_diff 
                    best_track = track

            folds[fold].append(best_track[-1])
            fold_profiles[fold] = fold_profiles[fold] + np.array(best_track[0:14])
            track_lists = list(filter(lambda x: x[-1] != best_track[-1], track_lists))

folds[0].append(track_lists[0][-1])
folds[1].append(track_lists[1][-1])

OUTPUT_DIR = "stratified_folds"
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
for i in range(5):
    pd.DataFrame(np.array(folds[i])).to_csv(os.path.join(OUTPUT_DIR, "test{:02d}.csv".format(i)), header=None, index=None)

    train = []
    for j in [k for k in range(5) if k != i]:
        train = train + folds[j]
    print(len(train))

    pd.DataFrame(np.array(train)).to_csv(os.path.join(OUTPUT_DIR, "train{:02d}.csv".format(i)), header=None, index=None)

    all_tracks_len = len(set(train + folds[i]))
    print(all_tracks_len)







        



        

    
        
    




    