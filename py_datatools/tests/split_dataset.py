import py_datatools.datatools as dt
import numpy as np

tracks, info_set = dt.load_whole_named_dataset('all_tracks_6_tracklets_valid_run_numbers_calib')

train_split = 0.8

e_tracks = tracks[info_set[:,0]==1]
e_info_set = info_set[info_set[:,0]==1]

p_tracks = tracks[info_set[:,0]==0]
p_info_set = info_set[info_set[:,0]==0]

num_electrons = len(e_tracks)
num_train_electrons = int(train_split * num_electrons)
num_test_electrons = num_electrons - num_train_electrons
num_train_pions = num_train_electrons
num_test_pions = num_train_pions

e_train_tracks = e_tracks[:num_train_electrons]
e_train_info_set = e_info_set[:num_train_electrons]

p_train_tracks = p_tracks[:num_train_pions]
p_train_info_set = p_info_set[:num_train_pions]

train_tracks = np.concatenate([e_train_tracks, p_train_tracks])
train_info_set = np.concatenate([e_train_info_set, p_train_info_set])

randomize = np.arange(len(train_tracks))
np.random.shuffle(randomize)

train_tracks = train_tracks[randomize]
train_info_set = train_info_set[randomize]
dt.save_dataset('6_tracklets_large_calib_train', train_tracks, train_info_set, -1)

e_test_tracks = e_tracks[num_train_electrons:num_train_electrons + num_test_electrons]
e_test_info_set = e_info_set[num_train_electrons:num_train_electrons + num_test_electrons]

p_test_tracks = p_tracks[num_train_pions:num_train_electrons + num_test_pions]
p_test_info_set = p_info_set[num_train_electrons:num_train_pions + num_test_pions]

test_tracks = np.concatenate([e_test_tracks, p_test_tracks])
test_info_set = np.concatenate([e_test_info_set, p_test_info_set])

randomize = np.arange(len(test_tracks))
np.random.shuffle(randomize)

test_tracks = test_tracks[randomize]
test_info_set = test_info_set[randomize]
dt.save_dataset('6_tracklets_large_calib_testx', test_tracks, test_info_set, -1)