import py_datatools.datatools as dt
import numpy as np

tracks, info_set = dt.load_whole_named_dataset('6_tracklets_large_calib')

e_tracks = tracks[info_set[:,0]==1]
e_info_set = info_set[info_set[:,0]==1]

p_tracks = tracks[info_set[:,0]==0]
p_info_set = info_set[info_set[:,0]==0]

num_electrons = len(e_tracks)
num_train_electrons = int(num_electrons)
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
dt.save_dataset('6_tracklets_large_calib_even', train_tracks, train_info_set, -1)