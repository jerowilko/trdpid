import py_datatools.datatools as dt
import numpy as np
import os
import glob

calib_params = {}
dataset_name = 'all_tracks_6_tracklets_valid_run_numbers'
new_dataset_name = dataset_name + '_calib'

for fil in glob.glob(os.path.dirname(__file__) + '/calib_files/*.txt'):
	print('Loading calib file:', fil)
	run_no = fil.split('/')[-1].split('.')[0]
	calib_params[run_no.split('_')[1]] = np.genfromtxt(fil, delimiter=', ')

tracks, infosets = dt.load_whole_named_dataset(dataset_name)

for run_no in set(infosets[:,13]):
	run_gains = calib_params[str(int(run_no))]
	track_gains = run_gains[infosets[infosets[:, 13] == run_no][:, 14:20].reshape(-1).astype(int)][:,3].reshape(infosets[infosets[:, 13] == run_no][:, 14:20].shape)
	track_gains = np.expand_dims(np.expand_dims(track_gains, axis=-1), axis=-1)

	tracks[infosets[:, 13] == run_no] *= track_gains

dt.save_dataset(new_dataset_name, tracks, infosets, -1)