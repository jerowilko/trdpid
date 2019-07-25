import py_datatools.datatools as dt
import numpy as np
import os
import glob

calib_params = {}
dataset_name = 'all_tracks_6_tracklets_valid_run_numbers'
new_dataset_name = dataset_name + '_calib'

for fil in glob.glob(os.path.dirname(__file__) + '/calib_files/combined_local_gains*.txt'):
	print('Loading calib file:', fil)
	run_no = fil.split('_')[-1].split('.')[0]
	calib_params[run_no] = np.genfromtxt(fil, delimiter=', ')[:,2:].reshape((540, 16, 144))

tracks, infosets = dt.load_whole_named_dataset(dataset_name)

for i in range(len(tracks)):
	run_no = infosets[i,13]
	run_gains = calib_params[str(int(run_no))]

	dets = infosets[i,14:20].astype(int)
	rows = infosets[i,21:27].astype(int)
	cols = infosets[i,28:34].astype(int)

	track_gains = np.expand_dims(run_gains[dets,rows,[cols - 8 + i for i in range(17)]].T, axis=-1)

	tracks[i] *= track_gains

	print("%d / %d" % (i, len(tracks)))

dt.save_dataset(new_dataset_name, tracks, infosets, -1)