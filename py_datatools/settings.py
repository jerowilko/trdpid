import os, sys

# run_no = '000265378/'
# python_dicts_directory = os.path.dirname(__file__) + '/input/' + run_no
# datasets_home_directory = os.path.dirname(__file__) + '/output/' + run_no

python_dicts_directory = os.path.dirname(__file__) + '/raw_data/'
datasets_home_directory = os.path.dirname(__file__) + '/datasets/'

tracklet_shape = (17, 24)
info_set_size = 41
num_tracklets_in_track = 6
track_shape = (num_tracklets_in_track,) + tracklet_shape

default_num_tracklets_per_file = 9170
default_minp = None
default_maxp = None

default_dataset = 'all_tracks_6_tracklets'
