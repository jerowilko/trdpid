
if __name__ == '__main__':
    from py_datatools.datatools import *
    import matplotlib.pyplot as plt
    import numpy as np

    tracks, infosets = load_whole_default_dataset()
    labels = infosets[:, 0]