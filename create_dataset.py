"""Create hdf5 dataset from mne-python epochs"""

import numpy as np
import os
import re
import glob
import os.path as op
from mne import read_epochs
import h5py

N_EPOCHS = 10
N_TIMES = 501
N_SEN = 204


def get_subj(fif_file, n_epochs):
    """Read epochs from single subject and create labels"""
    ep = read_epochs(fif_file).pick_types(meg='grad')
    data = ep.get_data()
    # if data.shape[0] > n_epochs:
    #     data = data[:n_epochs, :, :]

    match = re.search(r'([KR])\d{4}', fif_file)
    if match:
        label = match.group(1)
    else:
        print('ERROR')

    if label == 'K':
        labels = np.ones(data.shape[0])
    elif label == 'R':
        labels = np.zeros(data.shape[0])

    return data, labels


def get_data(main_path, cond, h5_fname):
    """Use hdf5 to avoid memory overfill by writing data as they arrive"""
    with h5py.File(h5_fname, "w") as data_5:

        subj_paths = [op.join(main_path, f) for f in os.listdir(main_path)
                      if op.isdir(op.join(main_path, f))]

        # fif_ep_files = glob.glob(subj_paths[0] + '/*' + cond + '-epo.fif')

        subj_paths_filt = [s for s in subj_paths
                           if glob.glob(s + '/*' + cond + '-epo.fif')]

        subj_names = [op.split(s)[1] for s in subj_paths_filt]

        XX = data_5.create_dataset(
                "X", [0, N_SEN, N_TIMES], maxshape=(None, N_SEN, N_TIMES))

        yy = data_5.create_dataset("y", [0, ], maxshape=(None,))

        for i, subj_path in enumerate(subj_paths_filt):
            fif_ep_files = glob.glob(subj_path + '/*' + cond + '-epo.fif')
            fif_file = fif_ep_files[0]
            data, label = get_subj(fif_file, N_EPOCHS)
            if data.shape[2] == 501:
                try:
                    n_epochs_now = XX.shape[0]
                    n_epochs_new = data.shape[0]
                    XX.resize(n_epochs_now + n_epochs_new, axis=0)
                    XX[n_epochs_now:n_epochs_now + n_epochs_new, :, :] = data

                    yy.resize(n_epochs_now + n_epochs_new, axis=0)
                    yy[n_epochs_now:n_epochs_now + n_epochs_new] = label
                    data_5.attrs[subj_names[i]] = list(range(n_epochs_now, n_epochs_now + n_epochs_new))
                except TypeError:
                    raise TypeError(
                            'data shape is {} for {}'.format(
                                str(data.shape), subj_path))

                # labels.append(label)
            else:
                raise ValueError(
                        'data shape is {} for {}'.format(
                            str(data.shape), subj_path))


if __name__ == '__main__':

    h5_fname = 'X.hdf5'
    get_data('/home/dmalt/Data/aut_gamma/Moscow_baseline_results_new/',
             'ec', h5_fname)
