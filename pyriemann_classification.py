from numpy.core.multiarray import ndarray
from typing import List

import os
import os.path as op
import re
import glob

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
# from sklearn.decomposition import PCA
# from sklearn.model_selection import cross_val_score

import h5py

from mne import read_epochs

# from pyriemann import estimation, classification
from pyriemann.tangentspace import TangentSpace
from pyriemann.channelselection import ElectrodeSelection
from pyriemann.estimation import CospCovariances
# from pyriemann.utils import CospBoostingClassifier

N_EPOCHS = 10
N_TIMES = 501
N_SEN = 204

# from sklearn.cross_validation import cross_val_score
from utils import CospBoostingClassifier

# def shrink_cov_mat(func):
#     def wrapper(*args, **kwargs):
#         shrink_coef = 0.2
#         ccov_mat = func(*args, **kwargs)
#         for i in range(ccov_mat.shape[0]):
#             for j in range(ccov_mat.shape[3]):
#                 ccov_mat[i, :, :, j] = (1 - shrink_coef) * ccov_mat[i, :, :, j] +\
#                                        shrink_coef * np.diag(np.diag(ccov_mat[i, :, :, j]))
#         print('Hi')


#         return ccov_mat
#     return wrapper


class ShrinkCovMat():
    """Class to shrink ccov matrix"""
    def transform(self, ccov_mat):
        shrink_coef = 0.7
        for i in range(ccov_mat.shape[0]):
            for j in range(ccov_mat.shape[3]):
                ccov_mat[i, :, :, j] =\
                        (1 - shrink_coef) * ccov_mat[i, :, :, j] +\
                        shrink_coef * np.diag(np.diag(ccov_mat[i, :, :, j]))
        print('Hi')
        return ccov_mat

    def fit(self, X, y, sample_weights=None):
        return self


def get_subj(fif_file, n_epochs):
    ep = read_epochs(fif_file).pick_types(meg='grad')
    data = ep.get_data()
    if data.shape[0] > n_epochs:
        data = data[:n_epochs, :, :]

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


def get_data(main_path, cond):
    datas = []
    labels = []
    with  h5py.File("X.hdf5", "w") as data_5:

        subj_paths = [op.join(main_path,f) for f in os.listdir(main_path) if op.isdir(op.join(main_path,f))]

        # fif_ep_files = glob.glob(subj_paths[0] + '/*' + cond + '-epo.fif')

        subj_paths_filt = [s for s in subj_paths if glob.glob(s + '/*' + cond + '-epo.fif')]
        n_subj = len(subj_paths_filt)
        dset = data_5.create_dataset("X", [n_subj * N_EPOCHS, N_SEN, N_TIMES])

        for i, subj_path in enumerate(subj_paths_filt):
            # fold = os.path.join(main_path, fold)

            # if os.path.isdir(fold):
            fif_ep_files = glob.glob(subj_path + '/*' + cond + '-epo.fif')
            fif_file = fif_ep_files[0]
            data, label = get_subj(fif_file, N_EPOCHS)
            if data.shape[2] == 501:
                try:
                    dset[i * N_EPOCHS:(i + 1) * N_EPOCHS,:,:] = data
                except TypeError:
                    raise TypeError('data shape is {} for {}'.format(str(data.shape), subj_path))

                labels.append(label)
            else:
                raise ValueError('data shape is {} for {}'.format(str(data.shape), subj_path))

                # print(fif_ep_files)
        # datas = np.concatenate(datas)
        labels = np.concatenate(labels)

    # dset[...] = datas
    return dset, labels
    # return datas, labels


if __name__ == '__main__':

    #  test on 4 subjects {{{ # 
    # subj1_path = '/home/dmalt/Data/aut_gamma/Moscow_baseline_results_new/K0008'
    # subj3_path = '/home/dmalt/Data/aut_gamma/Moscow_baseline_results_new/K0001'
    # subj2_path = '/home/dmalt/Data/aut_gamma/Moscow_baseline_results_new/R0008'
    # subj4_path = '/home/dmalt/Data/aut_gamma/Moscow_baseline_results_new/R0001'
    # s1_file = os.path.join(subj1_path, 'K0008ec-epo.fif')
    # s2_file = os.path.join(subj2_path, 'R0008ec-epo.fif')
    # s3_file = os.path.join(subj3_path, 'K0001ec-epo.fif')
    # s4_file = os.path.join(subj4_path, 'R0001ec-epo.fif')
    
    # raw_s1 = read_epochs(s1_file).pick_types(meg='grad')
    # raw_s2 = read_epochs(s2_file).pick_types(meg='grad')
    # raw_s3 = read_epochs(s3_file).pick_types(meg='grad')
    # raw_s4 = read_epochs(s4_file).pick_types(meg='grad')
    
    # ep_s1 = raw_s1.get_data()[:10,:,:]
    # ep_s2 = raw_s2.get_data()[:10,:,:]
    # ep_s3 = raw_s3.get_data()[:10,:,:]
    # ep_s4 = raw_s4.get_data()[:10,:,:]
    
    # # shrink_coef = 0.2
    
    # # X = np.concatenate([raw_s1, raw_s2, raw_s3, raw_s4])
    # X = np.concatenate([ep_s1, ep_s2, ep_s3, ep_s4])
    # y_s1 = np.ones(ep_s1.shape[0])
    # y_s2 = np.zeros(ep_s2.shape[0])
    # y_s3 = np.ones(ep_s3.shape[0])
    # y_s4 = np.zeros(ep_s4.shape[0])
    
    # y = np.concatenate([y_s1, y_s2, y_s3, y_s4])
    
    # X, y = get_data('/media/dmalt/SSD500/aut_gamma/Moscow_baseline_results_new/', 'ec')
    #  }}} test on 4 subjects # 

    X, y = get_data('/home/dmalt/Data/aut_gamma/Moscow_baseline_results_new/', 'ec')

    baseclf = make_pipeline(
            ElectrodeSelection(204, metric=dict(mean='logeuclid',
                               distance='riemann')),
            TangentSpace(metric='riemann'),
            LogisticRegression(penalty='l2'))

    cosp_cov = CospCovariances(
            fs=500, window=32, overlap=0.3, fmax=60, fmin=1)

    # # cosp_cov.fit_transform = ShrinkCovMat(cosp_cov.fit_transform)
    # # cosp_cov.transform = shrink_cov_mat(cosp_cov.transform)

    clf = make_pipeline(
            cosp_cov, ShrinkCovMat(), CospBoostingClassifier(baseclf))

    # # # # clf = CospBoostingClassifier(baseclf)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    # # cov = estimation.Covariances('oas').fit_transform(X)
    # # ccov = ShrinkCovMat(cosp_cov.fit_transform)(X)

    # # mdm = classification.MDM()
    # # accuracy = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

    # # print(accuracy.mean())
    acc: List[ndarray] = []
    for train, test in cv.split(XXX, y):
        clf.fit(XXX[list(train)], y[train])
        yte = np.argmax(clf.predict_proba(XXX[list(test)]), 1)
        acc.append(100 * np.mean(yte == y[test]))
