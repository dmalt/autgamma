import numpy as np
import h5py
import pandas as pd

table_name = 'Aut_gamma_database.xlsx'

t_R = pd.read_excel(table_name, 1)
t_K = pd.read_excel(table_name, 2)

age_thresh = 148
ages_R = t_R['MEG age in month']
ages_K = t_K['MEG age in month']

age_names_R = np.array(t_R[ages_R > age_thresh]['Code'])
age_names_K = np.array(t_K[ages_K > age_thresh]['Code'])

h5_fname = 'X.hdf5'
h5_data = h5py.File(h5_fname, 'r')
s_names = np.array(list(h5_data.attrs.keys()))

age_names = np.concatenate([age_names_K, age_names_R])
s_names = np.array([s for s in s_names if s in age_names])
np.save('s_names.npy', s_names)
