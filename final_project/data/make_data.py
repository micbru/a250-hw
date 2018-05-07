# This file will be run on cori to make a smaller test set we can use locally. It will take from the 2048 dataset.
import h5py
import numpy as np

# Read in full data 
path_to_data = "/global/cscratch1/sd/zarija/Bispectrum/z3_2048.h5"
f = h5py.File(path_to_data)
fdata = f['native_fields']
rho_bar = fdata['baryon_density']
temp = fdata['temperature']

# Get halo location from previously
x = 632
y = 1270
z = 1811

# Radius of small box
r = 32

rho_sub = rho_bar[x-r:x+r+1,y-r:y+r+1,z-r:z+r+1]
t_sub = temp[x-r:x+r+1,y-r:y+r+1,z-r:z+r+1]

np.save('rho_test.npy',rho_sub)
np.save('t_test.npy',t_sub)
