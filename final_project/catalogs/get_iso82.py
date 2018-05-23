# This file will find the mass and radius of the halos that we have WHIM data for according to the iso82 catalog
# Data is stored as:
# x, y, z, ijk, index, mass, # of cells, vx, vy, vz
import numpy as np

# Only need ID, mass, # of cells
cols = (4,5,6)

# Define the dtypes and converters to import the data:
dtypes = 'int,float,float'
# Converters is a dictionary that converts the strings to the correct formatting to read in.
# For column 4 (ID), we need to get rid of () on either side to make an int.
converters = {4: lambda index: index[1:-1]}

# Now read in halo data:
path_to_catalog = '/global/cscratch1/sd/zarija/4096/catalog_z05_iso82.txt'
ID82, mass, size = np.loadtxt(path_to_catalog,converters=converters,dtype=dtypes,usecols=cols,unpack=True)

# Find the halos that have the correct ID
ID138 = np.loadtxt('catalog_iso138_200_180.txt',usecols=0,unpack=True,dtype='int')
ID_args = [int(np.argwhere(ID82==ID)) for ID in ID138]

# Get IDs, mass, and size
ID82_s = ID82[ID_args]
mass_s = mass[ID_args]
size_s = size[ID_args]

# Now make halo_r from halo_size
r = (size_s/np.pi/4.*3.)**(1./3.)

d82 = np.column_stack([ID82_s,mass_s,r])

np.savetxt('catalog_iso82.txt',d82,header='ID,m,r (lattice)')