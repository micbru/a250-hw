import numpy as np
import h5py
import plot
# Get previous data:
path = './4096z05'
mWHIM,mCond,mDif,mHalo = np.loadtxt(path+'/mass_fraction.txt',unpack=True)
# Don't need these since I put them in the other catalog
#halo_ID, halo_mass, halo_r = np.loadtxt(path+'/WHIM_data.txt',unpack=True,usecols=(0,1,2)) # Get these seperate from WHIM_data
WHIM_data = np.loadtxt(path+'/WHIM_data.txt',usecols=range(3,29)) # Unpack the 26 directions of WHIM_data

# Get m200 and m180 data also.
# Read in:
IDs, m, r, m200, r200, m180, r180 = np.loadtxt('./catalogs/catalog_iso138_200_180.txt',usecols=(0,4,5,6,7,8,9),unpack=True)

# Make a list of halo_IDs we need:
ID_list = np.loadtxt(path+'/WHIM_data.txt',usecols=0)
args = [np.where(IDs==ID)[0][0] for ID in ID_list]

# Take subset of arrays
m = m[args]
r = r[args]
m200 = m200[args]
r200 = r200[args]
m180 = m180[args]
r180 = r180[args]

# Convert units:
# Get size and redshift:
path_to_data = path+"/universe_data.txt"
u_data = np.loadtxt(path_to_data)
# Get the length of the array (shape) and the physical size in Mpc/h (size):
size = u_data[1]
l = u_data[0]
convert_pt_Mpc = size/l

r *= convert_pt_Mpc
r200 *= convert_pt_Mpc
r180 *= convert_pt_Mpc
if False:
    # Get redshift:
    z = u_data[2]

    # Plot mass fraction
    plot.box_mass_fraction_plot(mWHIM,mCond,mDif,mHalo,size,z,path)

# Plot WHIMsize
ID = 'iso138' # This is the catalog we used.
plot.WHIMsize_HaloMass_plot(m,r,WHIM_data,ID,path)

ID = 'iso138_m200'
plot.WHIMsize_HaloMass_plot(m200,r200,WHIM_data,ID,path)

ID = 'iso138_m180'
plot.WHIMsize_HaloMass_plot(m180,r180,WHIM_data,ID,path)