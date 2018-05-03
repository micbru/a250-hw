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

# We need to cut off by halo ID to make sure we have the same size array as WHIM size!
# Since that array is sorted we know it is the first halo
min_halo_ID_arg = np.where(m==min(m[m>10**12]))[0]
min_halo_ID = IDs[min_halo_ID_arg]
sort_i = m.argsort()
m.sort()
r = r[sort_i]
IDs = IDs[sort_i]
# Now sort m200 and m180
m200 = m200[sort_i]
r200 = r200[sort_i]
m180 = m180[sort_i]
r180 = r180[sort_i]

# Convert units:
# Get size and redshift:
path_to_data = "/global/cscratch1/sd/zarija/4096/z05.h5"
f = h5py.File(path_to_data)
fsize = f['domain']
# Get the length of the array (shape) and the physical size in Mpc/h (size):
size = fsize.attrs['size'][0] # Need the zero since these are in 3D.
l = fsize.attrs['shape'][0]
convert_pt_Mpc = size/l

r *= convert_pt_Mpc
r200 *= convert_pt_Mpc
r180 *= convert_pt_Mpc
i_cutoff = int(np.where(IDs==min_halo_ID)[0])
print(i_cutoff)
if False:
    # Get size and redshift:
    path_to_data = "/global/cscratch1/sd/zarija/4096/z05.h5"
    f = h5py.File(path_to_data)
    fsize = f['domain']
    f_universe = f['universe']
    # Get the length of the array (shape) and the physical size in Mpc/h (size):
    size = fsize.attrs['size'][0] # Need the zero since these are in 3D.
    # Get redshift:
    z = f_universe.attrs['redshift']

    # Plot mass fraction
    plot.box_mass_fraction_plot(mWHIM,mCond,mDif,mHalo,size,z,path)

# Plot WHIMsize
ID = 'iso138' # This is the catalog we used.
plot.WHIMsize_HaloMass_plot(m[i_cutoff::],r[i_cutoff::],WHIM_data,ID,path)

ID = 'iso138_m200'
plot.WHIMsize_HaloMass_plot(m200[i_cutoff::],r200[i_cutoff::],WHIM_data,ID,path)

ID = 'iso138_m180'
plot.WHIMsize_HaloMass_plot(m180[i_cutoff::],r180[i_cutoff::],WHIM_data,ID,path)

# Check halos are the same:
#print("Halo ID problems for m200: {}, for m180: {}".format(np.count_nonzero(ID200[i_cutoff::]-halo_ID),np.count_nonzero(ID180[i_cutoff::]-halo_ID)))
#print("Halo ID: {}, 200: {}, 180: {}".format(halo_ID[0:5],ID200[i_cutoff:(i_cutoff+5)],ID180[i_cutoff:(i_cutoff+5)])) 
