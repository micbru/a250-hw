# Import
import numpy as np
import numpy.ma as ma
import h5py
import dask.array as da

# Set universal WHIM parameters to divide cosmic phase diagram into 4:
rhoMax = 100.
tMin = 10.**5

# Functions #####################################
def main(path_to_data,path_to_catalog,output_mass_frac,output_WHIM_data):
    '''
    This function will take the path to the data and the path to a halo catalog and output 
    a text file that has the mass fractions for all phases in smaller boxes, as well as a text
    file that contains the WHIM sizes for halos.
    '''
    # Read in the data.
    rho_bar, temp, l, size, z = read_data(path_to_data)
    # This mass cutoff determines how low we will analyze in halo mass.
    mass_cutoff = 10**12 
    halo_data = read_catalog(path_to_catalog,mass_cutoff)
    # Analyze the data.
    mass_fraction, WHIM_data, WHIM_troubleshoot = analyze(rho_bar,temp,halo_data,l,size)
    
    # Output to text.
#    np.savetxt(output_mass_frac,mass_fraction,header='mWHIM,mCond,mDif,mHalo)
    np.savetxt(output_WHIM_data,WHIM_data,header='ID, mass, radius (Mpc/h), WHIM sizes (Mpc/h)')
    np.savetxt(output_WHIM_data+'.troubleshoot.txt',WHIM_troubleshoot,header='ID, n halo bdry, n WHIM bdry, n no WHIM')
    

def read_data(path_to_data):
    '''
    Read in the data from hdf5 file and return 
    rho baryon array reference, temperature reference, size in lattice points, physical size, and redshift.
    '''
    f = h5py.File(path_to_data)
    fdata = f['native_fields']
    fsize = f['domain']
    f_universe = f['universe']

    # Get the length of the array (shape) and the physical size in Mpc/h (size):
    size = fsize.attrs['size'][0] # Need the zero since these are in 3D.
    shape = fsize.attrs['shape'][0]
    l = int(shape)

    # Get physical data:
    rho_bar = fdata['baryon_density']
    temp = fdata['temperature']

    # Get redshift:
    z = f_universe.attrs['redshift']

    return rho_bar, temp, l, size, z


def read_catalog(path_to_catalog,mass_cutoff):
    '''
    Read in halo catalog and eliminate entries below the mass cutoff.
    Data is stored as:
    x, y, z, ijk, index, mass, # of cells, vx, vy, vz

    Return halo_data in the form:
    i, j, k, ID, mass, r
    '''	
    # We want to read in columns 3,4,5,6 for the data we need,
    # Have to read in column 3 separately to get this to work properly
    cols = (4,5,6)

    # Define the dtypes and converters to import the data:
    dtypes = 'int,float,float'
    # Converters is a dictionary that converts the strings to the correct formatting to read in.
    # For column 3 (i j k), we need to get rid of x's to read in as array. 
    # For column 4 (ID), we need to get rid of () on either side to make an int.
    converters = {3: lambda ijk: ijk.split(b'x'), 4: lambda index: index[1:-1]}

    # Now read in halo data:
    x, y, z  = np.loadtxt(path_to_catalog,converters=converters,dtype='int',usecols=3,unpack=True)
    ID, mass, size = np.loadtxt(path_to_catalog,converters=converters,dtype=dtypes,usecols=cols,unpack=True)

    # Sort the array so that we can eliminate the first part below the mass cutoff.
    # This makes an array of sorted indices then sorts the halo_mass array in place.
    sort_inds = mass.argsort()
    mass.sort()
    # Sort the other arrays.
    x = x[sort_inds]
    y = y[sort_inds]
    z = z[sort_inds]
    ID = ID[sort_inds]
    size = size[sort_inds]

    # Now make halo_r from halo_size
    r = (size/np.pi/4.*3.)**(1./3.)

    # Now return only those below the mass cutoff
    i_cutoff = np.searchsorted(mass,mass_cutoff)
    print("Total number of halos above logM = {}: {}".format(np.log10(mass_cutoff),len(mass)-i_cutoff))
    return np.column_stack([x[i_cutoff::], y[i_cutoff::], z[i_cutoff::], ID[i_cutoff::], mass[i_cutoff::], r[i_cutoff::]])


def analyze(rho_bar,temp,halo_data,l,size):
    '''
    Analyze the data we read in by dividing into readable boxes. This means find the mass_fraction in 64 boxes, and finding the WHIM_size for all halos above mass cutoff.
    Takes in full rho_bar, temp data, full halo catalog, and size in physical and lattice units.
    Returns mass fraction in 64 boxes and WHIM size for all halos in catalog.
    WHIM_data stored as halo_ID, halo_mass, halo_size, WHIM size in 26 directions.
    '''
    # Define the box sizes we will be using.
    bl = int(l/4)   # Box length. The magic number 4 is because we want 64 boxes.
    nb = int(l/bl)  # Number of boxes along one dimension
#    large_halo = max(halo_data[:,5]) # Find the largest halo
#    buf = int(6*large_halo) # Use a buffer of magic number*the largest halo radius

    # Initialize the arrays that will store the mean for each box.
    mWHIM = 0#np.zeros(nb**3)
    mCond = 0#np.zeros(nb**3)
    mDif = 0#np.zeros(nb**3)
    mHalo = 0#np.zeros(nb**3)

    # Initialize the array for the WHIM sizes
    # 29 is a magic number: we have 26 directions, plus we want halo ID, halo mass, and halo radius
    WHIM_data = np.zeros([len(halo_data),29])
    WHIM_troubleshoot = np.zeros([len(halo_data),4])

    # Now loop for each box
    # This is density per pixel * size of 1 pixel / (avg density in box * box volume)
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                # Make sub arrays that are readable into memory:
                # Read in with a buffer zone so that we can find all the halos:
                ii = i*bl
                jj = j*bl
                kk = k*bl
                rho_sub = rho_bar[ii:ii+bl+1,jj:jj+bl+1,kk:kk+bl+1]
                t_sub = temp[ii:ii+bl+1,jj:jj+bl+1,kk:kk+bl+1]
                # Use mass_fraction function, only reading in without buffer
#                mWHIM[i+nb*j+nb*nb*k], mCond[i+nb*j+nb*nb*k], mDif[i+nb*j+nb*nb*k], mHalo[i+nb*j+nb*nb*k] = mass_fraction(rho_sub[buf:-buf,buf:-buf,buf:-buf],t_sub[buf:-buf,buf:-buf,buf:-buf],bl)

                # Now find all halos in this subarray. Loop through entire array.
                halo_list = []
                for q in range(len(halo_data)):
                    # Integer divide by bl to see if we fall in this box, excluding the buffer. If so then find WHIM size:
                    if ((int(halo_data[q,0]//bl)==i) & (int(halo_data[q,1]//bl)==j) & (int(halo_data[q,2]//bl)==k)):
                        halo_list.append(q)
                for q in halo_list:
                    WHIM_data[q,0:3] = halo_data[q,3::]
                    # Correct halo locations in the sub box.
                    WHIM_data[q,3::], n_halo_bdry, n_WHIM_bdry, n_noWHIM = WHIM_size(rho_sub,t_sub,l,size,halo_data[q,0]-ii,halo_data[q,1]-jj,halo_data[q,2]-kk)
                    # Set WHIM_data halo_r to physical units:
                    WHIM_data[q,2] *= size/l
                    # Save troubleshooting information:
                    WHIM_troubleshoot[q] = np.column_stack([halo_data[q,3],n_halo_bdry,n_WHIM_bdry,n_noWHIM])

                # Now delete these arrays from memory:
                del rho_sub
                del t_sub
                print("Box Number: {}/64, Number of halos in that box: {}".format(i*nb*nb+nb*j+k+1, len(halo_list)))

    return np.column_stack([mWHIM,mCond,mDif,mHalo]), WHIM_data, WHIM_troubleshoot
    

def mass_fraction(rho,temp,bl):
    '''
    Get the max fraction for a small box of size bl:
    Inputs:
        rho:   Baryon overdensity at each point in an n**3 array (mean is 1).
        temp:  Temperature at each point in an n**3 array.
        bl:    The size of the small box in lattice points (assuming a cube).
    Outputs:
        returns mWHIM, mCond, mDif, mHalo.
    '''
    # Calculate necessary means:
    brho_avg = rho.mean()
    bmass = brho_avg*bl**3 # bl**3 is volume of box.
   
    # Set up the masks for all four regions:
    rhoMask = ma.masked_greater_equal(rho,rhoMax)
    tMask = ma.masked_less_equal(temp,tMin)

    WHIM_Mask = rhoMask.mask + tMask.mask
    cond_Mask = ~rhoMask.mask + ~tMask.mask 
    dif_Mask = rhoMask.mask + ~tMask.mask
    halo_Mask = ~rhoMask.mask + tMask.mask

    # Now make the density arrays corresponding to the masks:
    rhoWHIM = ma.array(rho,mask=WHIM_Mask)
    rhoCond = ma.array(rho,mask=cond_Mask)
    rhoDif = ma.array(rho,mask=dif_Mask)
    rhoHalo = ma.array(rho,mask=halo_Mask)

    # Now calculate the amount of mass in each fraction.
    mWHIM = ma.sum(rhoWHIM)/bmass
    mCond = ma.sum(rhoCond)/bmass
    mDif = ma.sum(rhoDif)/bmass
    mHalo = ma.sum(rhoHalo)/bmass

    return mWHIM, mCond, mDif, mHalo


def WHIM_size(rho, t, l, size, halo_i, halo_j, halo_k):
    '''
    This function will take in simulation data and a halo catalog and find the radial extent of the WHIM in 13 directions away (both ways, total 26) 
    from the center of the halo. 
    It does this by taking skewers through the halo and finding the size of the WHIM.
    Inputs:
        rho:        Baryon overdensity at each point in an array (mean is 1).
        t:bbb       Temperature at each point in an array.
        l:          The size of the full box in lattice points (assuming a cube).
        size:       Physical size of the full box in Mpc/h.
        halo_i,j,k: Halo location in the sub array.
    Returns:
        WHIMsize
    '''
    # Make array of halo_loc:
    halo_loc = np.array([halo_i,halo_j,halo_k],dtype=int)
    # Set up physical unit conversions:
    # Multiply by this to go from lattice points to Mpc/h
    convert_pt_Mpc = size/l

    # Get length of sub array to check we are still in halo
    sub_l = len(rho)

    # Now make a numpy array of all possible directions:
    indices = np.array([[0,0,1],[0,1,0],[1,0,0],[1,1,0],[1,0,1],[1,-1,0],
                        [1,0,-1],[0,1,-1],[0,1,1],[1,-1,-1],[1,-1,1],[-1,1,1],[1,1,1]])
    n_ind = len(indices) # Should be 13

    # Initialize an array to store sizes in each direction:
    # Note that it is 2 times since we will store the WHIM forward and backward along each direction
    WHIMsize = np.zeros(2*n_ind)

    # Also initialize counters so we know why we are dropping information:
    n_halo_bdry = 0
    n_WHIM_bdry = 0
    n_noWHIM = 0

    # Now step through each possible case:
    for i in range(n_ind):
        # First set fwd_i and bck_i to be the starting indices.
        # Note that we can't add one here in case the halo is at the edge
        # These have to be arrays so we can add to them.
        fwd_i = np.copy(halo_loc)
        bck_i = np.copy(halo_loc)
        # Check to see if we are still in halo. If so, advance the indices. Let's only check density and do temperature checks once out of the halo.
        # Because indices has positive and negative components, we need to check both and make sure we are in the box.
        # If we go past the edge of the halo, do not enter the WHIM loops. Set boolean variables for this:
        WHIM_fwd = True
        WHIM_bck = True
        while (rho[tuple(fwd_i)] > rhoMax):
            fwd_i += indices[i]
            if (np.any(fwd_i >= sub_l) | np.any(fwd_i <= 0)):
                #print("The halo extends past the sub box range!")
                n_halo_bdry += 1
                WHIM_fwd = False
                fwd_i = halo_loc # Set this so that we get NaN for WHIMsize.
                break
        while (rho[tuple(bck_i)] > rhoMax):
            bck_i -= indices[i]
            if (np.any(bck_i >= sub_l) | np.any(bck_i <= 0)):
                #print("The halo extends past the sub box range!")
                n_halo_bdry += 1
                WHIM_bck = False
                bck_i = halo_loc # Set this so that we get NaN for WHIMsize.
                break
        ### FORWARD
        # Now we need to check temperature, if we didn't already hit a boundary
        if WHIM_fwd:
            if (t[tuple(fwd_i)] < tMin):
                # It's possible we are still in the halo, but much more likely there just is no WHIM.
                fwd_i = halo_loc # Set this so that we get NaN for WHIMsize.
                n_noWHIM += 1
            else:
                # We should now be in the WHIM
                while ((rho[tuple(fwd_i)] < rhoMax) & (t[tuple(fwd_i)] > tMin) ):
                    fwd_i += indices[i]
                    if (np.any(fwd_i >= sub_l) | np.any(fwd_i <= 0)):
                        #print("The WHIM extends past the sub box range!")
                        n_WHIM_bdry += 1
                        fwd_i = halo_loc # Set this so that we get NaN for WHIMsize.
                        break
        ### BACKWARD
        # Now we need to check temperature, if we didn't already hit a boundary
        if WHIM_bck:
            if (t[tuple(bck_i)] < tMin):
                # It's possible we are still in the halo, but much more likely there just is no WHIM.
                bck_i = halo_loc # Set this so that we get NaN for WHIMsize.
                n_noWHIM += 1
            else:
                # We should now be in the WHIM
                while ((rho[tuple(bck_i)] < rhoMax) & (t[tuple(bck_i)] > tMin) ):
                    bck_i -= indices[i]
                    if (np.any(bck_i >= sub_l) | np.any(bck_i <= 0)):
                        #print("The WHIM extends past the sub box range!")
                        n_WHIM_bdry += 1
                        bck_i = halo_loc # Set this so that we get NaN for WHIMsize.
                        break

        # Set the WHIM size since we have now found it:
        WHIMsize[i] = np.linalg.norm(fwd_i - halo_loc)
        WHIMsize[n_ind+i] = np.linalg.norm(bck_i - halo_loc)

        # Put a warning up if we don't find any WHIM and set to NaN.
        if (WHIMsize[i] == 0.):
            #print("There is no WHIM in the forward ", i,"th direction, or the starting index wasn't correct")
            WHIMsize[i] = np.nan
        if (WHIMsize[n_ind+i] == 0.):
            #print("There is no WHIM in the backward ", i,"th direction, or the starting index wasn't correct")
            WHIMsize[n_ind+i] = np.nan
    
    # Convert to physical units:
    WHIMsize *= convert_pt_Mpc

    return WHIMsize, n_halo_bdry, n_WHIM_bdry, n_noWHIM
