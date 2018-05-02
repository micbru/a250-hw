'''
This file contains the following functions for plotting:
box_mass_fraction_plot(mWHIM,mCond,mDif,mHalo,size,z,fname=".")
    # Tiled plot with best fits on histograms
WHIMsize_HaloMass_plot(halo_mass,halo_r,WHIMsize,ID,fname=".")
    # This will take plot the radial extent of the WHIM compared to the Halo Mass to see if there is an interesting relationship.
    # It turns out the mean isn't so good at giving this information, so let's make a boxplot. 
'''
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# Change plt backend to be able to save figures.
plt.switch_backend('agg')

# # Functions

# Tiled plot with best fits on histograms:
def box_mass_fraction_plot(mWHIM,mCond,mDif,mHalo,size,z,fname="."):
    '''
    Inputs:
        mWHIM:  Mass fraction of WHIM for each box.
        mCond:  Mass fraction of condensed matter for each box.
        mDif:   Mass fraction of diffuse matter for each box.
        mHalo:  Mass fraction of hot halos for each box.
        size:   Size of entire box. For labelling.
        z:      Redshift of entire box. For labelling.
        fname:  Can output to different folder.
    Outputs:
        {}boxes_{}Mpc_z{}.png: 
            Image file with naming convention of # of boxes, size of simulation, and redshift of simulation.
    '''
    # find number of boxes:
    nboxes = len(mWHIM)

    # Make the histograms to show the amount by mass in each quadrant:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.hist(mWHIM,normed=1,alpha=0.5,edgecolor='k',label='WHIM',color='g')
    ax.hist(mCond,normed=1,alpha=0.5,edgecolor='k',label='Condensed',color='b')
    ax.hist(mDif,normed=1,alpha=0.5,edgecolor='k',label='Diffuse',color='y')
    ax.hist(mHalo,normed=1,alpha=0.5,edgecolor='k',label='Hot Halo',color='r')

    # Plot best fit lines:
    xp = np.linspace(0,1,100)
    ax.plot(xp,mlab.normpdf(xp,np.mean(mWHIM),np.std(mWHIM)),c='g')
    ax.plot(xp,mlab.normpdf(xp,np.mean(mCond),np.std(mCond)),c='b')
    ax.plot(xp,mlab.normpdf(xp,np.mean(mDif),np.std(mDif)),c='y')
    ax.plot(xp,mlab.normpdf(xp,np.mean(mHalo),np.std(mHalo)),c='r')

    # Set up labelling
    ax.legend()
    ax.set_xlabel('Mass Fraction')
    ax.set_ylabel('PDF')
    fig.suptitle('{:.0f} Mpc/h at z = {:.1f}'.format(size, z))
    
    # We don't want one peak to make all others invisible, so set ylim max at 15 (magic number)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0,min(15,ymax))
    fig.savefig('{}/{:.0f}boxes_{:.0f}Mpc_z{:.1f}.png'.format(fname,nboxes,size,z))

# ### WHIM Size vs. Halo Mass
# This will take plot the radial extent of the WHIM compared to the Halo Mass to see if there is an interesting relationship.
# It turns out the mean isn't so good at giving this information, so let's make a boxplot. 
# For boxplot: The box extends from the lower to upper quartile values of the data, with a line at the median.
# Default: Upper whisker = Q3 + whis (1.5 by default) * (Q3-Q1)
# Default: Lower whisker = Q1 - whis * (Q3-Q1)
# Whiskers are now [5, 95]!
# Extra points are plotted individually.
def WHIMsize_HaloMass_plot(halo_mass,halo_r,WHIMsize,ID,fname="."):
    '''
    Inputs:
        halo_mass:  Vector of halo masses in units of M_sun.
        WHIMsize:   Size of WHIM around that halo in units of Mpc/h, also as vector. This will be 26 points for each halo_mass.
        halo_r:     To express size of WHIM in units of virial halo radius.
        ID:         Some form of ID, maybe the catalog ID, to make sure we don't overwrite plots.
        fname:      Can output to different folder.
    Outputs:
        WHIMsize_HaloMass_ID.png, WHIMsize_virial_HaloMass_ID.png, halo_mass_hist_ID.png
    '''
    # Take log of halo masses:
    log_hm = np.log10(halo_mass)

    # Plot histogram of halo masses
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(1,1,1)
    ax0.hist(log_hm)
    ax0.set_xlabel('Halo Mass (log(M/M_sun))')
    ax0.set_ylabel('Number of halos in catalog')
    fig0.savefig('{}/halo_mass_hist_ID{}.png'.format(fname,ID))

    # If we have more than 10 halos, put into n bins.
    nbins = 10
    # Whiskers:
    whis = [5, 95]
    # Outliers:
    out = ''
    if len(log_hm) > nbins:
        # Sort log_hm, WHIMsize, and WHIMsize_virial
        sort_ind = np.argsort(log_hm)
        log_hm.sort()
        # WHIMsize_virial must be defined here so we can sort properly:
        WHIMsize_virial = (WHIMsize.transpose()/halo_r).transpose()
        # Now sort
        WHIMsize = WHIMsize[sort_ind]
        WHIMsize_virial = WHIMsize_virial[sort_ind]

        # Divide log_hm into nbins+1
        log_hm_nbins = np.linspace(np.min(log_hm),np.max(log_hm),num=nbins+1)
        # Get the indices corresponding to each bin
        bin_ind = np.searchsorted(log_hm,log_hm_nbins)
        # Make datasets to plot:
        WHIMsize_nbins = []
        WHIMsize_virial_nbins = []
        for i in range(nbins):
            WHIMsize_nbins.append(WHIMsize[bin_ind[i]:bin_ind[i+1]].flatten())
            WHIMsize_virial_nbins.append(WHIMsize_virial[bin_ind[i]:bin_ind[i+1]].flatten())
            # Remove nan
            WHIMsize_nbins[i] = WHIMsize_nbins[i][~np.isnan(WHIMsize_nbins[i])]
            WHIMsize_virial_nbins[i] = WHIMsize_virial_nbins[i][~np.isnan(WHIMsize_virial_nbins[i])]

        # Labels/positions/widths
        pos = [log_hm[bin_ind[i]:bin_ind[i+1]].sum()/(bin_ind[i+1]-bin_ind[i]) for i in range(nbins)]
        width = [log_hm_nbins[i+1]-log_hm_nbins[i] for i in range(nbins)]

        # Now plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.boxplot(WHIMsize_nbins,positions=pos,whis=whis,sym=out,widths=width,manage_xticks=False)
        ax.set_xlabel('Halo Mass bins (log(M/M_sun))')
        ax.set_ylabel('Radial Size of "WHIM" (Mpc/h)')
        fig.savefig('{}/WHIMsize_HaloMass_ID{}.png'.format(fname,ID))

        # Now also make a plot dividing by the virial radius:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.boxplot(WHIMsize_virial_nbins,positions=pos,whis=whis,sym=out,widths=width,manage_xticks=False)
        ax1.set_xlabel('Halo Mass bins (log(M/M_sun))')
        ax1.set_ylabel('Radial Size of "WHIM" (Virial Radius of Halo)')
        fig1.savefig('{}/WHIMsize_virial_HaloMass_ID{}.png'.format(fname,ID))

    else: # We have fewer than 10 halos
        # Now plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.boxplot(WHIMsize.transpose(),labels=log_hm,whis=whis,sym=out)
        ax.set_xlabel('Halo Mass (log(M/M_sun))')
        ax.set_ylabel('Radial Size of "WHIM" (Mpc/h)')
        fig.savefig('{}/WHIMsize_HaloMass_ID{}.png'.format(fname,ID))

        # Now also make a plot dividing by the virial radius:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.boxplot((WHIMsize.transpose()/halo_r),labels=log_hm,whis=whis,sym=out)
        ax1.set_xlabel('Halo Mass (log(M/M_sun))')
        ax1.set_ylabel('Radial Size of "WHIM" (Virial Radius of Halo)')
        fig1.savefig('{}/WHIMsize_virial_HaloMass_ID{}.png'.format(fname,ID))

# Test ##########################################
if __name__ == '__main__':
    # Get previous data:
    path = './2048z3'
    mWHIM,mCond,mDif,mHalo = np.loadtxt(path+'/mass_fraction.txt',unpack=True)
    halo_ID, halo_mass, halo_r = np.loadtxt(path+'/WHIM_data.txt',unpack=True,usecols=(0,1,2)) # Get these seperate from WHIM_data
    WHIM_data = np.loadtxt(path+'/WHIM_data.txt',usecols=range(3,29)) # Unpack the 26 directions of WHIM_data
    
    # Get size and redshift:
    path_to_data = "/global/cscratch1/sd/zarija/Bispectrum/z3_2048.h5"
    f = h5py.File(path_to_data)
    fsize = f['domain']
    f_universe = f['universe']
    # Get the length of the array (shape) and the physical size in Mpc/h (size):
    size = fsize.attrs['size'][0] # Need the zero since these are in 3D.
    # Get redshift:
    z = f_universe.attrs['redshift']

    # Plot mass fraction
    box_mass_fraction_plot(mWHIM,mCond,mDif,mHalo,size,z,path)

    # Plot WHIMsize
    ID = 1
    WHIMsize_HaloMass_plot(halo_mass,halo_r,WHIM_data,ID,path)
