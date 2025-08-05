import numpy as np
import jax_cfd.base.grids as grids
from jax_cfd.spectral import utils as spectral_utils
import copy
import math
import torch


class fourierGrid():
    """ Need an object to store the Fourier grid for a given gridsize, for calculating
        the isotropically averaged 1d spectra """
    def __init__(self,nx,dk=1,dl=1):
        self.nx=nx
        self.dk=dk
        self.dl=dl

        self.nk = int(self.nx/2 + 1)
        
        self.ll = self.dl*np.concatenate((np.arange(0.,nx/2),
                    np.arange(-self.nx/2,0.)))
        self.kk = self.dk*np.arange(0.,self.nk)
        
        ## Get k1d
        self.kmax = np.minimum(np.max(abs(self.ll)), np.max(abs(self.kk)))
        self.dkr = np.sqrt(self.dk**2 + self.dl**2)
        self.k1d=np.arange(0, self.kmax, self.dkr)
        self.k1d_plot=self.k1d+self.dkr/2
        
        ## Get kappas
        self.k, self.l = np.meshgrid(self.kk, self.ll)
        self.kappa2=(self.l**2+self.k**2)
        self.kappa=np.sqrt(self.kappa2)

    def get_ispec(self,field):
        """ Calculate isotropically averaged spectra for a given input 2d field. The input field
            must be in Fourier space """
        ## Array to output isotropically averaged wavenumbers
        phr = np.zeros((self.k1d.size))
    
        ispec=copy.copy(np.array(field))
    
        ## Account for complex conjugate
        ispec[:,0] /= 2
        ispec[:,-1] /= 2
    
        ## Loop over wavenumbers. Average all modes within a given |k| range
        for i in range(self.k1d.size):
            if i == self.k1d.size-1:
                fkr = (self.kappa>=self.k1d[i]) & (self.kappa<=self.k1d[i]+self.dkr)
            else:
                fkr = (self.kappa>=self.k1d[i]) & (self.kappa<self.k1d[i+1])
            phr[i] = ispec[fkr].mean(axis=-1) * (self.k1d[i]+self.dkr/2) * math.pi / (self.dk * self.dl)
    
            phr[i] *= 2 # include full circle
            
        return phr

    def get_ispec_batch(self,field):
        """ Calculate isotropically averaged spectra for a batch of 2d fields. The input fields
            must be in Fourier space, with the first dimension being the batch dimension. """

        ## Array to output isotropically averaged wavenumbers
        phr = np.zeros((len(field),self.k1d.size))
        ispec=copy.copy(np.array(field))

        ## Account for complex conjugate
        ispec[:,:,0] /= 2
        ispec[:,:,-1] /= 2

        ## Loop over wavenumbers. Average all modes within a given |k| range
        for i in range(self.k1d.size):
            if i == self.k1d.size-1:
                fkr = (self.kappa>=self.k1d[i]) & (self.kappa<=self.k1d[i]+self.dkr)
            else:
                fkr = (self.kappa>=self.k1d[i]) & (self.kappa<self.k1d[i+1])
            phr[:,i] = ispec[:,fkr].mean(axis=-1) * (self.k1d[i]+self.dkr/2) * math.pi / (self.dk * self.dl)

            phr[:,i] *= 2 # include full circle

        return phr

        
def get_ke(omega,fourier_grid):
    """ For a voriticity field and fourier grid, calculate isotropically averaged
        KE spectra.
        omega:        2D tensor of vorticity in real space
        fourier_grid: Fourier grid object corresponding to the input vorticity field
    returns:
        k1d_plot: 1d wavenumber bins (centered)
        kespec:   KE spectrum in each wavenumber bin
    """
    omegah=np.fft.rfftn(omega)
    grid = grids.Grid((omega.shape[0], omega.shape[1]), domain=((0, 2 * np.pi), (0, 2 * np.pi)))
    velocity_solve = spectral_utils.vorticity_to_velocity(grid)
    vxhat, vyhat = velocity_solve(omegah)
    KEh=abs(vxhat**2)+abs(vyhat**2)
    kespec=fourier_grid.get_ispec(KEh)
    return fourier_grid.k1d_plot,kespec


def get_ke_batch(omega,fourier_grid):
    """ For a voriticity field and fourier grid, calculate isotropically averaged
        KE spectra.
        omega:        2D tensor of vorticity in real space
        fourier_grid: Fourier grid object corresponding to the input vorticity field
    returns:
        k1d_plot: 1d wavenumber bins (centered)
        kespec:   KE spectrum in each wavenumber bin
    """
    omegah=torch.fft.rfftn(omega,axis=(1,2))
    grid = grids.Grid((omega.shape[-1], omega.shape[-1]), domain=((0, 2 * np.pi), (0, 2 * np.pi)))
    velocity_solve = spectral_utils.vorticity_to_velocity(grid)
    vxhat, vyhat = velocity_solve(omegah.cpu().numpy())
    KEh=abs(vxhat**2)+abs(vyhat**2)
    kespec=fourier_grid.get_ispec_batch(KEh)
    return fourier_grid.k1d_plot,kespec


def spectral_similarity(batch1,batch2):
    """ Compare KE spectra for 2 batches of KE spectra. Assuming the first
        has no NaNs, i.e. this is a reference batch from a simulation 
        Returns the normalised MSE across stable samples, and the number
        of spectra that were nan/inf """

    norm_factors=batch1.mean(axis=0)
    nan_counter=0
    samp_counter=0
    running_ave=0
    for bb in range(len(batch1[1])):
        normed_batch1=batch1[bb]/norm_factors
        normed_batch2=batch2[bb]/norm_factors
        mse=np.sqrt((normed_batch1-normed_batch2)**2).sum()
        if math.isnan(mse) or math.isinf(mse):
            nan_counter+=1
        else:
            running_ave+=mse
            samp_counter+=1
    mse_tot=running_ave/samp_counter
    return mse_tot, nan_counter
