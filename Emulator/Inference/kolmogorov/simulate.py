import jax
import xarray
import jax.numpy as jnp
import numpy as np

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
from jax_cfd.base import resize
from jax_cfd.spectral import utils as spectral_utils


def get_sim_batch(gridsize,dt,viscosity,cuts,downsample):
    """
    gridsize: gridsize
    dt: numerical timestep
    viscosity: viscosity
    cuts: list of indices to subsample
    downsample: spatial downsampling scale
    """
    
    nsteps=cuts[-1]
    
    #ratio=int(Dt/dt)
    max_velocity = 7 ## For CFL violation
    grid = grids.Grid((gridsize, gridsize), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    ## max_velocity and the second argument here are just stability criterion
    # setup step function using crank-nicolson runge-kutta order 4
    smooth = True # use anti-aliasing 
    
    # **use predefined settings for Kolmogorov flow**
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt)
    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, 1), nsteps)
    
    ## Just want a random seed, so a random key? This is gross
    rand_key=np.random.randint(0,100000000)
    v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(rand_key), grid, max_velocity, 4)
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)
    
    ## Trajectory here is in Fourier space
    _, trajectory = trajectory_fn(vorticity_hat0)
    
    trajectory=trajectory[cuts,:,:]
    traj_real=np.fft.irfftn(trajectory, axes=(1,2))
    
    traj_real=np.empty((traj_real.shape[0],int(traj_real.shape[1]/downsample),int(traj_real.shape[1]/downsample)))
    ## Overwrite grid object
    grid = grids.Grid(((int(gridsize/downsample), int(gridsize/downsample))), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    for aa in range(len(trajectory)):
        coarse_h = resize.downsample_spectral(None, grid, trajectory[aa])
        traj_real[aa]=np.fft.irfftn(coarse_h) ## Using numpy here as jnp won't allow for loops.. but this is gross

    return traj_real


def run_kolmogorov_sim(dt,Dt,nsteps,spinup=0,downsample=None,viscosity=1e-3,gridsize=256):
    """ Run kolmogorov sim with a timestep of dt for nsteps
        returns xarray dataset with *all* snapshots. We perform **spatial** downsampling
        within this function - we will perform **temporal** downsampling outside, in the
        loop that constructs the training dataset.

        dt:         numerical timestep
        Dt:         physical timestep (must be >numerical timestep)
        spinup:     number of numerical timesteps to drop from output
                    dataarray
        viscosity:  viscosity for NS PDE
    return:
        xarray dataset containing snapshots for every timestep
    """

    ratio=int(Dt/dt)
    max_velocity = 7 ## For CFL violation
    grid = grids.Grid((gridsize, gridsize), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    ## max_velocity and the second argument here are just stability criterion
    # setup step function using crank-nicolson runge-kutta order 4
    smooth = True # use anti-aliasing 
    
    # **use predefined settings for Kolmogorov flow**
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt)
    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, 1), nsteps+spinup)
    
    ## Just want a random seed, so a random key? This is gross
    rand_key=np.random.randint(0,100000000)
    v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(rand_key), grid, max_velocity, 4)
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)
    
    ## Trajectory here is in Fourier space
    _, trajectory = trajectory_fn(vorticity_hat0)

    ## Drop spinup
    trajectory=trajectory[spinup:]
    ## Downsample to physical timesteps
    trajectory=trajectory[::ratio]

    traj_real=np.fft.irfftn(trajectory, axes=(1,2))

    ## Downscaling examples
    if downsample is not None:
        traj_real=np.empty((traj_real.shape[0],int(traj_real.shape[1]/downsample),int(traj_real.shape[1]/downsample)))
        ## Overwrite grid object
        grid = grids.Grid(((int(gridsize/downsample), int(gridsize/downsample))), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
        for aa in range(len(trajectory)):
            coarse_h = resize.downsample_spectral(None, grid, trajectory[aa])
            traj_real[aa]=np.fft.irfftn(coarse_h) ## Using numpy here as jnp won't allow for loops.. but this is gross

    spatial_coord = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0] # same for x and y
    coords = {
      'time': Dt * jnp.arange(len(traj_real)),
      'x': spatial_coord,
      'y': spatial_coord,
    }

    return xarray.DataArray(traj_real,dims=["time", "x", "y"], coords=coords)
    