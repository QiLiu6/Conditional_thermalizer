import jax
import xarray
import jax.numpy as jnp
import numpy as np

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
from jax_cfd.base import resize
from jax_cfd.spectral import utils as spectral_utils

def run_kolmogorov_sim_chunked(dt, Dt, nsteps, spinup=0, downsample=None, viscosity=1e-3, gridsize=256, chunk_size=1000):
    """ Run kolmogorov sim with chunked processing to avoid memory issues
        
        dt:         numerical timestep
        Dt:         physical timestep (must be >numerical timestep)
        nsteps:     total number of steps to generate (after spinup)
        spinup:     number of numerical timesteps to drop from output
        viscosity:  viscosity for NS PDE
        gridsize:   simulation grid size
        downsample: downsampling factor
        chunk_size: number of steps to process at once (reduce if OOM)
    
    return:
        xarray dataset containing snapshots for every timestep
    """
    
    ratio = int(Dt/dt)
    max_velocity = 7
    grid = grids.Grid((gridsize, gridsize), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    
    # Setup step function using crank-nicolson runge-kutta order 4
    smooth = True
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt)
    
    # Initialize
    rand_key = np.random.randint(0, 100000000)
    v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(rand_key), grid, max_velocity, 4)
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)
    
    # First, run spinup phase and discard
    if spinup > 0:
        print(f"Running spinup phase: {spinup} steps...")
        spinup_trajectory_fn = cfd.funcutils.trajectory(
            cfd.funcutils.repeated(step_fn, 1), spinup)
        vorticity_hat_current, _ = spinup_trajectory_fn(vorticity_hat0)
        # Clear memory
        del spinup_trajectory_fn
    else:
        vorticity_hat_current = vorticity_hat0
    
    # Now process the main simulation in chunks
    total_chunks = (nsteps + chunk_size - 1) // chunk_size
    all_trajectories = []
    
    print(f"Processing main simulation in {total_chunks} chunks of {chunk_size} steps...")
    
    for chunk_idx in range(total_chunks):
        start_step = chunk_idx * chunk_size
        end_step = min((chunk_idx + 1) * chunk_size, nsteps)
        actual_chunk_size = end_step - start_step
        
        print(f"Processing chunk {chunk_idx + 1}/{total_chunks}: steps {start_step}-{end_step} ({actual_chunk_size} steps)")
        
        # Create trajectory function for this chunk
        chunk_trajectory_fn = cfd.funcutils.trajectory(
            cfd.funcutils.repeated(step_fn, 1), actual_chunk_size)
        
        # Run this chunk
        vorticity_hat_current, chunk_trajectory = chunk_trajectory_fn(vorticity_hat_current)
        
        # Subsample to physical timesteps
        chunk_trajectory_subsampled = chunk_trajectory[::ratio]
        
        # Convert to real space
        chunk_traj_real = np.fft.irfftn(chunk_trajectory_subsampled, axes=(1, 2))
        
        # Downsample if needed
        if downsample is not None:
            downsampled_grid = grids.Grid(
                (int(gridsize/downsample), int(gridsize/downsample)), 
                domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi))
            )
            chunk_traj_downsampled = np.empty((
                chunk_traj_real.shape[0],
                int(gridsize/downsample),
                int(gridsize/downsample)
            ))
            
            for aa in range(len(chunk_trajectory_subsampled)):
                coarse_h = resize.downsample_spectral(None, downsampled_grid, chunk_trajectory_subsampled[aa])
                chunk_traj_downsampled[aa] = np.fft.irfftn(coarse_h)
            
            chunk_traj_real = chunk_traj_downsampled
            final_grid = downsampled_grid
        else:
            final_grid = grid
        
        # Store this chunk
        all_trajectories.append(chunk_traj_real)
        
        # Clean up memory
        del chunk_trajectory_fn, chunk_trajectory, chunk_trajectory_subsampled, chunk_traj_real
        if downsample is not None:
            del chunk_traj_downsampled
        
        print(f"Chunk {chunk_idx + 1} completed")
    
    # Concatenate all chunks
    print("Concatenating all chunks...")
    full_trajectory = np.concatenate(all_trajectories, axis=0)
    
    # Create coordinates
    spatial_coord = jnp.arange(final_grid.shape[0]) * 2 * jnp.pi / final_grid.shape[0]
    coords = {
        'time': Dt * jnp.arange(len(full_trajectory)),
        'x': spatial_coord,
        'y': spatial_coord,
    }
    
    return xarray.DataArray(full_trajectory, dims=["time", "x", "y"], coords=coords)


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
    
