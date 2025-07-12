import jax
import xarray
import jax.numpy as jnp
import numpy as np

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
from jax_cfd.base import resize
from jax_cfd.spectral import utils as spectral_utils

def run_kolmogorov_sim(nsteps, dt, Dt, spinup = 5000, decorr_steps = 1995, viscosity = 1e-4, gridsize = 512, downsample = 8, n_traj = 20, chunk_size = 1000):
    """ 
    Jump_size = nsteps + decorr_steps
    Batch_size_per_sim = n_traj * (traj_T - spinup)/Jump_size
    L = gridsize / downsample
        Output:
            
            Sim_stack (Batch_size_per_sim  x  nsteps  x  L  x  L   tensor): training data

        Inputs:
            nsteps:     number of steps in a rollout
            dt:         numerical timestep
            Dt:         physical timestep (must be >numerical timestep)
            spinup:     number of numerical timesteps to drop from output
            viscosity:  viscosity for NS PDE
            chunk_size: number of steps to process at once (reduce if OOM)
    
    """
    ratio = int(Dt/dt)
    
    ## These cuts split the simulation into short training trajectories
    cuts=[]
    for i in range(n_traj):
        for j in range(nsteps):
            cuts.append(spinup + i * (decorr_steps + nsteps) + j * (ratio))

    traj_T = cuts[-1]
    max_velocity = 7
    grid = grids.Grid((gridsize, gridsize), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    
    # Setup step function using crank-nicolson runge-kutta order 4
    smooth = True
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt)
    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, 1), traj_T)
    
    # Initialize
    rand_key = np.random.randint(0, 100000000)
    v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(rand_key), grid, max_velocity, 4)
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)
    
    if nsteps < 10:
        ## Trajectory here is in Fourier space
        _, trajectory = trajectory_fn(vorticity_hat0)
        
        trajectory=trajectory[cuts,:,:]
        traj_real=np.fft.irfftn(trajectory, axes=(1,2))
        
        traj_real=np.empty((traj_real.shape[0],int(traj_real.shape[1]/downsample),int(traj_real.shape[1]/downsample)))

        if downsample is not None:
            ## Overwrite grid object
            grid = grids.Grid(((int(gridsize/downsample), int(gridsize/downsample))), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
            for aa in range(len(trajectory)):
                coarse_h = resize.downsample_spectral(None, grid, trajectory[aa])
                traj_real[aa]=np.fft.irfftn(coarse_h)
        
        spatial_coord = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0]
        coords = {
            'time': Dt * jnp.arange(len(trajectory_real)),
            'x': spatial_coord,
            'y': spatial_coord,
        }
        
        return xarray.DataArray(traj_real, dims=["time", "x", "y"], coords=coords)
        
    else:
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
                ## Overwrite grid object
                grid = grids.Grid(((int(gridsize/downsample), int(gridsize/downsample))), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
                for aa in range(len(trajectory)):
                    coarse_h = resize.downsample_spectral(None, grid, chunk_trajectory_subsampled[aa])
                    chunk_traj_real[aa]=np.fft.irfftn(coarse_h) ## Using numpy here as jnp won't allow for loops.. but this is gross
                
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
        spatial_coord = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0]
        coords = {
            'time': Dt * jnp.arange(len(full_trajectory)),
            'x': spatial_coord,
            'y': spatial_coord,
        }
        
        return xarray.DataArray(full_trajectory, dims=["time", "x", "y"], coords=coords)
