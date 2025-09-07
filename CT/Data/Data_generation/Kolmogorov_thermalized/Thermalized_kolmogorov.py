'''
Python3
Author: Qi Liu
This script is for generating data for the flow matching model's training. See detailed in Qi Liu's notion workspace: flow matching implementation for emulation
Modified to process data in batches to avoid GPU memory issues.
'''
import torch
import CT.Inference.Kolmogorov.performance as performance
import thermalizer.models.misc as Emulator_misc
import CT.Models.misc as CT_misc

print("Loading data and emulator...")

# Loading the data from numerical runs
data_dict = torch.load("/scratch/ql2221/thermalizer_data/kolmogorov/reynold10k/combined_data.p")

# Loading the emulator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_string = '/scratch/ql2221/thermalizer_data/wandb_data/wandb/run-20250526_223850-r12kgbg1/files/checkpoint_best.p'
emulator = Emulator_misc.load_model(file_string).to(device)
checkpoint_string = "/scratch/ql2221/thermalizer_data/wandb_data/wandb/run-20250610_205843-bigdey10/files/checkpoint_last.p"
CT = CT_misc.load_diffusion_model(checkpoint_string).to(device)

# Normalize
data = data_dict['data'] / 4.44

# Get the initial conditions
ics = data[0:3455, 0, :, :].unsqueeze(1)  # Shape: [34560, 1, 64, 64]
print(f"Initial conditions shape: {ics.shape}")

# Process in much smaller batches to avoid GPU memory issues
batch_size = 100  # Much smaller batch size
total_samples = ics.shape[0]
num_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
emu_rollouts = []
delta = torch.tensor([2]).to(device)

print(f"Will process {total_samples} samples in {num_batches} batches of size {batch_size}")

# Clear any existing GPU memory before starting
torch.cuda.empty_cache()

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min(start_idx + batch_size, total_samples)  # Handle last batch
    actual_batch_size = end_idx - start_idx
    
    print(f"Processing batch {i+1}/{num_batches} (samples {start_idx}:{end_idx}, size: {actual_batch_size})...")
    
    # Get batch of initial conditions and move to GPU
    ics_batch = ics[start_idx:end_idx].to(device)
    print(f"Batch {i+1} shape: {ics_batch.shape}")
    
    try:
        # Roll out the initial conditions for this batch
        emu_rollout_batch, _ = performance.run_conditional_emu(ics_batch, emulator, therm=CT, n_steps=10000, delta = delta, denoising_steps=10, freq = 25, silent=True, sigma=None, Regression = False)
        print(f"Batch {i+1} rollout shape: {emu_rollout_batch.shape}")
        
        # Sub-sample to match the time-step of the numerical trajectories
        emu_rollout_batch_subsampled = emu_rollout_batch[:, ::100, :, :]
        print(f"Batch {i+1} subsampled shape: {emu_rollout_batch_subsampled.shape}")
        
        # Move back to CPU to save memory and store
        emu_rollouts.append(emu_rollout_batch_subsampled.cpu())
        
        # Clear GPU memory aggressively
        del ics_batch, emu_rollout_batch, emu_rollout_batch_subsampled
        torch.cuda.empty_cache()
        
        print(f"Batch {i+1} completed and moved to CPU")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM error in batch {i+1}. Try reducing batch_size further.")
        print(f"Error: {e}")
        # Clean up and exit
        del ics_batch
        torch.cuda.empty_cache()
        raise

print("Concatenating all batches...")

# Concatenate all batches along the first dimension
emu_rollout_subsampled = torch.cat(emu_rollouts, dim=0)
print(f"Final concatenated shape: {emu_rollout_subsampled.shape}")

# Save the subsampled data
print("Saving results...")
torch.save(emu_rollout_subsampled, "/scratch/ql2221/thermalizer_data/kolmogorov/reynold10k/Therm_rollout.p")
print("Subsampled emulator rollout saved successfully!")

# Clean up memory
del emu_rollouts, emu_rollout_subsampled
torch.cuda.empty_cache()
print("Memory cleanup completed")
