import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from rich.logging import RichHandler
from tensordict import TensorDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.loading import split_dataset_by_percentage
import data.preprocessing as preprocessing
from data.dataset import Dataset3d, spatiotemporal_collate_fn
from models.transformers.stt import Transformer3d
from utils.helpers import to_dataset

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("rich")

config = {
    "datasets": {
        "psc": "../data/processed",
        "sst": "../data/SST",
        "ssh": "../data/SSH",
    },
    "variables": ["Chl", "Micro", "Nano", "Pico"],
    "periods": {
        "train": 0.6,
        "test": 0.2,
        "val": 0.2,
    },
}
SAVE_DIR = "../data/outputs/PSC"

# Define time range (if we know them) else we can split the dataset by percentage
test_slice = slice("2018-06-24", "2021-12-31")
test_period = pd.date_range("2018-06-24", "2021-12-31", freq="D")

# Load PSC dataset
logger.info("Loading PSC dataset")
ds = xr.open_dataset(f"{config['datasets']['psc']}/*.nc", engine="h5netcdf")
ds["Chl"] = np.log10(ds["Chl"])
for var in ["Micro", "Nano", "Pico"]:
    ds[var] = ds[var] / 100

# Prepare the test dataset
logger.info("Loading test dataset")
test_ds = ds.sel(time=test_slice)
test_ds = test_ds.reindex(time=test_period)  # Add missing dates
logger.info(f"Test dataset info: {test_ds}")

# else:
# _, _, test_ds = split_dataset_by_percentage(ds, config["periods"])

# Compute min-max statistics for normalization
stats = {
    "min": test_ds["Chl"].min(dim="time"),
    "max": test_ds["Chl"].max(dim="time"),
}
test_ds["Chl"] = preprocessing.minmax_normalize(
    test_ds["Chl"], stats["min"], stats["max"]
)

# Create the test dataset
logger.info("Creating test dataset")
test_dataset = Dataset3d(
    TensorDict(
        source={key: torch.tensor(test_ds[key].values) for key in test_ds.data_vars},
        batch_size=test_ds.sizes["time"],
    ),
    time=test_ds.time,
    time_window=32,
)

# Create a DataLoader for the test dataset
logger.info("Creating test data loader")
data_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=spatiotemporal_collate_fn,
)

# Load the model checkpoint and create the model
logger.info("Loading model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_checkpoint = torch.load(
    "../models/transformer3d_psc.pt",
    map_location=device,
)
model_config = model_checkpoint["config"]

logger.info("Creating model")
model = Transformer3d(
    input_size=model_config["model"]["input_size"],
    in_channels=model_config["model"]["in_channels"],
    out_channels=model_config["model"]["out_channels"],
    patch_size=model_config["model"]["patch_size"],
    hidden_size=model_config["model"]["hidden_size"],
    depth=model_config["model"]["depth"],
    num_heads=model_config["model"]["num_heads"],
    mlp_ratio=model_config["model"]["mlp_ratio"],
    use_modulation=model_config["model"]["use_modulation"],
)
model.load_state_dict(model_checkpoint["model"])
logger.info(model)

# Initialize reconstruction storage
reconstructions = {"outputs": [], "time": []}
variables_names = list(test_dataset.data.keys())
logger.info(f"Variables in the dataset: {variables_names}")

# Perform reconstruction using the model
logger.info("Starting reconstruction")
for batch, (time, inputs, masks) in enumerate(tqdm(data_loader)):
    with torch.no_grad():
        outputs = model(inputs.float(), masks.float())

    # Extract Chl and PSC predictions, ensure PSC sums to 1 with softmax
    chl = outputs[:, 0, ...].unsqueeze(1)  # B, 1, T, H, W
    psc = F.softmax(outputs[:, 1:, ...], dim=1)  # B, 3, T, H, W
    outputs = torch.cat([chl, psc], dim=1)  # B, 4, T, H, W

    # Store the results
    reconstructions["outputs"].append(outputs)
    reconstructions["time"].append(time)

    # Save intermediate results in batches
    if (batch + 1) % 32 == 0:
        logger.info(f"Saving results at step {batch + 1}")
        data_ds = to_dataset(
            reconstructions["outputs"],
            reconstructions["time"],
            variables_names,
            test_ds,
        )
        data_ds.to_netcdf(
            f"{SAVE_DIR}/raw_reconstruct_{batch + 1:04d}.nc",
            engine="h5netcdf",
        )
        reconstructions = {"outputs": [], "time": []}

# Save any remaining outputs
if reconstructions["outputs"]:
    logger.info(f"Saving final batch at step {batch + 1}")
    data_ds = to_dataset(
        reconstructions["outputs"], reconstructions["time"], variables_names, test_ds
    )
    data_ds.to_netcdf(
        f"{SAVE_DIR}/raw_reconstruct_{batch + 1:04d}.nc", engine="h5netcdf"
    )
logger.info("Reconstruction complete")

# Save average reconstructions
logger.info("Merging saved files")
files = sorted(os.listdir(SAVE_DIR))

datasets = [
    xr.open_dataset(f"{SAVE_DIR}/{file}", engine="h5netcdf", chunks={"time": 1024})
    for file in files
]
concat_ds = xr.concat(datasets, dim="time")

# Denormalize Chl while keeping it in log scale (no need for PSC)
concat_ds["Chl"] = preprocessing.minmax_denormalize(
    concat_ds["Chl"], stats["min"], stats["max"]
)

# Calculate mean and standard deviation for both renormalized and normalized datasets
mean_ds = concat_ds.groupby("time").mean(dim="time")
std_ds = concat_ds.groupby("time").std(dim="time")

# Combine mean and std into a single dataset
results_dict = {f"{var}_mean": mean_ds[var] for var in config["variables"]}
results_dict.update({f"{var}_std": std_ds[var] for var in config["variables"]})
results_ds = xr.Dataset(results_dict)

# Save the average reconstruction statistics
avg_recon_path = f"{SAVE_DIR}/avg_reconstructions.nc"
logger.info(f"Saving average reconstructions to {avg_recon_path}")
results_ds.to_netcdf(avg_recon_path, engine="h5netcdf")

# Save best reconstructions
# Global RMSE
logger.info("Calculating global RMSE")
rmse = np.sqrt((concat_ds.groupby("time") - test_ds) ** 2).mean(
    dim=["latitude", "longitude"]
)
min_rmse = rmse.groupby("time").min()

# Find the time indices with minimum RMSE for each variable
logger.info("Identifying best reconstructions")
indices = {
    var: np.where(rmse[var].values == min_rmse[var].values[:, None])[1]
    for var in config["variables"]
}

# Save the best indices to a file
indices_path = "../data/outputs/PFT/best_indices.npy"
logger.info(f"Saving best reconstruction indices to {indices_path}")
np.save(indices_path, indices)

# Extract and save best reconstructions
logger.info("Extracting and saving best reconstructions")
best_results_dict = {
    var: concat_ds[var].isel(time=indices[var]).mean(dim="time")
    for var in config["variables"]
}
best_results_ds = xr.Dataset(best_results_dict)

# Save to NetCDF file
best_recon_path = "../data/outputs/PFT/best_reconstruct.nc"
logger.info(f"Saving best reconstructions to {best_recon_path}")
best_results_ds.to_netcdf(best_recon_path, engine="h5netcdf")
