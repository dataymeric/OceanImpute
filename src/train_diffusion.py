import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import data.loading as loading
import data.preprocessing as preprocessing
from data.dataset import SpatiotemporalDataset, spatial_collate_fn
from models.diffusion.denoising import (
    DenoisingDiffusion,
    SinusoidalPositionEmbedding,
)
from models.loss import mse_loss
from models.unets.diffusion import Unet
from utils.helpers import create_grid_from_2d_batch

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("rich")

device = torch.device("cpu")  # fallback to cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
logger.info(f"Using device: {device}")


base_dir = "./"
config = {
    "datasets": {
        "psc": f"{base_dir}/data/processed",
    },
    "periods": {
        "train": 0.6,
        "test": 0.2,
        "val": 0.2,
    },
    "num_epochs": 100,
    "bs": 16,
}

DATE = datetime.now().strftime("%Y%m%d_%H%M%S")
NAME = "diffusion_naive"


class NoiseModel(nn.Module):
    def __init__(self, dim, out_dim=1, channels=1):
        """Initialize our noise model

        Args:
            dim (int): The width/height of an image
            channels (int, optional): The number of channels of the image. Defaults to 1.
        """
        super().__init__()
        self.image_size = dim
        self.channels = channels
        self.out_dim = out_dim
        self.t_emb = SinusoidalPositionEmbedding(dim * 4)
        self.unet = Unet(dim, channels=channels, out_dim=out_dim, time_dim=dim * 4)

    def forward(self, x_t, time):
        return self.unet(x_t, self.t_emb(time))


def main(model, ddpm, loss_fn, optimizer, train_loader, config, writer):
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        epoch_task = progress.add_task("Epoch Progress", total=config["num_epochs"])
        train_task = progress.add_task(
            "[green]Training", total=len(train_loader), start=False
        )

        for epoch in range(config["num_epochs"]):
            progress.reset(task_id=train_task)

            total_loss = 0.0

            for step, (time, inputs, masks) in enumerate(train_loader):
                # Training
                batch_size = inputs.shape[0]
                inputs = inputs.to(device).float()  # B, 1, H, W
                masks = masks.to(device).float()  # B, 1, H, W

                # Sample random timesteps
                noise = torch.randn_like(inputs, device=device)
                t = torch.randint(
                    0, ddpm.schedule.timesteps, (batch_size,), device=device
                )
                x_t = ddpm.q_sample(inputs, t, noise=noise)

                # Predict the noise using the model
                predicted_noise = model(x_t, t)

                loss = loss_fn(noise, predicted_noise, (1 - masks))

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress.advance(train_task)

            avg_train_loss = total_loss / len(train_loader)

            writer.add_scalar("Loss/train", avg_train_loss, epoch)

            # Log images
            outputs = torch.stack(ddpm.sample())
            writer.add_figure(
                "Samples/train/Chl-a",
                create_grid_from_2d_batch(
                    outputs[-1], cmap="jet", use_log_scale=False, vrange=(0, 1)
                ),
                epoch,
            )

            progress.console.print(
                f"Epoch [{epoch + 1:04d}/{config['num_epochs']:04d}] | "
                + f"Train Loss: {avg_train_loss:.6f}  | "
            )

            progress.advance(epoch_task)


if __name__ == "__main__":
    # Load the dataset
    logger.info("Loading dataset...")
    ds = xr.open_mfdataset(f"{config["datasets"]["psc"]}/*.nc", engine="h5netcdf")
    ds = ds.isel(lat=range(64), lon=range(64))
    ds["Chl"] = np.log10(ds["Chl"])  # log-transform Chl-a
    logger.info(ds)

    logger.info("Preprocessing dataset...")
    train_ds, _, _ = loading.split_dataset_by_percentage(ds, config["periods"])

    stats = {
        "min": train_ds["Chl"].min(dim="time"),
        "max": train_ds["Chl"].max(dim="time"),
    }

    train_ds["Chl"] = preprocessing.minmax_normalize(
        train_ds["Chl"], stats["min"], stats["max"]
    )

    logger.info("Creating datasets...")
    train_dataset = SpatiotemporalDataset(
        TensorDict(
            source={"Chl": train_ds.Chl.values},
            batch_size=train_ds.sizes["time"],
        ),
        time=train_ds.time,
        time_window=1,
    )

    logger.info("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset, batch_size=config["bs"], collate_fn=spatial_collate_fn
    )

    model = NoiseModel(dim=64, channels=1).to(device)
    ddpm = DenoisingDiffusion(
        model, image_size=64, channels=1, timesteps=1000, schedule_type="sigmoid"
    )
    criterion = mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    logger.info(model)

    # log total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of trainable parameters: {total_params}")

    logger.info("Starting training...")
    writer = SummaryWriter()
    main(model, ddpm, criterion, optimizer, train_loader, config, writer)
    writer.close()
    logger.info("Training complete.")
