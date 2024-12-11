import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import wandb
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
from data.dataset import Dataset3d, spatiotemporal_collate_fn
from models.loss import mse_loss
from models.mask import random_spatial_masking, random_temporal_masking
from models.transformers.stt import Transformer3d
from utils.helpers import create_grid_from_3d_batch, early_stopping

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("rich")

device = torch.device("cpu")  # fallback to cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()

base_dir = "./"
config = {
    "datasets": {
        "psc": f"{base_dir}/data/processed",
        "sst": f"{base_dir}/data/SST",
        "clouds": f"{base_dir}/data/clouds.nc",
    },
    "checkpoints": {
        "save": True,
        "path": f"{base_dir}/models/PSC",
    },
    "periods": {
        "train": 0.6,
        "test": 0.2,
        "val": 0.2,
    },
    "time_window": 32,  # 8 days * 4 weeks
    "model": {
        "input_size": (32, 288, 288),
        "in_channels": 4,  # number of variables
        "out_channels": 4,
        "patch_size": (8, 18, 18),  # 4 * 16 * 16 patches
        "hidden_size": 192,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "use_modulation": False,
    },
    "num_epochs": 200,
    "patience": 30,  # early stopping patience
    "bs": 32,  # batch size
    "lr": 5e-4,  # learning rate
    "weight_decay": 0,  # L2 regularization, set to 0 to disable
}

DATE = datetime.now().strftime("%Y%m%d_%H%M%S")
NAME = "transformer3d_psc"
config["checkpoints"]["name"] = f"{DATE}_{NAME}"

wandb.init(
    project="imputation_psc",
    config=config,
    sync_tensorboard=True,
    name=NAME,
    group="Transformer3D",
    tags=[
        "Min-Max",
        "No conditioning",
        "MSE",
    ],
)
logger.info(f"Using device: {device}")


def run_epoch(
    model, loader, loss_fn, optimizer, device, mode="train", progress=None, task_id=None
):
    if mode == "train":
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    with torch.set_grad_enabled(mode == "train"):
        for time, inputs, masks in loader:
            inputs = inputs.to(device).float()  # B, 4, T, H, W
            masks = masks.to(device).float()  # B, 4, T, H, W

            inputs_masked, new_masks = random_temporal_masking(
                inputs, masks, mask_ratio=0.2
            )
            inputs_masked, new_masks = random_spatial_masking(
                inputs_masked, new_masks, mask_ratio=0.2
            )

            outputs = model(inputs_masked, new_masks)  # B, 4, T, H, W

            chl = outputs[:, 0, ...].unsqueeze(1)  # B, 1, T, H, W
            psc = outputs[:, 1:, ...]  # B, 3, T, H, W
            psc = F.softmax(psc, dim=1)  # ensure the sum of probabilities is 1
            outputs = torch.cat([chl, psc], dim=1)  # B, 4, T, H, W

            loss = loss_fn(inputs, outputs, (1 - masks))

            total_loss += loss.item()

            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if progress and task_id:
                progress.advance(task_id)

    avg_loss = total_loss / len(loader)

    return avg_loss, inputs_masked, inputs, outputs, time


def log_images(
    writer,
    inputs_masked,
    inputs,
    outputs,
    time,
    epoch,
    mode,
    log_interval=10,
    **kwargs,
):
    """Logs figures to TensorBoard."""
    if epoch % log_interval == 0:
        for data, label in zip(
            [inputs_masked, inputs, outputs], ["Masked", "Inputs", "Outputs"]
        ):
            for i, var in enumerate(["Chl-a", "Micro", "Nano", "Pico"]):
                writer.add_figure(
                    f"{label}/{mode}/{var}",
                    create_grid_from_3d_batch(
                        data[:, i, ...].unsqueeze(1), time, **kwargs
                    ),
                    epoch + 1,
                )


def main(
    model,
    loss_fn,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    test_loader,
    config,
    device,
    writer,
):
    best_loss = float("inf")
    patience = 0

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
        val_task = progress.add_task(
            "[magenta]Validation", total=len(val_loader), start=False
        )

        for epoch in range(config["num_epochs"]):
            # Training
            progress.reset(task_id=train_task)

            avg_train_loss, *last_train_data = run_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                mode="train",
                progress=progress,
                task_id=train_task,
            )

            if scheduler:
                writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch + 1)
                scheduler.step()

            writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
            log_images(
                writer,
                *last_train_data,
                epoch,
                mode="train",
                cmap="jet",
                use_log_scale=False,
                vrange=(0, 1),
            )

            # Validation
            progress.reset(task_id=val_task)

            avg_val_loss, *last_val_data = run_epoch(
                model,
                val_loader,
                loss_fn,
                optimizer,
                device,
                mode="validate",
                progress=progress,
                task_id=val_task,
            )

            writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
            log_images(
                writer,
                *last_val_data,
                epoch,
                mode="val",
                cmap="jet",
                use_log_scale=False,
                vrange=(0, 1),
            )

            progress.console.print(
                f"Epoch [{epoch + 1:04d}/{config['num_epochs']:04d}] | "
                + f"Train Loss: {avg_train_loss:.6f}  | "
                + f"Validation Loss: {avg_val_loss:.6f} | "
            )

            best_loss, patience = early_stopping(
                avg_val_loss,
                best_loss,
                patience,
                config["patience"],
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                logger,
            )
            if patience > config["patience"]:
                break

            progress.advance(epoch_task)

        # Testing after training completes
        test_task = progress.add_task("[cyan]Testing", total=len(test_loader))

        avg_mse, *last_test_data = run_epoch(
            model,
            test_loader,
            loss_fn,
            optimizer,
            device,
            mode="test",
            progress=progress,
            task_id=test_task,
        )

        progress.console.print(f"Test MSE: {avg_mse:.6f}")
        log_images(
            writer,
            *last_test_data,
            epoch,
            mode="test",
            cmap="jet",
            use_log_scale=False,
            vrange=(0, 1),
        )


if __name__ == "__main__":
    # Load the dataset
    logger.info("Loading dataset...")
    ds = xr.open_mfdataset(
        f"{config['datasets']['psc']}/*.nc",
        engine="h5netcdf",
    )
    ds["Chl"] = np.log10(ds["Chl"])  # log-transform Chl-a
    ds["Micro"] = ds["Micro"] / 100
    ds["Pico"] = ds["Pico"] / 100
    ds["Nano"] = ds["Nano"] / 100
    logger.info(ds)

    logger.info("Preprocessing dataset...")
    train_ds, val_ds, test_ds = loading.split_dataset_by_percentage(
        ds, config["periods"]
    )

    stats = {
        "min": train_ds["Chl"].min(dim="time"),
        "max": train_ds["Chl"].max(dim="time"),
    }

    train_ds["Chl"] = preprocessing.minmax_normalize(
        train_ds["Chl"], stats["min"], stats["max"]
    )
    val_ds["Chl"] = preprocessing.minmax_normalize(
        val_ds["Chl"], stats["min"], stats["max"]
    )
    test_ds["Chl"] = preprocessing.minmax_normalize(
        test_ds["Chl"], stats["min"], stats["max"]
    )
    # no need to minmax normalize PSC, they are already in [0, 1] (percentages)

    logger.info("Creating datasets...")
    train_dataset = Dataset3d(
        TensorDict(
            source={key: train_ds[key].values for key in train_ds.data_vars},
            batch_size=train_ds.sizes["time"],
        ),
        time=train_ds.time,
        time_window=config["time_window"],
    )
    test_dataset = Dataset3d(
        TensorDict(
            source={key: test_ds[key].values for key in test_ds.data_vars},
            batch_size=test_ds.sizes["time"],
        ),
        time=test_ds.time,
        time_window=config["time_window"],
    )
    val_dataset = Dataset3d(
        TensorDict(
            source={key: val_ds[key].values for key in val_ds.data_vars},
            batch_size=val_ds.sizes["time"],
        ),
        time=val_ds.time,
        time_window=config["time_window"],
    )

    logger.info("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["bs"],
        shuffle=True,
        collate_fn=spatiotemporal_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["bs"],
        shuffle=False,
        collate_fn=spatiotemporal_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["bs"],
        shuffle=False,
        collate_fn=spatiotemporal_collate_fn,
    )

    # Define the model
    model = Transformer3d(
        input_size=config["model"]["input_size"],
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        patch_size=config["model"]["patch_size"],
        depth=config["model"]["depth"],
        hidden_size=config["model"]["hidden_size"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        use_modulation=config["model"]["use_modulation"],
    ).to(device)
    criterion = mse_loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"]
    )

    logger.info(model)

    # log total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of trainable parameters: {total_params}")

    logger.info("Starting training...")
    writer = SummaryWriter()
    main(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        test_loader,
        config,
        device,
        writer,
    )
    writer.close()
    wandb.finish()
    logger.info("Training complete.")
