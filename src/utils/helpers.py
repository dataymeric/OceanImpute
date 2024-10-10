import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr


def cast_tuple(t, length=1):
    """Ensure that the input is a tuple of a specific length."""
    return t if isinstance(t, tuple) else ((t,) * length)


def plot_and_return_figure(data, cmap="jet", vmin=-2, vmax=1, title=None):
    """Plot data using a colormap, save the plot to a buffer, and return a PIL Image."""
    fig, ax = plt.subplots(layout="constrained")
    img = ax.pcolormesh(data, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(img, ax=ax, orientation="vertical")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    return fig


def create_grid(
    batch,
    time=None,
    time_indices=[None],
    cmap="jet",
    vrange=(-2, 1),
    use_log_scale=False,
):
    """General function to create a grid of images from a batch."""
    vmin, vmax = cast_tuple(vrange, 2)
    figures = []

    if time_indices is None:
        time_indices = [None]  # Handle 2D case where there's no time dimension

    for i in range(batch.shape[0]):
        for j in time_indices:
            if j is None:
                data = batch[i, 0].cpu()
            else:
                data = batch[i, 0, j].cpu()

            if use_log_scale:
                data = np.log10(data + 1e-3)

            title = f"Batch {i}" if j is None else f"{time[i][j]}"
            fig = plot_and_return_figure(
                data, cmap=cmap, vmin=vmin, vmax=vmax, title=title
            )
            figures.append(fig)
            plt.close(fig)  # Close the figure to avoid memory leaks

        if j is not None and i > 2:
            break

    return figures


@torch.no_grad()
def create_grid_from_2d_batch(batch, cmap="jet", vrange=(-2, 1), use_log_scale=False):
    """Create a grid image from a batch of 2D spatiotemporal data tensors."""
    assert len(batch.shape) == 4, "Input batch must have shape (B, C, H, W)"
    return create_grid(batch, cmap=cmap, vrange=vrange, use_log_scale=use_log_scale)


@torch.no_grad()
def create_grid_from_3d_batch(
    batch, time, cmap="jet", vrange=(-2, 1), use_log_scale=False
):
    """Create a grid image from a batch of 3D spatiotemporal data tensors."""
    assert len(batch.shape) == 5, "Input batch must have shape (B, C, T, H, W)"
    _, _, t, _, _ = batch.shape
    return create_grid(
        batch,
        time,
        time_indices=list(range(t)),
        cmap=cmap,
        vrange=vrange,
        use_log_scale=use_log_scale,
    )


def to_dataset(tensor_list, time_list, variable_names, original_ds):
    """Convert a list of tensors and corresponding time information to an 
    xarray Dataset.
    """
    # Concatenate tensors along the batch dimension
    tensor = torch.cat(tensor_list, dim=0).detach()

    # Concatenate time information
    time = np.concatenate(time_list, axis=-1).reshape(-1)

    # Reshape the tensor to combine batch and time dimensions
    # New shape: (b * t, c, h, w)
    b, c, t, h, w = tensor.shape
    reshaped_tensor = tensor.view(b * t, c, h, w)

    # Create coordinates for the dataset
    coords = {"time": time.astype("datetime64[ns]")}
    if "lat" in original_ds:
        coords["latitude"] = original_ds["lat"].values
    elif "latitude" in original_ds:
        coords["latitude"] = original_ds["latitude"].values

    if "lon" in original_ds:
        coords["longitude"] = original_ds["lon"].values
    elif "longitude" in original_ds:
        coords["longitude"] = original_ds["longitude"].values

    # Create an empty dataset
    data_ds = xr.Dataset(coords=coords)

    # Add each variable as a DataArray to the dataset
    for i, var_name in enumerate(variable_names):
        data_ds[var_name] = xr.DataArray(
            reshaped_tensor[:, i, :, :],
            coords={
                "time": data_ds["time"],
                "latitude": data_ds["latitude"],
                "longitude": data_ds["longitude"],
            },
            dims=["time", "latitude", "longitude"],
        )

    return data_ds
