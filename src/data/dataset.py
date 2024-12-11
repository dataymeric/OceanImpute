import torch
from torch.utils.data import Dataset
from tensordict import TensorDict
import numpy as np


class SpatiotemporalDataset(Dataset):
    """Dataset class designed for spatiotemporal data, specifically Xarray datasets.

    Note:
        - This class assumes that the input data is a TensorDict with the following
        structure:
            {
                "var1": torch.Tensor,
                "var2": torch.Tensor,
                ...
            }
        - As of now, this requires loading the whole data in memory, which might not be
        feasible for (very) large datasets.
    """

    def __init__(self, dataset, time, clouds=None, time_window=8, mode="zero"):
        """Initialize the dataset.

        Args:
            dataset (TensorDict): Dataset containing variables to impute.
            time (np.ndarray[datetime64]): Time values (for logging purpose).
            clouds (TensorDict, optional): Mask for missing values (1 if missing,
                0 otherwise). Defaults: None.
            time_window (int, optional): Time window size. Defaults: 8.
            mode (str, optional): Filling missing values. Defaults: "zero".
        """
        super().__init__()
        self.data = dataset
        self.missing_mask = (
            TensorDict(
                source={key: torch.isnan(value) for key, value in dataset.items()},
                batch_size=dataset.batch_size,
            )
            if clouds is None
            else clouds
        )
        self.time = time.values.astype("datetime64[D]")
        self.time_window = time_window

        self.fill_missing(mode)

    def __len__(self):
        return len(self.data) - self.time_window + 1

    def __getitem__(self, idx):
        timerange = range(idx, idx + self.time_window)

        return (
            self.time[timerange],
            self.data[timerange],
            self.missing_mask[timerange],
        )

    def fill_missing(self, mode):
        """Fill missing values in the dataset."""
        if mode == "constant":
            for key in self.data.keys():
                self.data[key][torch.isnan(self.data[key])] = 99999.0
        elif mode == "zero":
            for key in self.data.keys():
                self.data[key][torch.isnan(self.data[key])] = 0.0
        elif mode == "noise":
            for key in self.data.keys():
                self.data[key][torch.isnan(self.data[key])] = torch.randn_like(
                    self.data[key][torch.isnan(self.data[key])]
                )
        else:
            raise ValueError(f"Unknown mode: {mode}")


def spatial_collate_fn(batch):
    """Collate function that treats time as channels.

    Shape: (B, C, T, H, W) => (B, C * T, H, W)
    """
    time = [i[0] for i in batch]
    data = [i[1] for i in batch]
    mask = [i[2] for i in batch]

    time = np.stack(time)
    data = torch.cat(list(torch.stack(data).values()), dim=1)
    mask = torch.cat(list(torch.stack(mask).values()), dim=1)

    return time, data, mask


def spatiotemporal_collate_fn(batch):
    """Collate function that explicitly treats time as a dimension.

    Shape: (B, C, T, H, W)
    """
    time = [i[0] for i in batch]
    data = [i[1] for i in batch]
    mask = [i[2] for i in batch]

    time = np.stack(time)
    data = torch.stack(list(torch.stack(data).values()), dim=1)
    mask = torch.stack(list(torch.stack(mask).values()), dim=1)

    return time, data, mask
