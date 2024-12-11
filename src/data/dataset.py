import torch
from torch.utils.data import Dataset
from tensordict import TensorDict
import numpy as np


class Dataset2d(Dataset):
    """Dataset class treating time as channels."""

    def __init__(self, dataset, time_window=5, mode="zero"):
        self.data = dataset
        self.target = dataset["Chl"]
        self.target_time = time_window // 2
        self.missing_mask = torch.isnan(self.target)
        self.time_window = time_window

        self.fill_missing(mode)

    def __len__(self):
        return len(self.data) - self.time_window + 1

    def __getitem__(self, idx):
        timerange = range(idx, idx + self.time_window)
        target_timerange = timerange[self.target_time]

        target = self.target[target_timerange].unsqueeze(0)
        missing_mask = self.missing_mask[target_timerange].unsqueeze(0)

        return timerange, self.data[timerange], target, missing_mask

    def fill_missing(self, mode):
        if mode == "constant":
            for key in self.data.keys():
                self.data[key][torch.isnan(self.data[key])] = 99999.0
            self.target[self.missing_mask] = 99999.0
        elif mode == "zero":
            for key in self.data.keys():
                self.data[key][torch.isnan(self.data[key])] = 0.0
            self.target[self.missing_mask] = 0.0
        elif mode == "noise":
            for key in self.data.keys():
                self.data[key][torch.isnan(self.data[key])] = torch.randn_like(
                    self.data[key][torch.isnan(self.data[key])]
                )
            self.target[self.missing_mask] = torch.randn_like(
                self.target[self.missing_mask]
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")


class Dataset3d(Dataset):
    """Dataset class explicitly treating time as a dimension."""

    def __init__(self, dataset, time, clouds=None, time_window=8, mode="zero"):
        """
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
    """Collate function for use with Dataset2d."""
    time = [i[0] for i in batch]
    data = [i[1] for i in batch]
    target = [i[2] for i in batch]
    mask = [i[3] for i in batch]

    data = torch.cat(list(torch.stack(data).values()), dim=1)
    time = torch.tensor(time)
    target = torch.stack(target)
    mask = torch.stack(mask)

    return time, data, target, mask


def spatiotemporal_collate_fn(batch):
    """Collate function for use with Dataset3d."""
    time = [i[0] for i in batch]
    data = [i[1] for i in batch]
    mask = [i[2] for i in batch]

    time = np.stack(time)
    data = torch.stack(list(torch.stack(data).values()), dim=1)
    mask = torch.stack(list(torch.stack(mask).values()), dim=1)

    return time, data, mask
