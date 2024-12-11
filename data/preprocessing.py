import numpy as np


def standardize(ds, mean=None, std=None, use_log=False):
    if mean is None:
        mean = ds.mean(dim="time")
    if std is None:
        std = ds.std(dim="time")
    return (ds - mean) / std


def destandardize(ds, mean, std):
    return ds * std + mean


def log_standardize(ds, mean=None, std=None):
    ds = np.log10(ds)
    return standardize(ds, mean, std)


def log_destandardize(ds, mean, std):
    ds = ds * std + mean
    return 10**ds


def minmax_normalize(ds, min_val=None, max_val=None):
    if min_val is None:
        min_val = ds.min(dim="time")
    if max_val is None:
        max_val = ds.max(dim="time")
    return (ds - min_val) / (max_val - min_val)


def minmax_denormalize(ds, min_val, max_val):
    return ds * (max_val - min_val) + min_val


def minmax_log_normalize(ds, min_val=None, max_val=None):
    ds = np.log10(ds)
    return minmax_normalize(ds, min_val, max_val)


def minmax_log_denormalize(ds, min_val, max_val):
    ds = ds * (max_val - min_val) + min_val
    return 10**ds


def remove_seasonality(ds, daily_climatology=None, group="season"):
    if daily_climatology is None:
        daily_climatology = ds.groupby(f"time.{group}").mean("time")
    return ds.groupby(f"time.{group}") - daily_climatology


def readd_seasonality(ds, daily_climatology, group):
    return ds.groupby(f"time.{group}") + daily_climatology
