import numpy as np


def split_dataset(ds, periods, buffer_days=15):
    """Splits a xarray dataset into train, test, and val sets with a buffer period
    to avoid data leakage.

    Args:
        ds (xarray.Dataset or xarray.DataArray): The dataset to split.
        periods (dict): A dictionary with keys "train", "test", "val" and corresponding
            years or date ranges as values.
        buffer_days (int): The number of days to exclude between the sets to avoid
            data leakage.
    """

    def apply_buffer(start_date, end_date, buffer_days):
        """Create a buffer around the start and end date."""
        start_buffer = start_date + np.timedelta64(buffer_days, "D")
        end_buffer = end_date - np.timedelta64(buffer_days, "D")
        return start_buffer, end_buffer

    def select_period(ds, start_year, end_year):
        """Selects a period from the dataset given the start and end years."""
        return ds.sel(time=slice(str(start_year), str(end_year)))

    # Extract the years for train, test, and val periods
    train_years = periods.get("train", None)
    test_years = periods.get("test", None)
    val_years = periods.get("val", None)

    # 1. Get the data for each period without applying any buffer
    train_ds = (
        select_period(ds, train_years[0], train_years[1]) if train_years else None
    )
    test_ds = select_period(ds, test_years[0], test_years[1]) if test_years else None
    val_ds = select_period(ds, val_years[0], val_years[1]) if val_years else None

    # 2. Apply buffers
    if train_ds is not None and val_ds is not None:
        val_end = val_ds.time[-1].values
        train_start = train_ds.time[0].values
        val_end_buffer, train_start_buffer = apply_buffer(
            val_end, train_start, buffer_days
        )
        train_ds = train_ds.sel(time=slice(train_start_buffer, None))
        val_ds = val_ds.sel(time=slice(None, val_end_buffer))

    if test_ds is not None and train_ds is not None:
        train_end = train_ds.time[-1].values
        test_start = test_ds.time[0].values
        train_end_buffer, test_start_buffer = apply_buffer(
            train_end, test_start, buffer_days
        )
        test_ds = test_ds.sel(time=slice(test_start_buffer, None))
        train_ds = train_ds.sel(time=slice(None, train_end_buffer))

    return train_ds, val_ds, test_ds


def split_dataset_by_percentage(ds, split_percentages, buffer_days=15):
    """Splits a xarray dataset into train, test, and val sets based on percentage splits
    with a buffer period to avoid data leakage.

    Args:
        ds (xarray.Dataset or xarray.DataArray): The dataset to split.
        split_percentages (dict): A dictionary with keys "train", "test", "val" and
            corresponding percentage split as values (e.g.,
            {"train": 0.6, "val": 0.2, "test": 0.2}).
        buffer_days (int): The number of days to exclude between the sets to avoid
            data leakage.

    Returns:
        train_ds, val_ds, test_ds: Split datasets for training, validation, and testing.
    """

    # Step 1: Calculate total duration of the dataset
    total_days = (
        (ds.time[-1].values - ds.time[0].values).astype("timedelta64[D]").astype(int)
    )

    # Step 2: Calculate the number of days for each set based on the percentages
    val_days = int(total_days * split_percentages["val"])
    train_days = int(total_days * split_percentages["train"])
    # test_days = total_days - (val_days + train_days)

    # Step 3: Determine start and end dates for validation, training, and test sets
    val_start_date = ds.time[0].values
    val_end_date = (
        val_start_date
        + np.timedelta64(val_days, "D")
        - np.timedelta64(buffer_days, "D")
    )

    train_start_date = val_end_date + np.timedelta64(buffer_days * 2, "D")
    train_end_date = (
        train_start_date
        + np.timedelta64(train_days, "D")
        - np.timedelta64(buffer_days, "D")
    )

    test_start_date = train_end_date + np.timedelta64(buffer_days * 2, "D")
    test_end_date = ds.time[-1].values

    # Step 4: Select data for each set
    val_ds = ds.sel(time=slice(val_start_date, val_end_date))
    train_ds = ds.sel(time=slice(train_start_date, train_end_date))
    test_ds = ds.sel(time=slice(test_start_date, test_end_date))

    return train_ds, val_ds, test_ds
