import torch


def mae(y_true, y_pred, mask=None):
    """Mean absolute error (MAE)."""
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return torch.mean(torch.abs(y_true - y_pred))


def mse(y_true, y_pred, mask=None):
    """Mean squared error (MSE)."""
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return torch.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred, mask=None, normalize=False):
    """(Normalized) Root mean squared error ((N)RMSE)."""
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    score = torch.sqrt(torch.mean((y_true - y_pred) ** 2))

    if normalize:
        return score / torch.mean(y_true)

    return score


def relative_error(y_true, y_pred, mask=None):
    """Spatial distribution of relative error.

    See: https://www.mdpi.com/2072-4292/12/3/480.
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return (y_pred - y_true) / y_true


def psnr(y_true, y_pred, mask=None):
    """Peak signal-to-noise ratio (PSNR)."""
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return 20 * torch.log10(torch.max(y_true) / rmse(y_true, y_pred, mask))


def corrcoef(y_true, y_pred, mask):
    """Pearson correlation coefficient."""
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    pred_mean = torch.mean(y_pred)
    target_mean = torch.mean(y_true)

    pred_diff = y_pred - pred_mean
    target_diff = y_true - target_mean

    numerator = torch.sum(pred_diff * target_diff)
    denominator = torch.sqrt(torch.sum(pred_diff**2) * torch.sum(target_diff**2))

    return numerator / denominator
