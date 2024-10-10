import torch


def sinv(var, delta=torch.tensor(1e-3)):
    return 1 / torch.fmax(var, delta)


def mse_loss(x_obs, x_rec, mask):
    """Mean squared error with mask handling."""
    return torch.sum((x_rec - x_obs).pow(2) * mask) / mask.sum()


def rmse_loss(x_obs, x_rec, mask):
    """Root mean squared error with mask handling."""
    return torch.sqrt(mse_loss(x_obs, x_rec, mask))


def huber_loss(x_obs, x_rec, mask, delta=1.0):
    """Huber loss with mask handling."""
    abs_diff = torch.abs(x_rec - x_obs)
    loss = torch.where(
        abs_diff < delta, 0.5 * abs_diff.pow(2), delta * (abs_diff - 0.5 * delta)
    )
    return torch.sum(loss * mask) / mask.sum()


def nll_loss(x_obs, x_rec, var_hat, mask):
    """Negative log likelihood loss with mask handling inspired from DINCAE."""
    var_hat = 1 / var_hat.exp()
    x_hat = x_rec * var_hat

    # where there is cloud, σ2_rec_noncloud is 1 and its log is zero
    var_loss = var_hat * mask + (1 - mask)
    log_var_loss = torch.log(var_loss)

    rec_loss = ((x_hat - x_obs).pow(2) * mask) / var_hat

    loss = log_var_loss + rec_loss
    loss = loss.sum() / mask.sum()
    return loss


def kl_div_loss(x_obs, x_rec, var, var_hat, mask):
    """Kullback-Leibler divergence loss with mask handling inspired from DINCAE."""
    var = sinv(var)
    var_hat = sinv(torch.fmin(var_hat, torch.tensor(10)).exp())

    mu = x_obs * var
    mu_hat = x_rec * var_hat

    # Calculate the variance ratio with the mask handling
    var_ratio = (var_hat / var) * mask + (1 - mask)
    log_var_ratio = torch.log(var_ratio)

    # Calculate the mean term, equivalent to a reproduction loss with the mask
    # handling
    mean_term = (var + (mu_hat - mu).pow(2)) * mask
    mean_term = mean_term / (2 * var_hat)

    # Compute the KL divergence loss
    loss = log_var_ratio + mean_term - 0.5

    # Normalize the loss by the number of valid elements (where mask is 1)
    loss = loss.sum() / mask.sum()

    return loss
