import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionSchedule:
    """Manages the variance schedule for the diffusion process."""

    def __init__(self, timesteps, schedule_type="linear"):
        """Initialize the diffusion schedule.

        Args:
            timesteps (int): Number of diffusion steps.
            schedule_type (str): Type of noise schedule ('linear', 'quadratic', 
                'sigmoid'). Defaults: 'linear'.
        """
        self.timesteps = timesteps
        self.beta_start = 0.0001
        self.beta_end = 0.02

        # Select and compute betas based on schedule type
        if schedule_type == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, timesteps)
        elif schedule_type == "quadratic":
            betas = (
                torch.linspace(self.beta_start**0.5, self.beta_end**0.5, timesteps) ** 2
            )
        elif schedule_type == "sigmoid":
            noise = torch.linspace(-6, 6, timesteps)
            betas = (
                torch.sigmoid(noise) * (self.beta_end - self.beta_start)
                + self.beta_start
            )
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Pre-compute useful values
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        # Useful computational values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Compute posterior variance
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embeddings for time steps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class DenoisingDiffusion:
    def __init__(
        self, model, image_size=64, channels=3, timesteps=1000, schedule_type="linear"
    ):
        """Initialize the Diffusion Model.

        Args:
            model (nn.Module): The noise prediction model.
            image_size (int): Size of the generated images. Defaults: 64.
            channels (int): Number of image channels. Defaults: 3.
            timesteps (int): Number of diffusion steps. Defaults: 1000.
            schedule_type (str): Type of noise schedule. Defaults: "linear".
        """
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.schedule = DiffusionSchedule(timesteps, schedule_type)

    def _temporal_gather(self, alphas, t, x_shape):
        """Gather values from tensor `alphas` using indices from `t`.
        
        Adds dimensions at the end of the tensor to match with the number of dimensions
        of `x_shape`.
        """
        batch_size = len(t)
        out = alphas.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_0, t, noise=None):
        """Sample q(x_t|x_0) for a batch of images.

        Args:
            x_0 (torch.Tensor): Batch of images. Shape: (B, C, H, W)
            t (torch.Tensor): Time steps. Shape: (B,)
            noise (torch.Tensor, optional): Pre-generated noise. Defaults: None.

        Returns:
            torch.Tensor: Noisy images at time step t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        shape = x_0.size()
        sqrt_alphas_cumprod = self._temporal_gather(
            self.schedule.sqrt_alphas_cumprod, t, shape
        )
        sqrt_one_minus_alphas_cumprod = self._temporal_gather(
            self.schedule.sqrt_one_minus_alphas_cumprod, t, shape
        )
        return sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise

    @torch.no_grad()
    def p_sample(self, x, t_index):
        """
        Reverse diffusion process for a single time step.

        Args:
            x (torch.Tensor): Current noisy image.
            t_index (int): Current time step index.

        Returns:
            torch.Tensor: Denoised image for the previous time step
        """
        shape = x.size()
        device = x.device

        # Sample z (noise)
        z = (
            torch.randn(shape, device=device)
            if t_index > 1
            else torch.zeros(shape, device=device)
        )

        # Predict noise
        noise_pred = self.model(x, torch.LongTensor([t_index]).to(device))

        # Compute x_{t-1}
        x_prev = self.schedule.sqrt_recip_alphas[t_index] * (
            x
            - (1.0 - self.schedule.alphas[t_index])
            / self.schedule.sqrt_one_minus_alphas_cumprod[t_index]
            * noise_pred
        )
        x_prev = x_prev + self.schedule.posterior_variance[t_index] * z

        return x_prev

    @torch.no_grad()
    def sample(self, batch_size=16):
        """Generate images through the reverse diffusion process.

        Args:
            batch_size (int, optional): Number of images to generate. Defaults: 16.

        Returns:
            List[torch.Tensor]: Generated images at each time step
        """
        device = next(self.model.parameters()).device

        # Will contain images from x_{T-1} to x_0
        imgs = []

        # Sample x_T from a standard normal distribution
        x_t = torch.randn(
            batch_size, self.channels, self.image_size, self.image_size, device=device
        )

        # Sampling loop
        for t_index in range(self.schedule.timesteps - 1, -1, -1):
            x_t = self.p_sample(x_t, t_index)
            imgs.append(x_t.cpu())

        return imgs
