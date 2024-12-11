import torch
import torch.nn.functional as F


def random_temporal_masking(data, mask, mask_ratio=0.1):
    """Randomly masks entire time steps in the data tensor.

    Args:
        data (tensor): Tensor of shape (B, C, T, H, W).
        mask (tensor): Tensor of the same shape as data.
        mask_ratio (float, optional): Proportion of time steps to mask. Default: 0.1.

    Returns:
        data (tensor): Tensor with masked time steps.
        mask (tensor): Updated mask tensor.
    """
    B, C, T, _, _ = data.shape
    device = data.device
    num_masked_frames = int(T * mask_ratio)

    new_data = data.clone()
    new_mask = mask.clone()

    # Generate a mask for the temporal dimension
    temporal_mask = torch.zeros((B, C, T), dtype=torch.bool, device=device)

    for b in range(B):
        for c in range(C):
            indices = torch.randperm(T)[:num_masked_frames]
            temporal_mask[b, c, indices] = True

    # Expand dimensions to match data tensor
    temporal_mask = temporal_mask[:, :, :, None, None]  # Shape: (B, C, T, 1, 1)

    # Apply the temporal mask to data and update the mask tensor
    new_data = torch.where(temporal_mask, torch.tensor(0.0, device=device), new_data)
    new_mask = torch.where(
        temporal_mask, torch.tensor(1, device=device, dtype=new_mask.dtype), new_mask
    )

    return new_data, new_mask


def random_spatial_masking(data, mask, mask_ratio=0.1):
    """Randomly masks spatial regions that contain data in the data tensor.

    Args:
        data (tensor): Tensor of shape (B, C, T, H, W).
        mask (tensor): Tensor of the same shape as data (binary mask where 1 means
            missing, 0 means present).
        mask_ratio (float, optional): Proportion of available data to mask per channel
            (e.g., 0.1 for 10%). Default: 0.1.

    Returns:
        new_data: Tensor with masked regions.
        new_mask: Updated mask tensor.
    """
    B, C, T, H, W = data.shape

    # Create copies of the data and mask tensors to avoid in-place modification
    new_data = data.clone()
    new_mask = mask.clone()

    # Create Gaussian kernel for smoothing
    kernel_size = 21  # Desired smoothness
    sigma = 5.0  # Standard deviation for Gaussian kernel
    device = data.device  # Ensure tensors are on the correct device

    # Create Gaussian kernel
    grid = (
        torch.arange(kernel_size, dtype=torch.float32, device=device)
        - (kernel_size - 1) / 2.0
    )
    gauss = torch.exp(-0.5 * (grid / sigma).pow(2))
    gauss = gauss / gauss.sum()
    gauss_kernel = gauss[:, None] @ gauss[None, :]  # Outer product to get 2D kernel
    gauss_kernel = gauss_kernel.unsqueeze(0).unsqueeze(
        0
    )  # Shape: (1, 1, kernel_size, kernel_size)

    # Normalize the kernel
    gauss_kernel = gauss_kernel / gauss_kernel.sum()

    for b in range(B):
        for c in range(C):
            for t in range(T):
                # Get the 2D mask and data for the current sample, channel, and time step
                mask_bct = new_mask[b, c, t, :, :]  # (H, W)
                data_bct = new_data[b, c, t, :, :]  # (H, W)

                # Identify available positions (mask == 0)
                # Float tensor with 1s at available positions
                available_mask = (mask_bct == 0).float()
                N = int(available_mask.sum().item())

                if N == 0:
                    continue  # Skip if no available data in this channel

                # Generate random noise masked to available positions
                noise = torch.rand((H, W), device=device) * available_mask

                # Apply Gaussian filter to the noise to introduce spatial correlation
                noise = noise.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                smoothed_noise = F.conv2d(noise, gauss_kernel, padding=kernel_size // 2)
                smoothed_noise = smoothed_noise.squeeze(0).squeeze(0) * available_mask

                # Flatten the smoothed noise at available positions
                flat_smoothed_noise = smoothed_noise[available_mask.bool()]

                # Determine the threshold to mask approximately mask_ratio of the
                # available data
                n_to_mask = int(N * mask_ratio)
                if n_to_mask == 0:
                    continue  # Skip if number to mask is zero

                # Get the threshold by selecting the n_to_mask-th largest value
                threshold = torch.topk(
                    flat_smoothed_noise, n_to_mask, largest=True
                ).values[-1]

                # Create mask positions where smoothed noise is greater than or equal to
                # the threshold
                mask_positions = (smoothed_noise >= threshold) & (available_mask == 1)

                # Update the data and mask tensors at these positions
                # Data and mask are (H, W); mask_positions is (H, W)
                new_data[b, c, t] = torch.where(
                    mask_positions, torch.tensor(0.0, device=device), data_bct
                )
                new_mask[b, c, t] = torch.where(
                    mask_positions,
                    torch.tensor(1, device=device, dtype=new_mask.dtype),
                    mask_bct,
                )

    return new_data, new_mask
