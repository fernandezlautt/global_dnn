import torch


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (
        torch.arange(kernel_size).reshape(-1, 1),
        torch.arange(kernel_size).reshape(1, -1),
    )
    filt = (1 - torch.abs(og[0] - center) / factor) * (
        1 - torch.abs(og[1] - center) / factor
    )
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    # Iterate over in_channels and out_channels to assign the filter
    for i in range(in_channels):
        for j in range(out_channels):
            if i == j and i < min(
                in_channels, out_channels
            ):  # Assign only if i == j and within bounds
                weight[i, j, :, :] = filt
    return weight
