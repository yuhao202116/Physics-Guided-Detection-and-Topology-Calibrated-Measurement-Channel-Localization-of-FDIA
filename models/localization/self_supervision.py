"""Normal-window pseudo-corruptions for the node-topology prior."""

import torch


def corrupt(batch, membership, generator):
    # Draw on CPU for repeatability; transfer the random values with the batch.
    device = batch.device
    batch_size = batch.shape[0]
    bus_index = torch.randint(0, membership.shape[0], (batch_size,), generator=generator)
    selected = membership.cpu()[bus_index].bool().to(device)
    result = batch.clone()
    mode = torch.randint(0, 3, (batch_size,), generator=generator).to(device)
    magnitude = (0.02 + 0.28 * torch.rand(batch_size, 1, generator=generator)).to(device)
    sign = torch.where(torch.rand(batch_size, 1, generator=generator) < 0.5, -1.0, 1.0).to(device)
    coherent = batch[:, -1, :] + magnitude * sign
    independent = batch[:, -1, :] + magnitude * torch.randn(
        batch_size, batch.shape[-1], generator=generator
    ).to(device)
    stale = batch[:, -2, :]
    candidate = torch.where((mode == 0)[:, None], coherent, independent)
    candidate = torch.where((mode == 2)[:, None], stale, candidate)
    result[:, -1, :] = torch.where(selected, candidate, batch[:, -1, :])
    return result, bus_index
