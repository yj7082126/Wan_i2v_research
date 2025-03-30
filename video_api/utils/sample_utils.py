import torch

def cast_to(weight, dtype=None, device=None, non_blocking=False):
    if weight.device == device:
        if weight.dtype == dtype:
            return weight
        return weight.to(dtype=dtype, copy=False)
    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight, non_blocking=non_blocking)
    return r

def prepare_noise(latent_image, seed, device='cpu'):
    generator = torch.manual_seed(seed)
    return torch.randn(
        latent_image.size(), 
        dtype=latent_image.dtype, 
        layout=latent_image.layout, 
        generator=generator, 
        device='cpu').to(device)