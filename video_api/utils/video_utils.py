import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
import imageio

def common_upscale(samples, width, height, upscale_method, crop):
    orig_shape = tuple(samples.shape)
    if len(orig_shape) > 4:
        samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
        samples = samples.movedim(2, 1)
        samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
    if crop == "center":
        old_width = samples.shape[-1]
        old_height = samples.shape[-2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    else:
        s = samples

    out = F.interpolate(s, size=(height, width), mode=upscale_method)
    if len(orig_shape) == 4:
        return out

    out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
    return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))


#192, 64, 64, 64
def decode_tiled(vae, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=64):
    if tile_size < overlap * 4:
        overlap = tile_size // 4
    if temporal_size < temporal_overlap * 2:
        temporal_overlap = temporal_overlap // 2
    temporal_compression = vae.temporal_compression_decode()
    if temporal_compression is not None:
        temporal_size = max(2, temporal_size // temporal_compression)
        temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
    else:
        temporal_size = None
        temporal_overlap = None

    compression = vae.spacial_compression_decode()
    images = vae.decode_tiled(samples, tile_x=tile_size // compression, tile_y=tile_size // compression, overlap=overlap // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
    if len(images.shape) == 5: #Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images

def to_img(tensor, normalize=False, value_range=(-1,1)):
    assert len(tensor.shape) == 4
    tensor = tensor[None]
    if normalize:
        tensor = tensor.clap(value_range[0], value_range[1])
    tmp = [
        torchvision.utils.make_grid(u, nrow=1, normalize=normalize, value_range=value_range) 
        for u in tensor.unbind(1)
    ]
    tmp = (torch.stack(tmp, dim=0) * 255.).type(torch.uint8).cpu()
    tmp = [Image.fromarray(x) for x in tmp.numpy()]
    return tmp

def save_video(tensor, save_path=None, fps=16, normalize=False, value_range=(-1,1)):
    assert len(tensor.shape) == 4
    tensor = tensor[None]
    if normalize:
        tensor = tensor.clap(value_range[0], value_range[1])
    tmp = [
        torchvision.utils.make_grid(u, nrow=1, normalize=normalize, value_range=value_range) 
        for u in tensor.unbind(1)
    ]
    tmp = (torch.stack(tmp, dim=0) * 255.).type(torch.uint8).cpu()

    writer = imageio.get_writer(save_path, fps=fps, codec='libx264', quality=8)
    for frame in tmp.numpy():
        writer.append_data(frame)
    writer.close()