import torch
import torch.nn as nn
import torch.nn.functional as F

from video_api.gguf_loader.ops import ops
from video_api.utils.attention_utils import attention_pytorch
from video_api.utils.sample_utils import cast_to

def clip_preprocess(image, size=224, 
                    mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711], crop=True):
    mean = torch.tensor(mean, device=image.device, dtype=image.dtype)
    std = torch.tensor(std, device=image.device, dtype=image.dtype)
    image = image.movedim(-1, 1)
    if not (image.shape[2] == size and image.shape[3] == size):
        if crop:
            scale = (size / min(image.shape[2], image.shape[3]))
            scale_size = (round(scale * image.shape[2]), round(scale * image.shape[3]))
        else:
            scale_size = (size, size)

        image = F.interpolate(image, size=scale_size, mode="bicubic", antialias=True)
        h = (image.shape[2] - size)//2
        w = (image.shape[3] - size)//2
        image = image[:,:,h:h+size,w:w+size]
    image = torch.clip((255. * image), 0, 255).round() / 255.0
    return (image - mean.view([3,1,1])) / std.view([3,1,1])


class CLIPMLP(nn.Module):
    def __init__(self, embed_dim, intermediate_size, dtype, device):
        super().__init__()
        self.fc1 = ops.Linear(embed_dim, intermediate_size, bias=True, dtype=dtype, device=device)
        self.fc2 = ops.Linear(intermediate_size, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device):
        super().__init__()

        self.heads = heads
        self.q_proj = ops.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = ops.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = ops.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.out_proj = ops.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        out = attention_pytorch(q, k, v, self.heads, mask)
        return self.out_proj(out)



class CLIPLayer(nn.Module):
    def __init__(self, embed_dim, heads, intermediate_size, dtype, device):
        super().__init__()
        self.layer_norm1 = ops.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device)
        self.layer_norm2 = ops.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.mlp = CLIPMLP(embed_dim, intermediate_size, dtype, device)

    def forward(self, x, mask=None):
        x += self.self_attn(self.layer_norm1(x), mask)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, heads, intermediate_size, dtype, device):
        super().__init__()
        self.layers = nn.ModuleList([CLIPLayer(embed_dim, heads, intermediate_size, dtype, device) for i in range(num_layers)])

    def forward(self, x, mask=None, intermediate_output=None):
        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate
    
class CLIPVisionEmbeddings(torch.nn.Module):
    def __init__(self, embed_dim, num_channels=3, patch_size=14, image_size=224, dtype=None, device=None):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2 + 1
        self.class_embedding = nn.Parameter(torch.empty(embed_dim, dtype=dtype, device=device))
        patch_bias = False

        self.patch_embedding = ops.Conv2d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=patch_bias,
            dtype=dtype,
            device=device
        )

        self.position_embedding = ops.Embedding(
            num_patches, embed_dim, dtype=dtype, device=device)

    def forward(self, pixel_values):
        embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        if self.class_embedding is not None:
            
            embeds = torch.cat([cast_to(self.class_embedding, embeds.dtype, embeds.device).expand(pixel_values.shape[0], 1, -1), embeds], dim=1)
        return embeds + cast_to(self.position_embedding.weight, embeds.dtype, embeds.device)


class CLIPVision(nn.Module):
    def __init__(self, num_hidden_layers=32, 
                    hidden_size=1280,
                    num_attention_heads=16,
                    intermediate_size=5120, 
                    num_channels=3, patch_size=14, image_size=224,
                    device=None, dtype=None):
        super().__init__()

        self.embeddings = CLIPVisionEmbeddings(
            hidden_size, num_channels, patch_size, image_size, 
            dtype=dtype, device=device)
        self.pre_layrnorm = ops.LayerNorm(hidden_size)

        self.encoder = CLIPEncoder(
            num_hidden_layers, hidden_size, num_attention_heads, intermediate_size, 
            dtype=dtype, device=device)
        self.post_layernorm = ops.LayerNorm(hidden_size)

    def forward(self, pixel_values, attention_mask=None, intermediate_output=None):
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)
        #TODO: attention_mask?
        x, i = self.encoder(x, mask=None, intermediate_output=intermediate_output)
        pooled_output = self.post_layernorm(x[:, 0, :])
        return x, i, pooled_output

class CLIPVisionModelProjection(nn.Module):
    def __init__(self, num_hidden_layers=32, 
                    hidden_size=1280,
                    num_attention_heads=16,
                    intermediate_size=5120, 
                    num_channels=3, patch_size=14, image_size=224, 
                    projection_dim=1024,
                    device=None, dtype=None):
        super().__init__()
        self.vision_model = CLIPVision(
            num_hidden_layers=num_hidden_layers, 
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size, 
            num_channels=num_channels, 
            patch_size=patch_size, 
            image_size=image_size, 
            device=device, dtype=dtype)

        self.visual_projection = ops.Linear(hidden_size, projection_dim, bias=False)


    def forward(self, *args, **kwargs):
        x = self.vision_model(*args, **kwargs)
        out = self.visual_projection(x[2])

        return (x[0], x[1], out)
