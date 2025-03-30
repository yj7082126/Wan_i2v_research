
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from video_api.utils.attention_utils import attention_sage
from video_api.utils.sample_utils import cast_to

from video_api.gguf_loader.ops import ops

#%%
def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad

    return F.pad(img, pad, mode=padding_mode)

#%%
def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)

def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

#%%
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x):
        return F.rms_norm(x, self.weight.shape, weight=self.weight, eps=self.eps)
    
#%%
class WanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, device=None, dtype=None):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        
        self.q = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.k = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.v = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.o = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.norm_q = RMSNorm(dim, eps=eps, device=device, dtype=dtype) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps, device=device, dtype=dtype) if qk_norm else nn.Identity()

    def forward(self, x, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n * d)
        q, k = apply_rope(q, k, freqs)

        x = attention_sage(
            q.view(b, s, n * d), 
            k.view(b, s, n * d), 
            v, 
            heads=self.num_heads,
        )
        x = self.o(x)
        return x
    
class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, device=None, dtype=None):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, device, dtype)

        self.k_img = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.v_img = ops.Linear(dim, dim, device=device, dtype=dtype)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = RMSNorm(dim, eps=eps, device=device, dtype=dtype) if qk_norm else nn.Identity()

    def forward(self, x, context):
        context_img = context[:, :257]
        context = context[:, 257:]

        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        x = attention_sage(q, k, v, heads=self.num_heads)

        k_img = self.norm_k_img(self.k_img(context_img))
        v_img = self.v_img(context_img)
        img_x = attention_sage(q, k_img, v_img, heads=self.num_heads)

        x = x + img_x
        x = self.o(x)
        return x
    
#%%
class WanI2VAttentionBlock(nn.Module):

    def __init__(self, dim, ffn_dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=False, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = ops.LayerNorm(dim, eps, elementwise_affine=False, device=device, dtype=dtype)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, device=device, dtype=dtype)
        self.norm3 = ops.LayerNorm(dim, eps, elementwise_affine=True, device=device, dtype=dtype) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, device=device, dtype=dtype)
        self.norm2 = ops.LayerNorm(dim, eps, elementwise_affine=False, device=device, dtype=dtype)
        self.ffn = nn.Sequential(
            ops.Linear(dim, ffn_dim, device=device, dtype=dtype), 
            nn.GELU(approximate='tanh'),
            ops.Linear(ffn_dim, dim, device=device, dtype=dtype))

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 6, dim))

    def forward(self, x, e, freqs, context):
        e = (cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        # self-attention
        y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], freqs)
        x = x + y * e[2]

        # cross-attention & ffn
        x = x + self.cross_attn(self.norm3(x), context)
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x
    
#%%
class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = ops.LayerNorm(dim, eps, elementwise_affine=False, device=device, dtype=dtype)
        self.head = ops.Linear(dim, out_dim, device=device, dtype=dtype)

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 2, dim))

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        e = (cast_to(self.modulation, dtype=x.dtype, device=x.device) + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(nn.Module):

    def __init__(self, in_dim, out_dim, device=None, dtype=None):
        super().__init__()

        self.proj = nn.Sequential(
            ops.LayerNorm(in_dim, device=device, dtype=dtype),
            ops.Linear(in_dim, in_dim, device=device, dtype=dtype),
            nn.GELU(), 
            ops.Linear(in_dim, out_dim, device=device, dtype=dtype),
            ops.LayerNorm(out_dim, device=device, dtype=dtype)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens
    
#%%
class WanModelI2V(nn.Module):
    r"""
    Wan diffusion backbone supporting image-to-video.
    """

    def __init__(self,
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 device=None, dtype=None
                 ):
        super().__init__()

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = ops.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size, 
            device=device, dtype=torch.float32)
        self.text_embedding = nn.Sequential(
            ops.Linear(text_dim, dim, device=device, dtype=dtype), 
            nn.GELU(approximate='tanh'),
            ops.Linear(dim, dim, device=device, dtype=dtype)
        )

        self.time_embedding = nn.Sequential(
            ops.Linear(freq_dim, dim, device=device, dtype=dtype), 
            nn.SiLU(), 
            ops.Linear(dim, dim, device=device, dtype=dtype)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), 
            ops.Linear(dim, dim * 6, device=device, dtype=dtype)
        )

        # blocks
        self.blocks = nn.ModuleList([
            WanI2VAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps, device=device, dtype=dtype)

        d = dim // num_heads
        self.rope_embedder = EmbedND(dim=d, theta=10000.0, axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)])

        self.img_emb = MLPProj(1280, dim, device=device, dtype=dtype)

    def forward_orig(self, x, t, context, clip_fea=None, freqs=None):

        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        if clip_fea is not None and self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        for i, block in enumerate(self.blocks):
            x = block(x, e=e0, freqs=freqs, context=context)

        # head
        x = self.head(x, e)
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def forward(self, x, timestep, context, clip_fea=None, **kwargs):
        bs, c, t, h, w = x.shape
        x = pad_to_patch_size(x, self.patch_size)
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 0] = img_ids[..., 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
        img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        return self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs)[:, :, :t, :h, :w]

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        u = x
        b = u.shape[0]
        u = u[:, :math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        u = torch.einsum('bfhwpqrc->bcfphqwr', u)
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return u
