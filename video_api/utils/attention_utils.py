import torch
import torch.nn.functional as F
import xformers
import xformers.ops
from sageattention import sageattn

@torch.compiler.disable()
def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
        tensor_layout="HND"
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head),
            (q, k, v),
        )
        tensor_layout="NHD"
    
    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    out = sageattn(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)

    if tensor_layout == "HND":
        if not skip_output_reshape:
            out = (
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            )
    else:
        if skip_output_reshape:
            out = out.transpose(1, 2)
        else:
            out = out.reshape(b, -1, heads * dim_head)
    return out

@torch.compiler.disable()
def attention_xformers(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
    b = q.shape[0]
    dim_head = q.shape[-1]

    if skip_reshape:
        # b h k d -> b k h d
        q, k, v = map(
            lambda t: t.permute(0, 2, 1, 3),
            (q, k, v),
        )
    # actually do the reshaping
    else:
        dim_head //= heads
        q, k, v = map(
            lambda t: t.reshape(b, -1, heads, dim_head),
            (q, k, v),
        )

    if mask is not None:
        # add a singleton batch dimension
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a singleton heads dimension
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        # pad to a multiple of 8
        pad = 8 - mask.shape[-1] % 8
        # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
        # but when using separated heads, the shape has to be (B, H, Nq, Nk)
        # in flux, this matrix ends up being over 1GB
        # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
        mask_out = torch.empty([mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device)

        mask_out[..., :mask.shape[-1]] = mask
        # doesn't this remove the padding again??
        mask = mask_out[..., :mask.shape[-1]]
        mask = mask.expand(b, heads, -1, -1)

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

    if skip_output_reshape:
        out = out.permute(0, 2, 1, 3)
    else:
        out = (
            out.reshape(b, -1, heads * dim_head)
        )

    return out

@torch.compiler.disable()
def attention_xformers_vae(q, k, v):
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
    out = out.transpose(1, 2).reshape(orig_shape)
    return out

SDP_BATCH_LIMIT = 2**15

def attention_pytorch(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(
        lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
        (q, k, v),
    )

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    if SDP_BATCH_LIMIT >= b:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    else:
        out = torch.empty((b, q.shape[2], heads * dim_head), dtype=q.dtype, layout=q.layout, device=q.device)
        for i in range(0, b, SDP_BATCH_LIMIT):
            m = mask
            if mask is not None:
                if mask.shape[0] > 1:
                    m = mask[i : i + SDP_BATCH_LIMIT]

            out[i : i + SDP_BATCH_LIMIT] = F.scaled_dot_product_attention(
                q[i : i + SDP_BATCH_LIMIT],
                k[i : i + SDP_BATCH_LIMIT],
                v[i : i + SDP_BATCH_LIMIT],
                attn_mask=m,
                dropout_p=0.0, is_causal=False
            ).transpose(1, 2).reshape(-1, q.shape[2], heads * dim_head)
    return out
