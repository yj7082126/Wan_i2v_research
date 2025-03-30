import os
import json
import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F

from video_api.gguf_loader.ops import scaled_ops
from video_api.utils.attention_utils import attention_pytorch
from video_api.utils.sample_utils import cast_to

def gen_empty_tokens(special_tokens, length):
    start_token = special_tokens.get("start", None)
    end_token = special_tokens.get("end", None)
    pad_token = special_tokens.get("pad")
    output = []
    if start_token is not None:
        output.append(start_token)
    if end_token is not None:
        output.append(end_token)
    output += [pad_token] * (length - len(output))
    return output

class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        to_encode = list()
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0:
            if hasattr(self, "gen_empty_tokens"):
                to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))
            else:
                to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

        o = self.encode(to_encode)
        out, pooled = o[:2]

        if pooled is not None:
            first_pooled = pooled[0:1]
        else:
            first_pooled = pooled

        output = []
        for k in range(0, sections):
            z = out[k:k+1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        if (len(output) == 0):
            r = (out[-1:], first_pooled)
        else:
            r = (torch.cat(output, dim=-2), first_pooled)

        if len(o) > 2:
            extra = {}
            for k in o[2]:
                v = o[2][k]
                if k == "attention_mask":
                    v = v[:sections].flatten().unsqueeze(dim=0)
                extra[k] = v

            r = r + (extra,)
        return r

#%% T5

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return cast_to(self.weight, x.dtype, x.device, non_blocking=False) * x
    
#%%
class T5DenseActDense(nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.wi = scaled_ops.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = scaled_ops.Linear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.act = lambda a: F.gelu(a, approximate='tanh')

    def forward(self, x):
        x = self.act(self.wi(x))
        x = self.wo(x)
        return x
    
class T5DenseGatedActDense(nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.wi_0 = scaled_ops.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wi_1 = scaled_ops.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = scaled_ops.Linear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.act = lambda a: F.gelu(a, approximate='tanh')

    def forward(self, x):
        hidden_gelu = self.act(self.wi_0(x))
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        x = self.wo(x)
        return x
    
class T5LayerFF(nn.Module):
    def __init__(self, model_dim, ff_dim, gated_act, dtype, device):
        super().__init__()
        if gated_act:
            self.DenseReluDense = T5DenseGatedActDense(model_dim, ff_dim, dtype, device)
        else:
            self.DenseReluDense = T5DenseActDense(model_dim, ff_dim, dtype, device)

        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        x += forwarded_states
        return x
    
#%%
class T5Attention(nn.Module):
    def __init__(self, model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = scaled_ops.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.k = scaled_ops.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.v = scaled_ops.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.o = scaled_ops.Linear(inner_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.num_heads = num_heads

        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = scaled_ops.Embedding(self.relative_attention_num_buckets, self.num_heads, device=device, dtype=dtype)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device, dtype):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket, out_dtype=dtype)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self, x, mask=None, past_bias=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], x.device, x.dtype)

        if past_bias is not None:
            if mask is not None:
                mask = mask + past_bias
            else:
                mask = past_bias

        out = attention_pytorch(q, k * ((k.shape[-1] / self.num_heads) ** 0.5), v, self.num_heads, mask)
        return self.o(out), past_bias

class T5LayerSelfAttention(nn.Module):
    def __init__(self, model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.SelfAttention = T5Attention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x, mask=None, past_bias=None):
        output, past_bias = self.SelfAttention(self.layer_norm(x), mask=mask, past_bias=past_bias)
        x += output
        return x, past_bias
    
#%%
class T5Block(nn.Module):
    def __init__(self, model_dim, inner_dim, ff_dim, gated_act, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device))
        self.layer.append(T5LayerFF(model_dim, ff_dim, gated_act, dtype, device))

    def forward(self, x, mask=None, past_bias=None):
        x, past_bias = self.layer[0](x, mask, past_bias)
        x = self.layer[-1](x)
        return x, past_bias

class T5Stack(nn.Module):
    def __init__(self, num_layers, model_dim, inner_dim, ff_dim, gated_act, num_heads, relative_attention, dtype, device):
        super().__init__()

        self.block = nn.ModuleList(
            [T5Block(model_dim, inner_dim, ff_dim, gated_act, num_heads, 
                     relative_attention_bias=((not relative_attention) or (i == 0)), 
                     dtype=dtype, device=device) for i in range(num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x, attention_mask=None, intermediate_output=None, final_layer_norm_intermediate=True):
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), -torch.finfo(x.dtype).max)

        intermediate = None
        past_bias = None

        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.block) + intermediate_output

        for i, l in enumerate(self.block):
            x, past_bias = l(x, mask, past_bias)
            if i == intermediate_output:
                intermediate = x.clone()
        x = self.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)
        return x, intermediate

class T5(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict["num_layers"]
        model_dim = config_dict["d_model"]
        inner_dim = config_dict["d_kv"] * config_dict["num_heads"]

        self.encoder = T5Stack(
            self.num_layers, model_dim, inner_dim, config_dict["d_ff"], 
            config_dict["is_gated_act"], config_dict["num_heads"], config_dict["model_type"] != "umt5", dtype, device)
        self.dtype = dtype
        self.shared = scaled_ops.Embedding(config_dict["vocab_size"], model_dim, device=device, dtype=dtype)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, embeddings):
        self.shared = embeddings

    def forward(self, input_ids, attention_mask, embeds=None, num_tokens=None, **kwargs):
        if input_ids is None:
            x = embeds
        else:
            x = self.shared(input_ids, out_dtype=kwargs.get("dtype", torch.float32))
        if self.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.nan_to_num(x) #Fix for fp8 T5 base
        return self.encoder(x, attention_mask=attention_mask, **kwargs)


#%%
class SDClipModel(nn.Module, ClipTokenWeightEncoder):
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, device="cpu", max_length=77,
                 freeze=True, layer="last", layer_idx=None, textmodel_json_config=None, 
                 dtype=None, model_class=T5,
                 special_tokens={"start": 49406, "end": 49407, "pad": 49407}, 
                 layer_norm_hidden_state=True, enable_attention_masks=False, zero_out_masked=False,
                 return_projected_pooled=True, return_attention_masks=False):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS

        if isinstance(textmodel_json_config, dict):
            config = textmodel_json_config
        else:
            with open(textmodel_json_config) as f:
                config = json.load(f)

        self.transformer = model_class(config, dtype, device)
        self.transformer.scaled_fp8 = nn.Parameter(torch.tensor([], dtype=torch.float8_e4m3fn))

        self.num_layers = self.transformer.num_layers

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens

        self.logit_scale = nn.Parameter(torch.tensor(4.6055))
        self.enable_attention_masks = enable_attention_masks
        self.zero_out_masked = zero_out_masked

        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        self.return_attention_masks = return_attention_masks

        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_options(self):
        self.layer = self.options_default[0]
        self.layer_idx = self.options_default[1]
        self.return_projected_pooled = self.options_default[2]

    def process_tokens(self, tokens, device):
        end_token = self.special_tokens.get("end", None)
        if end_token is None:
            cmp_token = self.special_tokens.get("pad", -1)
        else:
            cmp_token = end_token

        embeds_out = []
        attention_masks = []
        num_tokens = []

        for x in tokens:
            attention_mask = []
            tokens_temp = []
            other_embeds = []
            eos = False
            index = 0
            for y in x:
                if isinstance(y, numbers.Integral):
                    if eos:
                        attention_mask.append(0)
                    else:
                        attention_mask.append(1)
                    token = int(y)
                    tokens_temp += [token]
                    if not eos and token == cmp_token:
                        if end_token is None:
                            attention_mask[-1] = 0
                        eos = True
                else:
                    other_embeds.append((index, y))
                index += 1

            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.transformer.get_input_embeddings()(tokens_embed, out_dtype=torch.float32)
            index = 0
            pad_extra = 0
            for o in other_embeds:
                emb = o[1]
                if torch.is_tensor(emb):
                    emb = {"type": "embedding", "data": emb}

                emb_type = emb.get("type", None)
                if emb_type == "embedding":
                    emb = emb.get("data", None)
                else:
                    if hasattr(self.transformer, "preprocess_embed"):
                        emb = self.transformer.preprocess_embed(emb, device=device)
                    else:
                        emb = None

                if emb is None:
                    index += -1
                    continue

                ind = index + o[0]
                emb = emb.view(1, -1, emb.shape[-1]).to(device=device, dtype=torch.float32)
                emb_shape = emb.shape[1]
                if emb.shape[-1] == tokens_embed.shape[-1]:
                    tokens_embed = torch.cat([tokens_embed[:, :ind], emb, tokens_embed[:, ind:]], dim=1)
                    attention_mask = attention_mask[:ind] + [1] * emb_shape + attention_mask[ind:]
                    index += emb_shape - 1
                else:
                    index += -1
                    pad_extra += emb_shape

            if pad_extra > 0:
                padd_embed = self.transformer.get_input_embeddings()(torch.tensor([[self.special_tokens["pad"]] * pad_extra], device=device, dtype=torch.long), out_dtype=torch.float32)
                tokens_embed = torch.cat([tokens_embed, padd_embed], dim=1)
                attention_mask = attention_mask + [0] * pad_extra

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens

    def forward(self, tokens):
        device = self.transformer.get_input_embeddings().weight.device
        embeds, attention_mask, num_tokens = self.process_tokens(tokens, device)

        attention_mask_model = None
        if self.enable_attention_masks:
            attention_mask_model = attention_mask

        outputs = self.transformer(None, attention_mask_model, embeds=embeds, num_tokens=num_tokens, intermediate_output=self.layer_idx, final_layer_norm_intermediate=self.layer_norm_hidden_state)

        if self.layer == "last":
            z = outputs[0].float()
        else:
            z = outputs[1].float()

        if self.zero_out_masked:
            z *= attention_mask.unsqueeze(-1).float()

        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        extra = {}
        if self.return_attention_masks:
            extra["attention_mask"] = attention_mask

        if len(extra) > 0:
            return z, pooled_output, extra

        return z, pooled_output

    def encode(self, tokens):
        return self(tokens)

    def load_sd(self, sd):
        return self.transformer.load_state_dict(sd, strict=False)
    
#%%
class UMT5XXlModel(SDClipModel):
    def __init__(self, textmodel_json_config, device="cpu", layer="last", layer_idx=None, dtype=None):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, 
                         special_tokens={"end": 1, "pad": 0}, model_class=T5, 
                         enable_attention_masks=True, zero_out_masked=True)