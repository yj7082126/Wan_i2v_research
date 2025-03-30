import torch


def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False):
    if device is None or weight.device == device:
        if not copy:
            if dtype is None or weight.dtype == dtype:
                return weight
        return weight.to(dtype=dtype, copy=copy)

    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight, non_blocking=non_blocking)
    return r

def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    bias = None
    non_blocking = True
    if s.bias is not None:
        has_function = len(s.bias_function) > 0
        bias = cast_to(s.bias, bias_dtype, device, non_blocking=non_blocking, copy=has_function)
        if has_function:
            for f in s.bias_function:
                bias = f(bias)

    has_function = len(s.weight_function) > 0
    weight = cast_to(s.weight, dtype, device, non_blocking=non_blocking, copy=has_function)
    if has_function:
        for f in s.weight_function:
            weight = f(weight)
    return weight, bias

class CastWeightBiasOp:
    comfy_cast_weights = False
    weight_function = []
    bias_function = []

class disable_weight_init:
    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
                bias = None
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 2
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose2d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 1
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose1d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Embedding(torch.nn.Embedding, CastWeightBiasOp):
        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            weight, bias = cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).to(dtype=output_dtype)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                if "out_dtype" in kwargs:
                    kwargs.pop("out_dtype")
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        comfy_cast_weights = True

    class Conv1d(disable_weight_init.Conv1d):
        comfy_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        comfy_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        comfy_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        comfy_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        comfy_cast_weights = True

    class ConvTranspose2d(disable_weight_init.ConvTranspose2d):
        comfy_cast_weights = True

    class ConvTranspose1d(disable_weight_init.ConvTranspose1d):
        comfy_cast_weights = True

    class Embedding(disable_weight_init.Embedding):
        comfy_cast_weights = True

#%% comfy/float.py

def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )

    mantissa_scaled += torch.rand(mantissa_scaled.size(), dtype=mantissa_scaled.dtype, layout=mantissa_scaled.layout, device=mantissa_scaled.device, generator=generator)
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)


def manual_stochastic_round_to_float8(x, dtype, generator=None):
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    # Combine exponent calculation and clamping
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )

    # Combine mantissa calculation and rounding
    normal_mask = ~(exponent == 0)

    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x
    )

    inf = torch.finfo(dtype)
    torch.clamp(sign, min=inf.min, max=inf.max, out=sign)
    return sign

def stochastic_rounding(value, dtype, seed=0):
    if dtype == torch.float32:
        return value.to(dtype=torch.float32)
    if dtype == torch.float16:
        return value.to(dtype=torch.float16)
    if dtype == torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        output = torch.empty_like(value, dtype=dtype)
        num_slices = max(1, (value.numel() / (4096 * 4096)))
        slice_size = max(1, round(value.shape[0] / num_slices))
        for i in range(0, value.shape[0], slice_size):
            output[i:i+slice_size].copy_(manual_stochastic_round_to_float8(value[i:i+slice_size], dtype, generator=generator))
        return output

    return value.to(dtype=dtype)