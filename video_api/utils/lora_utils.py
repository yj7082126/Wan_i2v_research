import torch
from video_api.gguf_loader.comfy_ops import cast_to

def fix_musubi(lora, prefix='lora_unet_'):
    musubi = False
    lora_alphas = {}
    for key, value in lora.items():
        if key.startswith(prefix):
            lora_name = key.split(".", 1)[0]
            if lora_name not in lora_alphas and "alpha" in key:
                lora_alphas[lora_name] = value
                musubi = True

    if musubi:
        converted_lora = {}
        for key, weight in lora.items():
            if key.startswith(prefix):
                if "alpha" in key:
                    continue
                lora_name = key.split(".", 1)[0]
                module_name = lora_name[len(prefix):]  # remove "lora_unet_"
                module_name = module_name.replace("_", ".")  # replace "_" with "."
                module_name = module_name.replace("double.blocks.", "double_blocks.")  # fix double blocks
                module_name = module_name.replace("single.blocks.", "single_blocks.")  # fix single blocks
                module_name = module_name.replace("img.", "img_")  # fix img
                module_name = module_name.replace("txt.", "txt_")  # fix txt
                module_name = module_name.replace("attn.", "attn_")  # fix attn

                if "lora_down" in key:
                    new_key = f"diffusion_model.{module_name}.lora_A.weight"
                    dim = weight.shape[0]
                elif "lora_up" in key:
                    new_key = f"diffusion_model.{module_name}.lora_B.weight"
                    dim = weight.shape[1]
                else:
                    continue

                if lora_name in lora_alphas:
                    scale = lora_alphas[lora_name] / dim
                    scale = scale.sqrt()
                    weight = weight * scale
                converted_lora[new_key] = weight
        return converted_lora
    else:
        return lora

def model_lora_keys_unet_wan(model, key_map={}):
    sdk = model.state_dict().keys()

    for k in sdk:
        if k.endswith(".weight"):
            key_lora = k[:-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = k
            key_map["{}".format(k[:-len(".weight")])] = k 
        else:
            key_map["{}".format(k)] = k 
    return key_map

def load_lora(sd, to_load, log_missing=True):
    lora = {}
    for k, v in sd.items():
        if k.startswith("diffusion_model."):
            k_to = k.replace("diffusion_model.", "")
            lora[k_to] = v

    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        regular_lora = "{}.lora_up.weight".format(x)
        diffusers_lora = "{}_lora.up.weight".format(x)
        diffusers2_lora = "{}.lora_B.weight".format(x)
        diffusers3_lora = "{}.lora.up.weight".format(x)
        mochi_lora = "{}.lora_B".format(x)
        transformers_lora = "{}.lora_linear_layer.up.weight".format(x)
        A_name = None

        if regular_lora in lora.keys():
            A_name = regular_lora
            B_name = "{}.lora_down.weight".format(x)
            mid_name = "{}.lora_mid.weight".format(x)
        elif diffusers_lora in lora.keys():
            A_name = diffusers_lora
            B_name = "{}_lora.down.weight".format(x)
            mid_name = None
        elif diffusers2_lora in lora.keys():
            A_name = diffusers2_lora
            B_name = "{}.lora_A.weight".format(x)
            mid_name = None
        elif diffusers3_lora in lora.keys():
            A_name = diffusers3_lora
            B_name = "{}.lora.down.weight".format(x)
            mid_name = None
        elif mochi_lora in lora.keys():
            A_name = mochi_lora
            B_name = "{}.lora_A".format(x)
            mid_name = None
        elif transformers_lora in lora.keys():
            A_name = transformers_lora
            B_name ="{}.lora_linear_layer.down.weight".format(x)
            mid_name = None

        if A_name is not None:
            mid = None
            if mid_name is not None and mid_name in lora.keys():
                mid = lora[mid_name]
                loaded_keys.add(mid_name)
            patch_dict[to_load[x]] = (lora[A_name], lora[B_name])
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)

    if log_missing:
        for x in lora.keys():
            if x not in loaded_keys:
                print("lora key not loaded: {}".format(x))

    return patch_dict

@torch.no_grad()
def calculate_weight(patch, weight, key, intermediate_dtype=torch.float32, original_weights=None):
    for p in patch:
        strength = p[0]
        v = p[1]

        mat1 = cast_to(v[0], intermediate_dtype, weight.device)
        mat2 = cast_to(v[1], intermediate_dtype, weight.device)
        
        lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(weight.shape)
        lora_diff = (strength * lora_diff).type(weight.dtype)
        # print(strength, weight.shape, weight.dtype, lora_diff.shape, lora_diff.dtype)
        weight = torch.add(weight, lora_diff)
    return weight