from pathlib import Path
import numpy as np
import json
from PIL import Image
import hashlib
import re
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, set_seed

from video_api.models.ldm.wan import WanModelI2V
from video_api.models.ldm.vae import WanVAE, encode_crop_pixels
from video_api.models.text_encoder.tokenizer import SPieceTokenizer, SDTokenizer
from video_api.models.text_encoder.clip_vision import CLIPVisionModelProjection, clip_preprocess
from video_api.models.text_encoder.wan import SDClipModel, T5
from video_api.sampling.k_diffusion import sampling as k_diffusion_sampling
from video_api.sampling.model_sampling import ModelSamplingAdvanced, simple_scheduler
from video_api.utils.file_utils import load_torch_file
from video_api.utils.latent_utils import Wan21_latent
from video_api.utils.sample_utils import prepare_noise


class WanUNet:

    def __init__(self, unet_path, unet_config_path, device='cuda:0', dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        unet_config = json.loads(Path(unet_config_path).read_bytes())

        sd = load_torch_file(unet_path)
        self.unet = WanModelI2V(device=device, dtype=dtype, **unet_config).eval().to(device)
        m, u = self.unet.load_state_dict(sd, strict=False)
        del sd

        self.model_sampling = ModelSamplingAdvanced()

    def __call__(self, input_x, sigma, c_concat, c_crossattn, clip_fea):
        xc = self.model_sampling.calculate_input(sigma, input_x)
        xc = torch.cat([xc, c_concat], dim=1).to(self.dtype)
        context = c_crossattn.to(self.dtype)
        clip_fea = clip_fea.to(self.dtype)
        
        t = self.model_sampling.timestep(sigma).float()
        output = self.unet(xc, t, context=context, clip_fea=clip_fea)
        output = self.model_sampling.calculate_denoised(sigma, output, input_x)
        return output


class WanTextEncoder:

    def __init__(self, textmodel_path, textmodel_config_path, device='cuda:0'):
        self.device = device

        clip_sd = load_torch_file(textmodel_path)
        self.cond_stage_model = SDClipModel(
            model_class=T5, textmodel_json_config=textmodel_config_path, 
            device=device, dtype=torch.float16, 
            layer="last", layer_idx=None, 
            special_tokens={"end": 1, "pad": 0}, 
            enable_attention_masks=True, zero_out_masked=True)
        m,u = self.cond_stage_model.load_sd(clip_sd)

        tokenizer = SPieceTokenizer(clip_sd.get("spiece_model", None))
        self.tokenizer= SDTokenizer(tokenizer, 
            pad_with_end=False, 
            has_start_token=False, pad_to_max_length=False, 
            max_length=99999999, min_length=512, pad_token=0)
        del clip_sd

    @torch.no_grad()
    def __call__(self, prompt:str, negative_prompt:str):
        positive_tokens = self.tokenizer.tokenize_with_weights(prompt, return_word_ids=False)
        positive_cond, positive_pooled = self.cond_stage_model.encode_token_weights(positive_tokens)[:2]
        negative_tokens = self.tokenizer.tokenize_with_weights(negative_prompt, return_word_ids=False)
        negative_cond, negative_pooled = self.cond_stage_model.encode_token_weights(negative_tokens)[:2]
        return positive_cond, negative_cond
    

class WanImageEncoder:

    def __init__(self, clip_vision_path, clip_vision_config_path, 
                 vae_path, vae_config_path, device='cuda:0', dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        
        clip_vision_config = json.loads(Path(clip_vision_config_path).read_bytes())
        clip_vision_sd = load_torch_file(clip_vision_path)
        self.clip_vision_model = CLIPVisionModelProjection(device=device, dtype=torch.float16, **clip_vision_config).eval().to(device)
        m, u = self.clip_vision_model.load_state_dict(clip_vision_sd, strict=False)
        del clip_vision_sd

        vae_config = json.loads(Path(vae_config_path).read_bytes())
        vae_sd = load_torch_file(vae_path)
        self.first_stage_model = WanVAE(**vae_config).eval().to(device)
        m, u = self.first_stage_model.load_state_dict(vae_sd, strict=False)
        del vae_sd

        self.latent_format = Wan21_latent()

    @torch.no_grad()
    def __call__(self, source, width, height, batch_size, length):
        source_t = torch.from_numpy(np.array(source).astype(np.float32) / 255.0)[None,]
        resized_source = source.resize((width,height), Image.Resampling.LANCZOS)
        resized_source_t = torch.from_numpy(np.array(resized_source).astype(np.float32) / 255.0)[None,]

        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=self.device)
        image = torch.ones((length, height, width, resized_source_t.shape[-1]), dtype=torch.float32) * 0.5
        image[:resized_source_t.shape[0]] = resized_source_t
        mask = torch.ones((batch_size, 1) + latent.shape[2:], dtype=torch.float32)
        mask[:, :, :((resized_source_t.shape[0] - 1) // 4) + 1] = 0.0

        pixel_values = clip_preprocess(source_t.to(self.device), size=224, 
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711], 
            crop=False).float()
        clip_fea = self.clip_vision_model(pixel_values=pixel_values, intermediate_output=-2)[1]

        # concat_latent_image = vae.encode(image[:, :, :, :3])
        pixel_samples = encode_crop_pixels(image[:, :, :, :3])
        pixel_samples = pixel_samples.movedim(-1, 1).movedim(1, 0).unsqueeze(0)
        pixels_in = (pixel_samples * 2.0 - 1.0).to(self.device, dtype=self.dtype)
        
        concat_latent_image = self.first_stage_model.encode(pixels_in).float()
        latent_image = self.latent_format.process_in(concat_latent_image.to(self.device))
        mask = 1.0 - torch.mean(mask, dim=1, keepdim=True).repeat(1, 4, 1, 1, 1).to(self.device)
        c_concat = torch.cat((mask, latent_image), dim=1)

        return latent, c_concat, clip_fea
    
    @torch.no_grad()
    def decode(self, sample):
        result = self.first_stage_model.decode(sample.to(self.dtype)).float()
        result = torch.clamp((result + 1.0) / 2.0, min=0.0, max=1.0).movedim(1,-1)
        return result
    

class CFGGuider:
    def __init__(self, unet, latent_format, debug_output=False, device="cuda:0", dtype=torch.bfloat16):
        self.unet = unet
        self.latent_format = latent_format
        self.debug_output = debug_output
        self.device = device
        self.dtype = dtype

        self.low_stop = 0
        self.high_stop = 100
        self.sigmas = []
        self.positive_conds = None
        self.negative_conds = None
        self.cfg = 1.0

    def set_sigmas(self, steps, shift, low_stop, high_stop):
        self.low_stop = low_stop
        self.high_stop = high_stop

        self.unet.model_sampling.set_parameters(shift=shift)
        self.sigmas = simple_scheduler(steps, self.unet.model_sampling.sigmas).to(self.device)
        self.sigmas = self.sigmas[low_stop:high_stop + 1]

    def __call__(self, x, timestep, *args, **kwargs):
        if self.cfg > 1.0:
            input_x = torch.cat([x] * 2)
            sigma = torch.cat([timestep] * 2)
            c_list = [self.positive_conds, self.negative_conds]
        else:
            input_x = torch.cat([x])
            sigma = torch.cat([timestep])
            c_list = [self.positive_conds]
        c = {k:torch.cat([c[k] for c in c_list]).to(self.device) for k in ['c_concat', 'c_crossattn', 'clip_fea']}
            
        output = self.unet(input_x, sigma, c['c_concat'], c['c_crossattn'], c['clip_fea'])

        if self.cfg > 1.0:
            output = output.chunk(2)
            output = output[1] + (output[0] - output[1]) * self.cfg
        return output
    
    def sample(self, sampler_name, latent, positive_conds, negative_conds, cfg, seed):
        if self.low_stop == self.high_stop:
            return latent, latent
        
        self.positive_conds = positive_conds
        self.negative_conds = negative_conds
        self.cfg = cfg

        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))

        noise = prepare_noise(latent, seed, device=self.device)
        latent_image = self.latent_format.process_in(latent)
        extra_args = {"model_options": {'sample_sigmas' : self.sigmas}, "seed": seed}

        noise = self.unet.model_sampling.noise_scaling(self.sigmas[0], noise, latent_image)
        x0_output = {'x0' : latent}
        def callback(x):
            if self.debug_output:
                print(x)
            x0_output["x0"] = x['denoised']

        samples = sampler_function(self, noise, self.sigmas, extra_args=extra_args, callback=callback, disable=False)
        samples = self.unet.model_sampling.inverse_noise_scaling(self.sigmas[-1], samples)
        
        samples = self.latent_format.process_out(samples.to(torch.float32))
        denoised = self.latent_format.process_out(x0_output['x0'])
        return samples, denoised
    


class FlorenceModelCaptioner:
    def __init__(self, model_path, device='cuda:0'):
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation='sdpa', device_map=device, 
            torch_dtype=torch.float16, trust_remote_code=True).to(device)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.words_to_replace = ['photo', 'image', 'picture', 'illustration', 'drawing']
        self.replacement_word = 'video'

    @torch.no_grad()
    def __call__(self, source, seed):
        hash_object = hashlib.sha256(str(seed).encode('utf-8'))
        hashed_seed = int(hash_object.hexdigest(), 16) % (2**32)
        set_seed(hashed_seed)

        inputs = self.processor(
            text='<MORE_DETAILED_CAPTION>', images=source, return_tensors="pt", do_rescale=False
        ).to(torch.float16).to(self.device)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, do_sample=True, num_beams=5,
        )

        results = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        clean_results = str(results).replace('</s>', '').replace('<s>', '')

        pattern = r'\b(?:' + '|'.join(map(re.escape, self.words_to_replace)) + r')\b'
        clean_results = re.sub(pattern, self.replacement_word, clean_results, flags=re.IGNORECASE)
        return clean_results