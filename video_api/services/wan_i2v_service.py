from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from PIL.Image import Image
import torch

from video_api.service import InferenceService
from video_api.services.pipelines import WanUNet, WanTextEncoder, WanImageEncoder, CFGGuider
from video_api.utils.video_utils import to_img

@dataclass
class WanI2V_CkptConfig:
    unet_path: str
    unet_config_path: str
    textmodel_path: str
    textmodel_config_path: str
    clip_vision_path: str
    clip_vision_config_path: str
    vae_path: str
    vae_config_path: str
    device: str = 'cuda:0'
    
@dataclass
class WanI2V_Input:
    prompt: str
    negative_prompt: str
    image: Image
    width: int
    height: int
    batch_size: int = 1
    length: int = 49
    sampler_name: str = "euler_ancestral_RF"
    steps: int = 20
    step_multiplier: float = 0.75
    shift: float = 5.0
    cfg_1: float = 6.0
    cfg_2: float = 1.0
    seed: int = 42

@dataclass
class WanI2V_Output:
    images: Sequence[Sequence[Image]]

class WanI2V_Service(
    InferenceService[WanI2V_CkptConfig, WanI2V_Input, WanI2V_Output]
):
    
    def __init__(self, config: WanI2V_CkptConfig):
        self.device = config.device
        self.dtype = torch.bfloat16

        self.unet = WanUNet(
            config.unet_path, config.unet_config_path, 
            device=self.device, dtype=self.dtype
        )
        self.text_encoder = WanTextEncoder(
            config.textmodel_path, config.textmodel_config_path, 
            device=self.device
        )
        self.image_encoder = WanImageEncoder(
            config.clip_vision_path, config.clip_vision_config_path, 
            config.vae_path, config.vae_config_path, 
            device=self.device, dtype=self.dtype
        )

        self.guider_1 = CFGGuider(
            self.unet, self.image_encoder.latent_format, 
            debug_output=False, 
            device=self.device, dtype=self.dtype
        )
        self.guider_2 = CFGGuider(
            self.unet, self.image_encoder.latent_format, 
            debug_output=False, 
            device=self.device, dtype=self.dtype
        )

    @torch.no_grad()
    def __call__(self, inp: WanI2V_Input) -> WanI2V_Output:
        positive_cond, negative_cond = self.text_encoder(
            inp.prompt, inp.negative_prompt
        )
        latent, c_concat, clip_fea = self.image_encoder(
            inp.image, inp.width, inp.height, 
            inp.batch_size, inp.length
        )

        input_positive = { 
            'c_concat': c_concat, 
            'c_crossattn': positive_cond, 
            'clip_fea': clip_fea 
        }
        input_negative = { 
            'c_concat': c_concat, 
            'c_crossattn': negative_cond, 
            'clip_fea': clip_fea 
        }

        self.guider_1.set_sigmas(
            inp.steps, inp.shift, 
            low_stop=0, 
            high_stop=int(inp.steps*inp.step_multiplier)
        )
        self.guider_2.set_sigmas(
            inp.steps, inp.shift, 
            low_stop=int(inp.steps*inp.step_multiplier), 
            high_stop=inp.steps
        )

        samples, denoised = self.guider_1.sample(
            inp.sampler_name, latent, 
            input_positive, input_negative, 
            inp.cfg_1, inp.seed
        )
        samples, denoised_2 = self.guider_2.sample(
            inp.sampler_name, denoised, 
            input_positive, input_negative, 
            inp.cfg_2, inp.seed
        )
        result = self.image_encoder.decode(denoised_2)
        result = WanI2V_Output([to_img(x) for x in result])
        return result
    
    def close(self):
        return super().close()