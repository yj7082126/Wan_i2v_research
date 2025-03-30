import torch

def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)
#%%

def simple_scheduler(steps, sigmas):
    ss = len(sigmas) / steps
    sigs = [float(sigmas[-(1+int(x*ss))]) for x in range(steps)]
    sigs += [0.0]
    sigmas = torch.FloatTensor(sigs)
    sigmas = sigmas[-(steps + 1):]
    return sigmas

#%%
class EPS:
    def calculate_input(self, sigma, noise):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        if max_denoise:
            noise = noise * torch.sqrt(1.0 + sigma ** 2.0)
        else:
            noise = noise * sigma

        noise += latent_image
        return noise

    def inverse_noise_scaling(self, sigma, latent):
        return latent

class V_PREDICTION(EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input * self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2) - model_output * sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

class EDM(V_PREDICTION):
    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input * self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2) + model_output * sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

class CONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (latent.ndim - 1))
        return latent / (1.0 - sigma)

#%% 
class ModelSamplingDiscreteFlow(torch.nn.Module):
    def __init__(self, shift=8.0, multiplier=1000):
        super().__init__()
        self.set_parameters(shift=shift, multiplier=multiplier)

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.shift = shift
        self.multiplier = multiplier
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier)
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return time_snr_shift(self.shift, 1.0 - percent)
    
class ModelSamplingAdvanced(ModelSamplingDiscreteFlow, CONST):
    pass