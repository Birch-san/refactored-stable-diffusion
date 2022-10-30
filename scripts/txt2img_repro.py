import argparse, sys, os
sys.path.append(os.getcwd())

import torch
import numpy as np
from PIL import Image
from einops import rearrange
import yaml

from pytorch_lightning import seed_everything
from torch import autocast, Tensor, nn
from contextlib import nullcontext
from typing import Protocol, Iterable, Optional

from sd.util import load_model_from_config
from sd.samplers.ddpm import DDPMSampler
from sd.samplers.ddim import DDIMSampler
from sd.samplers.plms import PLMSSampler
from sd.modules.device import get_device_type
from sd.models.diffusion import StableDiffusion

from k_diffusion.sampling import sample_heun, sample_lms, get_sigmas_karras, append_zero
from k_diffusion.external import DiscreteEpsDDPMDenoiser
from k_diffusion.utils import append_dims

class KCFGDenoiser(DiscreteEpsDDPMDenoiser):
    inner_model: StableDiffusion
    def __init__(self, model: StableDiffusion):
        super().__init__(model, model.schedule.alphas_cumprod, quantize=True)
    
    def get_eps(self, *args, **kwargs):
        return self.inner_model.apply_model(*args, **kwargs)

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        uncond: Tensor,
        cond: Tensor, 
        cond_scale: float,
        **kwargs
    ) -> Tensor:
        if uncond is None or cond_scale == 1.0:
            return super().forward(input=x, sigma=sigma, cond=cond)
        cond_in = torch.cat([uncond, cond])
        del uncond, cond
        x_in = x.expand(cond_in.size(dim=0), -1, -1, -1)
        del x
        uncond, cond = super().forward(input=x_in, sigma=sigma, cond=cond_in).chunk(cond_in.size(dim=0))
        del x_in, cond_in
        return uncond + (cond - uncond) * cond_scale

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--prompt',
        type=str,
        nargs='?',
        default='a painting of a virus monster playing guitar',
        help='the prompt to render'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        nargs='?',
        help='dir to write results to',
        default='outputs/txt2img-samples'
    )
    parser.add_argument(
        '--skip_grid',
        action='store_true',
        help='do not save a grid, only individual samples. Helpful when evaluating lots of samples',
    )
    parser.add_argument(
        '--skip_save',
        action='store_true',
        help='do not save indiviual samples. For speed measurements.',
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='number of ddim sampling steps',
    )
    parser.add_argument(
        '--ddim_eta',
        type=float,
        default=1.0,
        help='ddim eta (eta=0.0 corresponds to deterministic sampling)',
    )
    parser.add_argument(
        '--n_iter',
        type=int,
        default=1,
        help='sample this often',
    )
    parser.add_argument(
        '--H',
        type=int,
        default=512,
        help='image height, in pixel space',
    )
    parser.add_argument(
        '--W',
        type=int,
        default=512,
        help='image width, in pixel space',
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=4,
        help='how many samples to produce for each given prompt. A.k.a batch size',
    )
    parser.add_argument(
        '--n_rows',
        type=int,
        default=0,
        help='rows in the grid (default: n_samples)',
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=5.0,
        help='unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))',
    )
    parser.add_argument(
        '--from-file',
        type=str,
        help='if specified, load prompts from this file',
    )
    parser.add_argument(
        '--config',
        type=str,
        help='path to config which constructs model',
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='./models/model.ckpt',
        help='path to checkpoint of model',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='the seed (for reproducible sampling)',
    )
    parser.add_argument(
        '--precision',
        type=str,
        help='evaluate at this precision',
        choices=['full', 'autocast'],
        default='full' if get_device_type == 'mps' else 'autocast'
    )
    parser.add_argument(
        '--sampler',
        type=str,
        help='sampler type to use',
        choices=['ddpm', 'ddim', 'plms', 'heun', 'lms'],
        default='heun'
    )
    parser.add_argument(
        '--use_ema',
        action='store_true',
        help='Use EMA weights',
    )
    parser.add_argument(
        '--same_seed',
        action='store_true',
        help='Use same seed for every prompt',
    )
    opt = parser.parse_args()

    if opt.config:
        config = yaml.safe_load(open(opt.config, 'r'))
    else:
        config = {'model': {'target': 'sd.models.diffusion.StableDiffusion'}}
    model: StableDiffusion = load_model_from_config(config, opt.ckpt, verbose=True, swap_ema=opt.use_ema, no_ema=not opt.use_ema)

    device = torch.device(get_device_type())
    model = model.to(device)
    model.eval()

    if opt.sampler == 'ddpm':
        sampler = DDPMSampler(num_timesteps=opt.steps)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(num_timesteps=opt.steps, unconditional_guidance_scale=opt.scale, eta=opt.ddim_eta)
    elif opt.sampler == 'plms':
        sampler = PLMSSampler(num_timesteps=opt.steps, unconditional_guidance_scale=opt.scale)
    elif opt.sampler == 'heun':
        model_k_wrapped = KCFGDenoiser(model)
        sampler = None
    else:
        raise ValueError(f'Unknown sampler type {opt.sampler}')

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    prompt = opt.prompt
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    if device.type == 'mps':
        precision_scope = nullcontext # have to use f32 on mps
    with precision_scope(device.type):
        seed_everything(opt.seed)
        downsample = 2**(model.first_stage_model.encoder.num_resolutions-1)
        shape = [1, model.first_stage_model.z_channels, opt.H // downsample, opt.W // downsample]
        # https://github.com/CompVis/stable-diffusion/issues/25#issuecomment-1229706811
        # MPS random is not currently deterministic w.r.t seed, so compute randn() on-CPU
        x_init = torch.randn(shape, device='cpu').to(device) if device.type == 'mps' else torch.randn(shape, device=device)

        prompts, *_ = data

        match opt.sampler:
            case 'heun' | 'lms':
                sigmas=get_sigmas_karras(
                    n=opt.steps,
                    sigma_min=model_k_wrapped.sigma_min.cpu(),
                    sigma_max=model_k_wrapped.sigma_max.cpu()
                    ).to(device)
                match opt.sampler:
                    case 'heun':
                        sample = sample_heun
                    case 'lms':
                        sample = sample_lms
                first_sigma, *_ = sigmas
                x = x_init * first_sigma
                uc = model.get_learned_conditioning('')
                c = model.get_learned_conditioning(prompts)
                extra_args = {
                    'cond': c,
                    'uncond': uc,
                    'cond_scale': opt.scale,
                    # 'target_embed': target_embed,
                }
                latents = sample(
                    model=model_k_wrapped,
                    x=x,
                    sigmas=sigmas,
                    extra_args=extra_args
                )
                x_samples = model.decode_first_stage(latents)
            case _:
                x_samples = model.sample(prompts, sampler, x_init=x_init)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0).cpu()

        x_sample, *_ = x_samples
        x_sample = 255 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f'{base_count:05}.{opt.seed}.png'))


if __name__ == '__main__':
    main()
