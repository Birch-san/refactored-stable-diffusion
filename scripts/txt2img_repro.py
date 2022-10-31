import argparse, sys, os
sys.path.append(os.getcwd())

import torch
import numpy as np
from PIL import Image
from einops import rearrange
import yaml

from pytorch_lightning import seed_everything
from torch import autocast, Tensor, enable_grad
from torch.nn import functional as F
from contextlib import nullcontext
from typing import Protocol, Iterable, Optional, Tuple
import open_clip
from open_clip import CLIP as OpenCLIP
from torchvision import transforms
from kornia import augmentation as KA

from sd.util import load_model_from_config
from sd.samplers.ddpm import DDPMSampler
from sd.samplers.ddim import DDIMSampler
from sd.samplers.plms import PLMSSampler
from sd.modules.device import get_device_type
from sd.models.diffusion import StableDiffusion

from k_diffusion.sampling import sample_heun, sample_lms, get_sigmas_karras, append_zero
from k_diffusion.external import DiscreteEpsDDPMDenoiser
from k_diffusion.utils import append_dims

K_DIFF_SAMPLERS = { 'heun', 'lms' }

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

def spherical_dist_loss(x: Tensor, y: Tensor) -> Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

class ClipGuidedDenoiser():
    denoiser: KCFGDenoiser
    clip_model: OpenCLIP
    clip_augmentations: bool
    clip_guidance_scale: float
    clip_normalize: transforms.Normalize
    clip_size: Tuple[int, int]
    aug: Optional[KA.RandomAffine]
    def __init__(
        self,
        denoiser: KCFGDenoiser,
        clip_model: OpenCLIP,
        clip_augmentations: bool,
        clip_guidance_scale: float,
    ):
        self.denoiser = denoiser
        self.clip_model = clip_model
        self.clip_size = clip_model.visual.image_size
        self.clip_augmentations = clip_augmentations
        self.clip_guidance_scale = clip_guidance_scale
        self.clip_normalize = transforms.Normalize(mean=clip_model.visual.image_mean, std=clip_model.visual.image_std)
        self.aug = KA.RandomAffine(0, (1/14, 1/14), p=1, padding_mode='border') if clip_augmentations else None

    @enable_grad()
    def __call__(self, x: Tensor, sigma: Tensor, target_embed: Tensor, **kwargs) -> Tensor:
        x = x.detach().requires_grad_()
        denoised: Tensor = self.denoiser(x, sigma, **kwargs)
        cond_grad: Tensor = self.cond_fn(x, denoised=denoised, target_embed=target_embed).detach()
        ndim = x.ndim
        del x
        cond_denoised: Tensor = denoised.detach() + cond_grad * append_dims(sigma**2, ndim)
        return cond_denoised

    def cond_fn(self, x: Tensor, denoised: Tensor, target_embed: Tensor) -> Tensor:
        device = denoised.device
        decoded: Tensor = self.denoiser.inner_model.decode_first_stage(denoised)
        del denoised
        renormalized: Tensor = decoded.add(1).div(2)
        del decoded
        if self.clip_augmentations:
            # this particular approach to augmentation crashes on MPS, so we transfer to CPU (for now)
            # :27:11: error: invalid input tensor shapes, indices shape and updates shape must be equal
            # -:27:11: note: see current operation: %25 = "mps.scatter_along_axis"(%23, %arg3, %24, %1) {mode = 6 : i32} : (tensor<786432xf32>, tensor<512xf32>, tensor<262144xi32>, tensor<i32>) -> tensor<786432xf32>
            # TODO: this approach (from k-diffusion example) produces just the one augmentation,
            #       whereas diffusers approach is to use many and sum their losses. should we?
            renormalized = self.aug(renormalized.cpu()).to(device) if device.type == 'mps' else self.aug(renormalized)
        clamped: Tensor = renormalized.clamp(0, 1)
        del renormalized
        image_embed: Tensor = self.get_image_embed(clamped)
        del clamped
        # TODO: does this do the right thing for multi-sample?
        # TODO: do we want .mean() here or .sum()? or both?
        #       k-diffusion example used just .sum(), but k-diff was single-aug. maybe that was for multi-sample?
        #       whereas diffusers uses .mean() (this seemed to be over a single number, but maybe when you have multiple samples it becomes the mean of the loss over your n samples?),
        #       then uses sum() (which for multi-aug would sum the losses of each aug)
        loss: Tensor = spherical_dist_loss(image_embed, target_embed).sum() * self.clip_guidance_scale
        del image_embed
        # TODO: does this do the right thing for multi-sample?
        grad: Tensor = -torch.autograd.grad(loss, x)[0]
        return grad
    
    def get_image_embed(self, x: Tensor) -> Tensor:
        if x.shape[2:4] != self.clip_size:
            # k-diffusion example used a bicubic resize, via resize_right library
            # x = resize(x, out_shape=clip_size, pad_mode='reflect')
            # but diffusers' bilinear resize produced a nicer bear
            x = transforms.Resize(self.clip_size)(x)
        x: Tensor = self.clip_normalize(x)
        x: Tensor = self.clip_model.encode_image(x).float()

        return F.normalize(x)

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
    parser.add_argument(
        "--clip_guidance",
        action='store_true',
        help="guides diffusion using OpenCLIP"
    )
    parser.add_argument(
        "--clip_guidance_scale",
        type=float,
        default=500.,
        help="CLIP guidance scale",
    )
    parser.add_argument(
        "--clip_prompt",
        type=str,
        nargs="?",
        default=None,
        help="alternative prompt upon which OpenCLIP should guide diffusion."
    )
    parser.add_argument(
        "--clip_augmentations",
        action='store_true',
        help="CLIP-guided diffusion will compute embedding from a subtly-transformed copy of the denoised latents instead of the real thing"
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="ViT-B-32",
        # big:
        # --clip_model_name "ViT-H-14" --clip_model_version "laion2b_s32b_b79k"
        # less big:
        # --clip_model_name "ViT-B-32" --clip_model_version "laion2b_s34b_b79k"
        # the big one is slow on Mac (30s/it) and the bear it generated looked worse
        help="CLIP model name passed to OpenCLIP",
    )
    parser.add_argument(
        "--clip_model_version",
        type=str,
        default="laion2b_s34b_b79k",
        # big:
        # --clip_model_name "ViT-H-14" --clip_model_version "laion2b_s32b_b79k"
        # less big:
        # --clip_model_name "ViT-B-32" --clip_model_version "laion2b_s34b_b79k"
        # the big one is slow on Mac (30s/it) and the bear it generated looked worse
        help="CLIP checkpoint name passed to OpenCLIP",
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

    match opt.sampler:
        case 'ddpm':
            sampler = DDPMSampler(num_timesteps=opt.steps)
        case 'ddim':
            sampler = DDIMSampler(num_timesteps=opt.steps, unconditional_guidance_scale=opt.scale, eta=opt.ddim_eta)
        case 'plms':
            sampler = PLMSSampler(num_timesteps=opt.steps, unconditional_guidance_scale=opt.scale)
        case 'heun' | 'lms':
            model_k_wrapped = KCFGDenoiser(model)
            sampler = None
        case _:
            raise ValueError(f'Unknown sampler type {opt.sampler}')
    
    clip_model: Optional[OpenCLIP] = None
    if opt.clip_guidance:
        assert opt.sampler in K_DIFF_SAMPLERS
        clip_model: OpenCLIP = open_clip.create_model(opt.clip_model_name, opt.clip_model_version, device=device)
        # clip_model: OpenCLIP = open_clip.create_model(opt.clip_model_name, device=device)
        clip_model.requires_grad_(False)
        clip_denoiser = ClipGuidedDenoiser(
            denoiser=model_k_wrapped,
            clip_model=clip_model,
            clip_augmentations=opt.clip_augmentations,
            clip_guidance_scale=opt.clip_guidance_scale,
        )

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
        prompt, *_ = prompts

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
                conditions = ['', prompt]
                uc, c = model.get_learned_conditioning(conditions).chunk(len(conditions))
                target_embed: Optional[Tensor] = None
                if opt.clip_guidance:
                    tokens: Tensor = open_clip.tokenize(opt.clip_prompt or prompts[0]).to(device)
                    encoded: Tensor = clip_model.encode_text(tokens).to(device)
                    del tokens
                    target_embed: Tensor = F.normalize(encoded.float())
                extra_args = {
                    'cond': c,
                    'uncond': uc,
                    'cond_scale': opt.scale,
                    'target_embed': target_embed,
                }
                denoiser = clip_denoiser if opt.clip_guidance else model_k_wrapped
                latents = sample(
                    model=denoiser,
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
