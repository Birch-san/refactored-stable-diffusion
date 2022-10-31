import argparse, sys, os
sys.path.append(os.getcwd())

import torch
import numpy as np
from PIL import Image
from einops import rearrange

from pytorch_lightning import seed_everything
from torch import __version__, autocast, Tensor, enable_grad
from torch.nn import functional as F
from contextlib import nullcontext
from typing import Optional
import open_clip
from open_clip import CLIP as OpenCLIP
from torchvision import transforms

from sd.modules.device import get_device_type
from sd.models.autoencoder import AutoencoderKL
from sd.modules.unet import UNetModel

def spherical_dist_loss(x: Tensor, y: Tensor) -> Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

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

    device = torch.device(get_device_type())
    
    unet = UNetModel()
    unet = unet.to(device)
    for param in unet.parameters():
        param.requires_grad = False
    autoencoder = AutoencoderKL()
    autoencoder = autoencoder.to(device)
    for param in autoencoder.parameters():
        param.requires_grad = False
    
    clip_model: OpenCLIP = open_clip.create_model(opt.clip_model_name, device=device)
    clip_model.requires_grad_(False)
    clip_normalize = transforms.Normalize(mean=clip_model.visual.image_mean, std=clip_model.visual.image_std)

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
    print(f'torch.__version__: {__version__}')
    with precision_scope(device.type):
        seed_everything(opt.seed)
        downsample = 2**(autoencoder.encoder.num_resolutions-1)
        shape = [1, autoencoder.z_channels, opt.H // downsample, opt.W // downsample]
        # https://github.com/CompVis/stable-diffusion/issues/25#issuecomment-1229706811
        # MPS random is not currently deterministic w.r.t seed, so compute randn() on-CPU
        x_init = torch.randn(shape, device='cpu').to(device) if device.type == 'mps' else torch.randn(shape, device=device)

        prompts, *_ = data
        prompt, *_ = prompts

        x = x_init * 14.614643096923828
        # uc = torch.ones((1, 77, 768), dtype=torch.float32, device=device)
        c = torch.ones((1, 77, 768), dtype=torch.float32, device=device)
        target_embed: Optional[Tensor] = None
        if opt.clip_guidance:
            target_embed = torch.ones((1, 512), device=device)

        def get_image_embed(x: Tensor) -> Tensor:
            if x.shape[2:4] != clip_model.visual.image_size:
                # k-diffusion example used a bicubic resize, via resize_right library
                # x = resize(x, out_shape=clip_size, pad_mode='reflect')
                # but diffusers' bilinear resize produced a nicer bear
                x = transforms.Resize(clip_model.visual.image_size)(x)
            x: Tensor = clip_normalize(x)
            x: Tensor = clip_model.encode_image(x).float()

            return F.normalize(x)

        def cond_fn(x: Tensor, denoised: Tensor, target_embed: Tensor) -> Tensor:
            unscaled = 1. / 0.18215 * x
            decoded: Tensor = autoencoder.decode(unscaled)
            del denoised
            renormalized: Tensor = decoded.add(1).div(2)
            del decoded
            clamped: Tensor = renormalized.clamp(0, 1)
            del renormalized
            image_embed: Tensor = get_image_embed(clamped)
            # image_embed: Tensor = torch.ones((1, 512), dtype=x.dtype, device=x.device, requires_grad=True)
            del clamped
            # TODO: does this do the right thing for multi-sample?
            # TODO: do we want .mean() here or .sum()? or both?
            #       k-diffusion example used just .sum(), but k-diff was single-aug. maybe that was for multi-sample?
            #       whereas diffusers uses .mean() (this seemed to be over a single number, but maybe when you have multiple samples it becomes the mean of the loss over your n samples?),
            #       then uses sum() (which for multi-aug would sum the losses of each aug)
            loss: Tensor = spherical_dist_loss(image_embed, target_embed).sum() * opt.clip_guidance_scale
            del image_embed
            # TODO: does this do the right thing for multi-sample?
            grad: Tensor = -torch.autograd.grad(loss, x)[0]
            return grad

        with enable_grad():
            x = x.detach().requires_grad_()
            c_in = torch.full((1, 1, 1, 1), 0.0682649165391922, device='mps')
            c_out = torch.full((1, 1, 1, 1), -14.614643096923828, device='mps')
            eps: Tensor = unet.forward(x=x * c_in, timesteps=torch.tensor([999], device=device), context=c)
            denoised: Tensor = x + eps * c_out
            cond_grad: Tensor = cond_fn(x, denoised=denoised, target_embed=target_embed).detach()
            print(f'NaN gradients: {cond_grad.isnan().any().item()}')
            del x
            cond_denoised: Tensor = denoised.detach() + cond_grad * torch.full((1, 1, 1, 1), 14.614643096923828**2, device='mps')

        unscaled = 1. / 0.18215 * cond_denoised
        x_samples = autoencoder.decode(unscaled)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0).cpu()

        x_sample, *_ = x_samples
        x_sample = 255 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f'{base_count:05}.{opt.seed}.png'))
        print("Didn't crash")


if __name__ == '__main__':
    main()
