import sys, os
sys.path.append(os.getcwd())

import torch

from pytorch_lightning import seed_everything
from torch import __version__, Tensor, enable_grad
from torch.nn import functional as F, GELU
from open_clip.model import VisualTransformer
from torchvision import transforms

from sd.modules.device import get_device_type
from sd.models.autoencoder import AutoencoderKL
from sd.modules.unet import UNetModel

def main():
    device = torch.device(get_device_type())
    
    unet = UNetModel(
        num_res_blocks=0,
        channel_mult=[],
        num_heads=1,
        attention_resolutions=[1]
    )
    unet = unet.to(device)
    unet.requires_grad_(False)
    autoencoder = AutoencoderKL()
    autoencoder = autoencoder.to(device)
    autoencoder.requires_grad_(False)

    clip_vit = VisualTransformer(
        image_size=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        mlp_ratio=4.0,
        output_dim=512,
        act_layer=GELU,
    )
    clip_vit = clip_vit.to(device)
    clip_vit.requires_grad_(False)
    clip_normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    clip_guidance_scale = 100

    print(f'torch.__version__: {__version__}')
    seed_everything(2400270449)
    downsample = 2**(autoencoder.encoder.num_resolutions-1)
    width = 128
    height = 128
    shape = [1, autoencoder.z_channels, height // downsample, width // downsample]
    # https://github.com/CompVis/stable-diffusion/issues/25#issuecomment-1229706811
    # MPS random is not currently deterministic w.r.t seed, so compute randn() on-CPU
    x_init = torch.randn(shape, device='cpu').to(device) if device.type == 'mps' else torch.randn(shape, device=device)

    x = x_init * 14.614643096923828
    c = torch.ones((1, 77, 768), dtype=torch.float32, device=device)
    target_embed = torch.ones((1, 512), device=device)
    target_embed = F.normalize(target_embed)

    with enable_grad():
        x = x.detach().requires_grad_()
        c_in = torch.full((1, 1, 1, 1), 0.0682649165391922, device='mps')
        c_out = torch.full((1, 1, 1, 1), -14.614643096923828, device='mps')
        eps: Tensor = unet.forward(x=x * c_in, timesteps=torch.tensor([999], device=device), context=c)
        denoised: Tensor = x + eps * c_out

        decoded: Tensor = autoencoder.decode(1. / 0.18215 * denoised)
        decoded = decoded.add(1).div(2)
        decoded = decoded.clamp(0, 1)

        image_embed = transforms.Resize(clip_vit.image_size)(decoded)
        image_embed: Tensor = clip_normalize(image_embed)
        image_embed: Tensor = clip_vit.forward(image_embed)
        image_embed = F.normalize(image_embed)

        loss: Tensor = (image_embed - target_embed).norm(dim=-1).div(2).arcsin().pow(2).mul(2).sum() * clip_guidance_scale
        grad: Tensor = -torch.autograd.grad(loss, x)[0]
        print(f'NaN gradients: {grad.detach().isnan().any().item()}')

    print("Didn't crash")


if __name__ == '__main__':
    main()
