import sys, os
sys.path.append(os.getcwd())

import torch

from pytorch_lightning import seed_everything
from torch import __version__, Tensor, enable_grad, autograd
from torch.nn import functional as F, Conv2d, Linear

from sd.modules.device import get_device_type
from sd.modules.encoder import Decoder

def main():
    device = torch.device(get_device_type())

    encoder_z_channels=4
    encoder_num_resolutions=4
    post_quant_conv = Conv2d(4, encoder_z_channels, 1, device=device)
    decoder = Decoder()
    decoder = decoder.to(device)
    decoder.requires_grad_(False)

    print(f'torch.__version__: {__version__}')
    seed_everything(2400270449)
    downsample = 2**(encoder_num_resolutions-1)
    width = 128
    height = 128
    shape = [1, encoder_z_channels, height // downsample, width // downsample]
    # https://github.com/CompVis/stable-diffusion/issues/25#issuecomment-1229706811
    # MPS random is not currently deterministic w.r.t seed, so compute randn() on-CPU
    x_init = torch.randn(shape, device='cpu').to(device) if device.type == 'mps' else torch.randn(shape, device=device)

    x = x_init * 14.614643096923828
    target_embed = torch.ones((1, 512), device=device)
    target_embed = F.normalize(target_embed)

    silly_vit = Linear(3*128**2, 512, device=device)

    with enable_grad():#, autograd.detect_anomaly():
        x = x.detach().requires_grad_()

        denoised = 1. / 0.18215 * x
        denoised = post_quant_conv(denoised)
        decoded: Tensor = decoder.forward(denoised)
        decoded = decoded.add(1).div(2)
        decoded = decoded.clamp(0, 1)

        image_embed = silly_vit(decoded.flatten(1))
        image_embed = F.normalize(image_embed)

        loss: Tensor = (image_embed - target_embed).norm(dim=-1).div(2).arcsin().pow(2).mul(2).sum()
        grad: Tensor = -autograd.grad(loss, x)[0]
        print(f'NaN gradients: {grad.detach().isnan().any().item()}')

    print("Didn't crash")


if __name__ == '__main__':
    main()
