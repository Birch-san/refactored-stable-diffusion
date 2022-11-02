import sys, os
sys.path.append(os.getcwd())

import torch

from pytorch_lightning import seed_everything
from torch import __version__, Tensor, enable_grad, autograd
from torch.nn import functional as F, Conv2d, Linear, GroupNorm
from typing import Literal

def get_device_type() -> Literal['cuda', 'mps', 'cpu']:
    if(torch.cuda.is_available()):
        return 'cuda'
    elif(torch.backends.mps.is_available()):
        return 'mps'
    else:
        return 'cpu'

def main():
    device = torch.device(get_device_type())

    encoder_z_channels=4
    encoder_num_resolutions=4
    post_quant_conv = Conv2d(4, encoder_z_channels, 1, device=device)

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
    target = torch.ones((1, 3, 128, 128), device=device)

    conv_in = Conv2d(4, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), device=device)
    conv_out = Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), device=device)
    norm = GroupNorm(32, 128, eps=1e-6, affine=True, device=device)
    norm_out = GroupNorm(32, 128, eps=1e-06, affine=True, device=device)

    with enable_grad():#, autograd.detect_anomaly():
        x = x.detach().requires_grad_()

        denoised = 5.5 * x
        denoised = post_quant_conv(denoised)

        decoded = conv_in(denoised)
        decoded = decoded+norm(decoded)
        decoded = decoded+norm(decoded)
        decoded = decoded+norm(decoded)
        decoded = F.interpolate(decoded, scale_factor=8.0, mode="nearest")
        decoded = norm_out(decoded)
        decoded = conv_out(decoded)

        loss: Tensor = (decoded - target).norm(dim=-1).sum()
        grad: Tensor = -autograd.grad(loss, x)[0]
        print(f'NaN gradients: {grad.detach().isnan().any().item()}')

    print("Didn't crash")


if __name__ == '__main__':
    main()
