import torch

from torch import __version__, Tensor, enable_grad, autograd
from torch.nn import functional as F, Conv2d, GroupNorm

device = torch.device('mps')

print(f'torch.__version__: {__version__}')
shape = [1, 4, 16, 16]
x = torch.full(shape, 7.0, device=device)

target = torch.ones((1, 3, 128, 128), device=device)

conv_in = Conv2d(4, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), device=device)
conv_out = Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), device=device)
norm = GroupNorm(32, 128, eps=1e-6, affine=True, device=device)

with enable_grad(), autograd.detect_anomaly():
    x = x.detach().requires_grad_()

    out = 5.5 * x

    out = conv_in(out)
    out = out+norm(out)
    out = out+norm(out)
    out = out+norm(out)
    out = F.interpolate(out, scale_factor=8.0, mode="nearest")
    out = norm(out)
    out = conv_out(out)

    loss: Tensor = (out - target).norm(dim=-1).sum()
    grad: Tensor = -autograd.grad(loss, x)[0]
    assert not grad.detach().isnan().any().item(), 'NaN gradients returned by autograd'

print('Ran to completion successfully; did not encounter NaN gradients.')
