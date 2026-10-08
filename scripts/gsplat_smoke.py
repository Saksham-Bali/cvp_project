"""Check that gsplat works on this GPU (forces the one-time CUDA compile and times it)."""
import time

import torch

t = time.time()
from gsplat import rasterization  # noqa: E402

assert torch.cuda.is_available(), "No CUDA GPU (Colab: Runtime > Change runtime type > T4 GPU)"
dev = "cuda"
N = 1000
means = torch.randn(N, 3, device=dev) * 0.5 + torch.tensor([0, 0, 3.0], device=dev)
quats = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)
scales = torch.full((N, 3), 0.03, device=dev)
opac = torch.full((N,), 0.8, device=dev)
cols = torch.rand(N, 3, device=dev)
V = torch.eye(4, device=dev)[None]
K = torch.tensor([[200.0, 0, 128], [0, 200.0, 128], [0, 0, 1]], device=dev)[None]
img, alpha, info = rasterization(means, quats, scales, opac, cols, V, K, 256, 256)
torch.cuda.synchronize()
print(f"gsplat OK on {torch.cuda.get_device_name(0)}: rendered {tuple(img.shape)}, "
      f"mean alpha {alpha.mean().item():.3f}  (incl. compile: {time.time() - t:.0f}s)")
