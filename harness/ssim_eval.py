"""
PATCH 3 — ssim_eval.py
Plain SSIM (NOT MS-SSIM) evaluation wrapper for Config A (source 2606.00114).

WITT natively reports MS-SSIM; Config A's source paper (Recursive Vision
Transformer, arXiv:2606.00114) declares plain **SSIM** as its evaluation metric
("We employ the Structural Similarity Index Measure (SSIM) ... as the evaluation
metric"). To keep Config A faithful, both M1 and M2 are scored with this single
SSIM implementation.

SSIM parameters: 2606.00114 does not state the window size / data range in the
extractable text, so we use the canonical Wang et al. (2004) definition:
  - Gaussian window, win_size = 11, win_sigma = 1.5
  - data_range = 1.0 (images in [0,1])
  - channel-averaged (mean over RGB)
These are the de-facto defaults in both pytorch_msssim and skimage; flag in the
methodology that the exact window of 2606.00114 was unspecified and standard
SSIM was used.

Primary backend: pytorch_msssim.ssim (GPU-capable, batched).
Fallback: skimage.metrics.structural_similarity (CPU, per-image).
"""
from __future__ import annotations
import torch

_WIN_SIZE = 11
_WIN_SIGMA = 1.5
_DATA_RANGE = 1.0

try:
    from pytorch_msssim import ssim as _torch_ssim
    _HAVE_TORCH_SSIM = True
except Exception:
    _HAVE_TORCH_SSIM = False


def ssim_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean SSIM over a batch. x,y: (B,3,H,W) in [0,1]. Returns scalar tensor.

    Uses pytorch_msssim if available (preferred: batched, differentiable, same
    device). Otherwise falls back to skimage per-image on CPU.
    """
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)
    if _HAVE_TORCH_SSIM:
        return _torch_ssim(x, y, data_range=_DATA_RANGE,
                           win_size=_WIN_SIZE, win_sigma=_WIN_SIGMA,
                           size_average=True)
    # ---- skimage fallback (CPU) ----
    from skimage.metrics import structural_similarity as sk_ssim
    import numpy as np
    xs = x.detach().cpu().numpy()
    ys = y.detach().cpu().numpy()
    vals = []
    for i in range(xs.shape[0]):
        a = np.transpose(xs[i], (1, 2, 0))
        b = np.transpose(ys[i], (1, 2, 0))
        vals.append(sk_ssim(a, b, data_range=_DATA_RANGE,
                            win_size=_WIN_SIZE, channel_axis=2,
                            gaussian_weights=True, sigma=_WIN_SIGMA))
    return torch.tensor(float(np.mean(vals)))


@torch.no_grad()
def ssim_over_loader(model_forward, loader, device: str = "cuda:0") -> float:
    """Average SSIM over a loader. model_forward(imgs01)->recon01, both in [0,1]."""
    tot, n = 0.0, 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        recon = model_forward(imgs).clamp(0, 1)
        b = imgs.size(0)
        tot += ssim_batch(recon, imgs).item() * b
        n += b
    return tot / max(n, 1)


# Dependency note: `pip install pytorch-msssim` into the ictc env (CPU-side, no
# GPU). If absent, the skimage fallback (already present via scikit-image) is used.
