"""
PATCH 1 — shared_channel.py
Single shared channel module for the Method B rank-inversion experiment.

WHY THIS EXISTS
---------------
DeepJSCC (M1) and WITT (M2) ship DIFFERENT, non-equivalent channel models:
  * DeepJSCC/channel.py: adds REAL Gaussian noise per real element (noi_pwr/2);
    "Rayleigh" = TWO scalar gains applied to the two halves of the tensor.
  * WITT/net/channel.py: complex split-half mapping; clean per-complex-symbol
    AWGN; "Rayleigh" = i.i.d. MAGNITUDE-only fade (no phase), with per-element h.
A fair rank-inversion test requires an IDENTICAL channel in both methods. This
module replaces BOTH native channels. It implements proper i.i.d. per-symbol
COMPLEX fading (magnitude AND phase), not a scalar-gain approximation.

MODELLING CHOICES (documented for the paper's methodology section)
------------------------------------------------------------------
* Symbols: the real encoder output is mapped to complex symbols by split-half
  (first half -> real part, second half -> imaginary part), matching WITT's
  convention so the two methods are treated identically.
* Power: per-sample normalization to unit average power per COMPLEX symbol
  (E[|z|^2] = 1) BEFORE the channel, then rescaled back afterwards. This makes
  the SNR definition independent of each encoder's internal scaling -> identical
  channel semantics for M1 and M2.
* SNR: average SNR in dB. For unit signal power, complex AWGN has
  N0 = 10^(-snr/10); real and imag parts each have variance N0/2.
* Fading: h ~ i.i.d. per complex symbol.
    - Rayleigh: h = (a + jb)/sqrt(2), a,b ~ N(0,1)  ->  E[|h|^2] = 1.
    - Rician:   h = sqrt(K/(K+1)) * e^{j*theta_LoS}  +  sqrt(1/(K+1)) * (a+jb)/sqrt(2)
                with E[|h|^2] = 1. theta_LoS fixed = 0 (deterministic LoS phase).
* Receiver: perfect-CSI zero-forcing (ZF) equalization for fading channels
  (y/h = z + n/h). This is the standard coherent-receiver assumption and keeps
  the task learnable; its known cost is noise enhancement in deep fades, which
  is realistic and is itself part of the harness being studied. Set
  equalization='none' to feed faded+noisy symbols without equalization.

RICIAN K-FACTOR
---------------
Default K = 1 (0 dB): equal power in the LoS and scattered components
(K = LoS-power / scattered-power = 1) — a standard "moderate fading" operating
point. Rician block fading with a fixed K-factor is used for semantic-comms
evaluation in TONIC (arXiv:2605.21553, our Config B source paper); that paper
fixes a K-factor but does not state its numeric value in the extractable text,
so K=1 is OUR explicit, documented choice (moderate fading), NOT attributed to
TONIC. Override via k_factor=... if a different operating point is desired.

INTERFACE
---------
Functional:  apply_channel(x, snr_db, channel_type='awgn', k_factor=1.0,
                           equalization='zf', generator=None) -> Tensor
Module:      SharedChannel(channel_type, snr_db, k_factor=1.0, equalization='zf')
             .forward(x) -> Tensor      (drop-in replacement; see integration notes)

Both return a real tensor of the SAME shape as the input.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

_VALID = {"awgn", "rayleigh", "rician"}


def _to_complex_splithalf(xb: torch.Tensor):
    """xb: (B, N) real, N even -> (B, N//2) complex via split-half."""
    N = xb.shape[1]
    if N % 2 != 0:
        raise ValueError(f"per-sample element count must be even, got {N}")
    half = N // 2
    return torch.complex(xb[:, :half], xb[:, half:])


def _from_complex_splithalf(z: torch.Tensor) -> torch.Tensor:
    """(B, half) complex -> (B, N) real via split-half (inverse of above)."""
    return torch.cat([z.real, z.imag], dim=1)


def apply_channel(x: torch.Tensor,
                  snr_db: float,
                  channel_type: str = "awgn",
                  k_factor: float = 1.0,
                  equalization: str = "zf",
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """Pass real tensor x through the shared complex channel. Returns same shape."""
    channel_type = channel_type.lower()
    if channel_type not in _VALID:
        raise ValueError(f"channel_type must be one of {_VALID}, got {channel_type}")

    orig_shape = x.shape
    B = orig_shape[0]
    xb = x.reshape(B, -1)
    z = _to_complex_splithalf(xb)                      # (B, L) complex

    # per-sample unit average power per complex symbol
    pwr = torch.mean((z.real ** 2 + z.imag ** 2), dim=1, keepdim=True)  # (B,1)
    pwr = pwr.clamp_min(1e-12)
    zn = z / torch.sqrt(pwr)

    N0 = 10.0 ** (-snr_db / 10.0)                      # noise power per complex symbol
    dev = x.device
    shape = zn.shape

    def crandn():
        r = torch.randn(shape, generator=generator, device=dev)
        i = torch.randn(shape, generator=generator, device=dev)
        return torch.complex(r, i)

    noise = crandn() * math.sqrt(N0 / 2.0)             # CN(0, N0)

    if channel_type == "awgn":
        y = zn + noise
    else:
        if channel_type == "rayleigh":
            h = crandn() / math.sqrt(2.0)              # CN(0,1), E|h|^2 = 1
        else:  # rician
            K = float(k_factor)
            los = math.sqrt(K / (K + 1.0))             # theta_LoS = 0 -> real
            sca = math.sqrt(1.0 / (K + 1.0)) * (crandn() / math.sqrt(2.0))
            h = torch.complex(torch.full(shape, los, device=dev),
                              torch.zeros(shape, device=dev)) + sca
        y_rx = h * zn + noise
        if equalization == "zf":
            y = y_rx / h                               # perfect-CSI ZF: z + n/h
        elif equalization == "none":
            y = y_rx
        else:
            raise ValueError("equalization must be 'zf' or 'none'")

    y = y * torch.sqrt(pwr)                             # restore scale for decoder
    out = _from_complex_splithalf(y).reshape(orig_shape)
    return out.to(x.dtype)


class SharedChannel(nn.Module):
    """Drop-in replacement for both repos' Channel.

    DeepJSCC integration (model.py): replace `self.channel = Channel(channel_type, snr)`
        with `self.channel = SharedChannel(channel_type, snr)`; forward(z) is unchanged.
    WITT integration (net/channel.py): construct SharedChannel(args.channel_type, snr)
        and call .forward(x); ignore WITT's chan_param/avg_pwr (this module handles
        normalization internally). Provide a thin adapter at integration time.
    """

    def __init__(self, channel_type: str = "awgn", snr: float = 10.0,
                 k_factor: float = 1.0, equalization: str = "zf"):
        super().__init__()
        self.channel_type = channel_type.lower()
        self.snr = float(snr)
        self.k_factor = float(k_factor)
        self.equalization = equalization
        if self.channel_type not in _VALID:
            raise ValueError(f"channel_type must be one of {_VALID}")

    def forward(self, x: torch.Tensor, snr_db: float | None = None,
                channel_type: str | None = None) -> torch.Tensor:
        return apply_channel(
            x,
            snr_db=self.snr if snr_db is None else float(snr_db),
            channel_type=self.channel_type if channel_type is None else channel_type,
            k_factor=self.k_factor,
            equalization=self.equalization,
        )

    def get_channel(self):
        return self.channel_type, self.snr


class WITTChannelAdapter(nn.Module):
    """Signature adapter so WITT's `channel.forward(feature, chan_param, avg_pwr)`
    routes into the shared complex channel. `chan_param` is the SNR in dB.
    `avg_pwr` is accepted for signature compatibility and IGNORED: WITT never
    sets it (always False in its forward path; the avg_pwr=True branch of the
    native channel is dead code), and SharedChannel always applies per-sample
    unit-power normalization internally."""

    def __init__(self, args, config=None, k_factor=1.0, equalization="zf"):
        super().__init__()
        self.channel_type = args.channel_type.lower()   # awgn | rayleigh | rician
        self.k_factor = float(k_factor)
        self.equalization = equalization

    def forward(self, input, chan_param, avg_pwr=False):
        return apply_channel(input, snr_db=float(chan_param),
                             channel_type=self.channel_type,
                             k_factor=self.k_factor,
                             equalization=self.equalization)

    def get_channel(self):
        return self.channel_type, None


# ----------------------------------------------------------------------------
# UNIT TEST (CPU-only; no GPU). Verifies realized AWGN SNR matches the target
# within 0.1 dB over 1000 noise realizations, per the patch requirement.
# ----------------------------------------------------------------------------
def test_awgn_snr(tol_db: float = 0.1, n_real: int = 1000, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    results = {}
    for target in [0.0, 2.0, 7.0, 12.0, 20.0]:
        # fixed unit-power complex signal so realized SNR is unambiguous
        x = torch.randn(8, 256, generator=g)           # 128 complex symbols/sample
        sig_acc, noi_acc = 0.0, 0.0
        for _ in range(n_real):
            # reconstruct the normalized symbols the module uses internally
            xb = x.reshape(8, -1)
            z = _to_complex_splithalf(xb)
            pwr = torch.mean(z.real ** 2 + z.imag ** 2, dim=1, keepdim=True).clamp_min(1e-12)
            zn = z / torch.sqrt(pwr)
            y = apply_channel(x, target, "awgn", generator=g)
            yb = _to_complex_splithalf(y.reshape(8, -1))
            yn = yb / torch.sqrt(pwr)                   # undo rescale -> compare to zn
            noise = yn - zn
            sig_acc += torch.mean(zn.real ** 2 + zn.imag ** 2).item()
            noi_acc += torch.mean(noise.real ** 2 + noise.imag ** 2).item()
        realized = 10.0 * math.log10((sig_acc / n_real) / (noi_acc / n_real))
        err = abs(realized - target)
        results[target] = (realized, err)
        status = "PASS" if err <= tol_db else "FAIL"
        print(f"  target={target:5.1f} dB | realized={realized:7.3f} dB | "
              f"err={err:.4f} dB | {status}")
    worst = max(e for _, e in results.values())
    print(f"worst error = {worst:.4f} dB (tol {tol_db} dB) -> "
          f"{'ALL PASS' if worst <= tol_db else 'FAIL'}")
    return results


if __name__ == "__main__":
    print("AWGN realized-SNR unit test (CPU):")
    test_awgn_snr()
