# Integrating the shared channel into the two evaluated methods

The two methods are third-party. Clone them, put `harness/` on the import path, and
replace each repo's native channel with the shared one via a **single import change**
(the rest of each repo is unchanged, so the original code stays auditable).

## Methods

- **M1 — DeepJSCC (CNN):** `https://github.com/chunbaobao/Deep-JSCC-PyTorch`
- **M2 — WITT (Swin-transformer):** `https://github.com/KeYang8/WITT`

## Why replace the channel

The two repos ship different, non-equivalent channel models (real-noise per element vs.
complex split-half; magnitude-only vs. proper fading). A fair re-evaluation requires an
**identical** channel in both. `harness/shared_channel.py` provides it: proper i.i.d.
per-symbol complex fading (AWGN / Rayleigh / Rician), perfect-CSI zero-forcing
equalization, Rician K=1, with a unit-tested ±0.1 dB AWGN SNR calibration.

## M1 — `Deep-JSCC-PyTorch/model.py`

```diff
-from channel import Channel
+import os, sys
+sys.path.insert(0, "<path-to>/harness")
+from shared_channel import SharedChannel as Channel
```
`DeepJSCC.__init__` / `change_channel` then build `SharedChannel`; `forward(z)` is
unchanged. Train with MSE on the **[0,255]** scale (matches the upstream recipe).

## M2 — `WITT/net/network.py`

```diff
-from net.channel import Channel
+import os, sys
+sys.path.insert(0, "<path-to>/harness")
+from shared_channel import WITTChannelAdapter as Channel
```
`WITT.__init__` builds `WITTChannelAdapter(args, config)`; the adapter maps WITT's
`forward(feature, chan_param, avg_pwr)` onto the shared channel (`chan_param` = SNR in
dB; `avg_pwr` ignored, since WITT never sets it). Set WITT `C=16` to match k/n = 1/6.

## Per-batch channel/SNR control (used by the experiment scripts)

- M1: set `model.channel.channel_type` and `model.channel.snr` before each forward.
- M2: set `net.channel.channel_type`; pass `given_SNR` for a continuous SNR.

See `run_methodB_train.py` (`recon()`) for the exact usage.
