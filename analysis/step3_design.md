# Step 3 — Method B Rank-Inversion Harness Design

**Saved (UTC): 2026-06-11T02:21:49Z**
Companion to `method_b_preregistration.md` (frozen hypotheses/configs/decision rule + Amendment 1).
This document records the *execution* design; it does not alter the frozen pre-registration core.

## Confirmed design decisions
- **D1 — CONFIRMED. Training objective held constant = MSE on [0,255] for all runs.** Only the
  evaluation metric varies per config (SSIM for A, top-1 accuracy for B). Isolates the inversion to
  the evaluation harness (channel + SNR + metric); avoids the "you trained different objectives"
  confound.
- **D2 — CONFIRMED. SNR-adaptive + channel-adaptive training:** one model per (method, config, seed);
  each batch samples SNR uniformly in the config range and a channel uniformly from the config set.
  12 models total.
- **D3 — CONFIRMED. WITT `C=16`** so CBR = C/96 = 1/6 matches M1's k/n = 1/6 (c=8).
- **D4 — CONFIRMED. Model selection on a 45k/5k train/val split** (best checkpoint by the config
  metric on val); final numbers on the 10k test set.
- **Epochs — CONFIRMED: M1 = 300, M2 (WITT) = 300.** (M2 sanity also 300.)
- **Seeds — CONFIRMED: {0, 1, 2}.** Seed `random`, `numpy`, `torch`, `torch.cuda`; cudnn
  deterministic = True, benchmark = False for the seeded full runs.

## Per-method training
| | M1 DeepJSCC | M2 WITT |
|---|---|---|
| bottleneck | c=8 (k/n=1/6) | C=16 (CBR=1/6) |
| optimizer / lr / wd | Adam / 1e-3 / 5e-4 | Adam / 1e-4 / 5e-4 |
| loss | MSE [0,255] | MSE [0,255] |
| batch / epochs | 128 / 300 | 128 / 300 |

## Configs (sampled per batch)
- **Config A:** channel ∈ {AWGN, Rayleigh}, SNR ~ U[2,12] dB; eval metric **SSIM**. Source: arXiv:2606.00114.
- **Config B:** channel ∈ {AWGN, Rayleigh, Rician(K=1)}, SNR ~ U[0,12] dB; eval metric **top-1 accuracy**. Source: arXiv:2605.21553.
- Shared channel module (`patches/shared_channel.py`), identical for both methods.

## GATE 1 — M2 (WITT) sanity
- AWGN, SNR-adaptive 2–12 dB, C=16, [0,255] MSE, lr=1e-4, 300 epochs, device 2.
- Eval test PSNR at SNR {2,7,12} + noiseless (channel disabled on the trained model).
- Reference = M1's validated AWGN curve under the same shared channel: M1 @ {2,7,12} ≈ {24.5, 27.8, 29.7}; noiseless 32.2 dB.
- **PASS iff** (i) PSNR monotonic in SNR, (ii) WITT ≥ M1 − 1 dB at each point, (iii) noiseless ≥ ~31 dB.
- **FAIL → diagnostic path:** noiseless WITT check → confirm C=16 + [0,255] loss applied → native-vs-shared channel at one SNR → raise epochs if undertrained. No full runs until pass.

## GATE 2 — full 12 runs
- {M1, M2} × {Config A, Config B} × {seed 0,1,2}.
- Parallelization: device 2 → WITT (heavy) runs; device 0 → M1 runs where VRAM permits (~6 GB free);
  device 1 unavailable (busy). Confirm free memory at launch.
- Checkpoint **best-on-val** per (method,config,seed) → `methods/methodB_runs/{m}_{cfg}_s{seed}/best.pt` (+ `last.pt`, `run.json`).

## GATE 3 — eval + decision
- **Classifier sanity first:** chenyaofo ResNet-20, `assert_clean_accuracy ≥ 0.91` on clean test; abort Config B if it fails.
- Eval grids (test set): A = {AWGN,Rayleigh} × SNR{2,4,7,10,12} → mean SSIM (+ mean PSNR alongside); B = {AWGN,Rayleigh,Rician} × SNR{0,2,4,7,10,12} → mean top-1 acc.
- Four numbers = mean ± std over seeds {0,1,2}: [SSIM-A-M1], [SSIM-A-M2], [ACC-B-M1], [ACC-B-M2].
- **Decision rule (deterministic script):** per config winner = higher mean metric; inversion CONFIRMED
  iff winner(A) ≠ winner(B) AND each config's margin exceeds noise floor = max(absolute floor, 2× pooled
  seed-std), absolute floor = 0.5 dB PSNR-equiv (A, checked on PSNR margin) / 2% accuracy (B). Script emits
  CONFIRMED / NOT CONFIRMED + margins + per-seed values. **User confirms the verdict before any number
  enters the paper.**

## Estimated GPU-hours
| Component | Runs | ~time/run | Subtotal |
|---|---|---|---|
| M2 sanity (300 ep) | 1 | est. (sanity measures it) | ~1–2 h |
| M1 full (300 ep) | 6 | ~0.35 h | ~2.1 h |
| M2 full (300 ep) | 6 | ~2 h (est., refine after sanity) | ~12 h |
| Eval + classifier sanity | — | minutes | ~0.5 h |
| **Total compute** | | | **~16 h** |
Wall-clock under current occupancy (~1–2 usable L40s): ~7–14 h; refined after the M2 sanity pins WITT timing.

## Execution log
- **2026-06-11 — GATE 1 (WITT sanity, 300 ep, AWGN, C=16) PASS.** PSNR {2,7,12}={25.67,29.48,32.11},
  noiseless 34.45; monotonic; WITT beats M1 (600 ep {24.5,27.8,29.7}, noiseless 32.2) at every point;
  noiseless ≥ 31. Measured **96 s/epoch (8.3 h / 300 ep)**.
- **2026-06-11 — Epoch-budget calibration: WITT@150 = {25.03, 29.36, 32.03}.** Gaps to @300:
  SNR2 0.64 (>0.5), SNR7 0.12, SNR12 0.08. Fails "all-three within 0.5 dB" at low SNR ⇒
  **full-run budget = 300 epochs.**
- **Cost consequence:** M2 full = 6 runs × 8.3 h ≈ **50 h**; M1 full ≈ 2 h. Wall-clock ≈ 50 h on device 2
  alone; ≈ 25 h if a 2nd L40 frees up for parallel M2 runs.
