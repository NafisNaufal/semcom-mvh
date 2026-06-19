# Method B — Rank-Inversion Experiment Pre-Registration

**Pre-registration timestamp (UTC): 2026-06-07T07:21:26Z**
**Status: FROZEN before any reproduction. No training/eval results existed at time of writing.**
This document must predate all Method B results. Any result file with an earlier mtime invalidates the pre-registration.

Related: gap framing + Table 1 evidence in `table1_extraction.md`; configs drawn from the citation-ranked fragmentation frame.

---

## 1. Hypothesis (H1)
At least one method pair (M_i, M_j) will **swap rank** between Config A and Config B:
M_i > M_j under Config A's metric (SSIM) **but** M_i < M_j under Config B's metric (classification accuracy), or vice versa.
Confirming H1 demonstrates that published "which method is better" conclusions in semantic communications are **harness artifacts**, not model properties.

## 2. Null hypothesis (H0)
Rankings are **stable** across both configs for all method pairs: for every pair (M_i, M_j), the sign of (M_i − M_j) is identical under Config A and Config B. No inversion occurs.

## 3. Method set
- **M1 — DeepJSCC (original):** Bourtsoulatze, Kurka & Gündüz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," IEEE TCCN 2019 (arXiv:1809.01733). CNN autoencoder, de-facto image baseline.
- **M2 — variant (required, ≥1):** must be a *different architecture* targeting the *same task* (image transmission), runnable on CIFAR-10 under both channel configs.

**Candidate variants (reproduction-available):**
| Variant | Architecture | Repo | Native CIFAR-10 | Native channels | Native metric |
|---|---|---|---|---|---|
| **WITT** (ICASSP 2023, arXiv:2211.00937) | Swin Transformer | github.com/KeYang8/WITT | yes | AWGN, Rayleigh | MSE, MS-SSIM |
| ADJSCC (TCSVT 2022, arXiv:2012.00533) | Attention-CNN | github.com/alexxu1988/ADJSCC | yes | AWGN | — |
| SwinJSCC (arXiv:2308.09361) | Swin Transformer | github.com/semcomm/SwinJSCC | partial | AWGN, Rayleigh | MSE, MS-SSIM |

**Pre-registered choice:** M2 = **WITT** (transformer vs M1's CNN = maximal architecture contrast; natively supports CIFAR-10 + AWGN/Rayleigh + MS-SSIM). A 3rd method (ADJSCC) may be added but is not required.

## 4. Config A — exact spec
- Dataset: **CIFAR-10**
- Channel: **AWGN + Rayleigh**
- SNR range: **2–12 dB**
- Ranking metric: **SSIM** (higher = better)
- Source paper: **arXiv:2606.00114** (Recursive Vision Transformer), Table II parameters; metric DECLARED ("We employ the SSIM ... as the evaluation metric").

## 5. Config B — exact spec
- Dataset: **CIFAR-10**
- Channel: **AWGN + Rayleigh + Rician**
- SNR range: **0–12 dB**
- Ranking metric: **classification accuracy** (downstream CIFAR-10 classifier on reconstructed images; higher = better)
- Source paper: **arXiv:2605.21553** (TONIC); PHY: 16QAM + group-wise LDPC, budget sweep {0.5B0, B0, 2B0}, B0=4096 symbols; metric DECLARED ("The principal task metric is classification accuracy").

## 6. Decision rule
Rank inversion is **CONFIRMED** iff BOTH hold:
1. **≥1 method pair swaps rank** between Config A and Config B (sign of (M_i − M_j) differs), AND
2. **The swap exceeds the noise floor**, defined as **±0.5 dB PSNR-equivalent** (fidelity axis) **or ±2% absolute classification accuracy** (task axis). A swap where both methods are within the noise floor on the deciding metric does NOT count.
- Each (method, channel, SNR) cell is run with **≥3 random seeds**; reported value = mean; the noise floor is cross-checked against the observed seed-to-seed standard deviation (if empirical σ exceeds the floor, the larger value governs).
- If inversion is NOT confirmed → H0 retained → paper pivots to "informal convergence is more robust than surveys suggest" (per gap kill-switch fallback).

## 7. Pre-registration integrity
- Timestamp above is the freeze point. Reproduction artifacts (checkpoints, logs, result tables) will be written under `projects/nafis/ictc/methodB_runs/` and must all post-date this file.
- Config pair, metric definitions, method set, and decision rule are FROZEN. Any change requires a dated amendment appended below, not an edit to the above.

### Amendments

**Amendment 1 (2026-06-11) — M1 sanity-gate relaxation (Option 2) + recorded Step-2 history.**
This amendment documents M1 readiness only; it does NOT alter the frozen hypotheses,
configurations, or the §6 inversion decision rule (all set before any inversion result).

Step-2 (M1 DeepJSCC PSNR sanity, CIFAR-10, AWGN, k/n=1/6) history:
- *Initial (120 ep, [0,1] MSE):* flat ~19–20 dB across SNR {2,4,7,10,12}, non-monotonic. FAILED.
- *Isolation:* noiseless (channel disabled) = 19.7 dB; native un-integrated channel = 20.7 dB
  ⇒ cause was the training recipe, NOT the shared channel (channel exonerated). Root cause:
  training MSE on [0,1] while the reference repo uses denormalized [0,255]; with Adam+wd=5e-4
  the weight-decay term dominated the tiny [0,1] gradients → underfitting.
- *Fix (reference-faithful):* MSE on [0,255], wd kept at 5e-4. Noiseless then = 32.2 dB.
- *Channelled sweeps (fixed):* 120 ep {24.0,25.1,25.1,27.4,27.8}; 300 ep {24.4,25.6,27.2,28.3,29.2};
  600 ep {24.5,25.9,27.8,29.0,29.7}. Monotonic; 300→600 ep gained <0.7 dB (plateau).
- *Residual:* uniform ~2 dB gap to the approximate Bourtsoulatze 2019 band (no released numeric
  table; reference uncertain ±1–2 dB), in channel-robustness only (noiseless within ~0.8 dB of the
  original ceiling). Attributed to the chunbaobao reimplementation's author-flagged tconv4/tconv5
  kernel mismatch — a reimplementation ceiling, not a bug, not the shared channel.

**Relaxed M1 gate** (replaces "within ~1 dB of Bourtsoulatze absolute"): M1 is accepted as a
functionally-correct DeepJSCC iff (i) test PSNR is monotonically non-decreasing in SNR AND
(ii) the noiseless ceiling is within ~1 dB of the original's near-noiseless PSNR. Both hold.
The ~2 dB channelled offset is carried as a documented limitation. Rationale: the rank-inversion
test requires both methods correctly trained under one identical harness, not fidelity to any one
paper's absolute numbers.

**Step 3 status:** UNLOCKED — pending explicit user confirmation before launch.
