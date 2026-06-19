# Table 1 — Semantic Communications Evaluation-Harness Fragmentation

**Frame:** citation-ranked (Semantic Scholar), field-filtered to wireless semantic comms, surveys excluded.
**Selection rule:** 2022–23 top-15 by citations; 2024 top-10 by citations; 2025–26 top-10 by recency (citation-independent).
**Extraction rule:** full-text only; "not reported (NR)" = paper genuinely omits; "not fetched" = source inaccessible (PDF-only/paywalled), slot preserved, never skipped, never fabricated.

## Ranked candidate frame (locked 2026-06-06)

### 2022–2023 (top 15 by citations)
| # | Cit | Paper | arXiv |
|---|-----|-------|-------|
| 1 | 261 | Wireless Semantic Communications for Video Conferencing | 2204.07790 |
| 2 | 257 | Wireless Deep Video Semantic Transmission | 2205.13129 |
| 3 | 215 | Predictive and Adaptive Deep Coding for Wireless Image Transmission | (none) |
| 4 | 213 | Semantic-Preserved Communication System for Speech | 2205.12727 |
| 5 | 200 | Performance Optimization for Semantic Comms (Attention RL) | 2208.08239 |
| 6 | 183 | Task-Oriented Communications for 6G: Vision, Principles | 2303.10920 |
| 7 | 158 | Energy Efficient Semantic Communication Over Wireless | 2301.01987 |
| 8 | 156 | DRL-Driven Dynamic Resource Allocation Task-Oriented | (none) |
| 9 | 155 | Personalized Saliency in Task-Oriented SemCom | 2209.12274 |
| 10 | 153 | Transformer-Empowered 6G Intelligent Networks | 2205.03770 |
| 11 | 143 | Large AI Model Empowered Multimodal SemCom | 2309.01249 |
| 12 | 142 | Deep Joint Source-Channel Coding for Semantic Comms | 2211.08747 |
| 13 | 136 | Wireless End-to-End Image Transmission System | 2302.13721 |
| 14 | 123 | Semantic Communication Meets Edge Intelligence | 2202.06471 |
| 15 | 117 | Semantic Comms for Image Recovery and Classification | 2304.02317 |

### 2024 (top 10 by citations)
| # | Cit | Paper | arXiv |
|---|-----|-------|-------|
| 1 | 128 | Multi-Functional RIS-Assisted Semantic Anti-Jamming | (none) |
| 2 | 102 | Goal-Oriented and Semantic Communication in 6G AI-Native | 2402.07573 |
| 3 | 100 | Latency-Aware Generative Semantic Comms (Diffusion) | 2403.17256 |
| 4 | 66 | Multimodal and Multiuser Semantic Comms | (none) |
| 5 | 51 | Compression Ratio Allocation Probabilistic SemCom | 2403.00434 |
| 6 | 50 | Feature Importance-Aware Task-Oriented Sem Transmission | (none) |
| 7 | 47 | Toward Goal-Oriented Semantic Comms: New Metrics | (none) |
| 8 | 45 | Diffusion-Driven Semantic Communication | 2407.18468 |
| 9 | 42 | Task-Oriented SemCom over Rate Splitting | (none) |
| 10 | 40 | LLM Enabled Multi-Task Physical Layer Network | 2412.20772 |

### 2025–2026 (top 10 by recency)
| # | Paper | arXiv |
|---|-------|-------|
| 1 | Robust Multi-Branch SemCom in LAWN | (none) |
| 2 | Distortion-Aware UAV Placement Aerial Semantic Relay | 2606.04013 |
| 3 | Hybrid Bit and Semantic Comms for UAV | 2606.00668 |
| 4 | Recursive Vision Transformer Dynamic Depth | 2606.00114 |
| 5 | Hybrid transformer–zero-shot framework | (none) |
| 6 | TWIST: Closed-Loop Token Synchronization | 2605.27205 |
| 7 | DSRDM: Digital Signal Recovery Diffusion Model | 2605.27730 |
| 8 | Toward mobility-aware SemCom: DeepJSCC | (none) |
| 9 | TONIC: Token-Centric Semantic Communication | 2605.21553 |
| 10 | Perception-Aware Video Semantic Communication | 2605.19397 |

## Table 1 — RAW EXTRACTION (analysis deferred)

Legend: NR = field genuinely not reported in paper; "not fetched" = source inaccessible (no arXiv / conversion failed); "no harness" = vision/position paper with no original evaluation experiment.

### Stratum A: 2022–2023 (top 15 by citations)

| # | Paper (arXiv) | Domain | Dataset | Channel | SNR (dB) | Primary Metric | Secondary | Baseline(s) | Preproc? | DeepJSCC baseline? | SNR overlaps 1–13? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Video Conferencing (2204.07790) | Video | VoxCeleb | AWGN + Rayleigh | NR (BER 0–0.2) | Perceptual loss (VGG-19) | SSIM, AKD | H.264, RS-HARQ | Y | N | unknown |
| 2 | Deep Video Sem Transmission (2205.13129) | Video | Vimeo-90k, HEVC, UVG | AWGN | −2 to 14 | PSNR | MS-SSIM | H.264/H.265+LDPC, +Capacity | N | N | partial |
| 3 | Predictive & Adaptive Deep Coding | — | not fetched (no arXiv) | — | — | — | — | — | — | — | — |
| 4 | Speech SemCom (2205.12727) | Speech | LibriSpeech | AWGN + Rayleigh | 5 to 10 (train) | WER / MCD | sem-sim, MOS | DeepSC-SR, SE-DeepSC, DeepSC-S | Y | N | Y |
| 5 | Performance Opt. Attention-RL (2208.08239) | Text | NR | Rayleigh (OFDMA) | NR | MSS (semantic sim) | — | "standard network" | Y | N | unknown |
| 6 | Task-Oriented 6G Vision (2303.10920) | — | no harness (position) | — | — | — | — | — | — | — | — |
| 7 | Energy-Eff. Rate Splitting (2301.01987) | Text | NR | wireless net (rate-split) | NR | energy minimization | NR | NR | N | N | unknown |
| 8 | DRL Dynamic Resource Alloc | — | not fetched (no arXiv) | — | — | — | — | — | — | — | — |
| 9 | Personalized Saliency (2209.12274) | Image | STREET (59 imgs) | Fisher-Snedecor F fading | NR (P 1–3 kW) | utility function | user-match score | Naive-SemCom, OA-SemCom | Y | N | unknown |
| 10 | Transformer-Empowered 6G (2205.03770) | — | no harness (survey, quotes DeepSC) | — | — | — | — | — | — | — | — |
| 11 | Large AI Multimodal (2309.01249) | Multi | VOC2012, LibriSpeech, UCF101 | "fading" (unspecified) | 10 to 25 | transmission acc (cosine) | cosine sim | NR | N | N | partial |
| 12 | Deep JSCC SemCom (2211.08747) | Image | CIFAR-10, ImageNet, Kodak | AWGN | 0 to 20 | PSNR | MS-SSIM, LPIPS, Top-1 | BPG+polar, H.265+LDPC | NR | N (is proposed) | Y |
| 13 | Wireless E2E Image (2302.13721) | Image | COCO-Stuff | AWGN | 2.0 to 3.0 (Eb/No) | PSNR | compr. ratio, subj. MOS, BER | JPEG | N | N | partial(Eb/No) |
| 14 | SemCom Meets Edge Intelligence (2202.06471) | — | no harness (vision) | — | — | — | — | — | — | — | — |
| 15 | Image Recovery & Classification (2304.02317) | Image | MNIST, CIFAR-10 | AWGN + Rayleigh + Rician | 0, 21, −3 (+ domain rand.) | PSNR | classification acc | DeepJSCC-MSE, DeepJSCC-SSIM, SSCC(JPEG2000) | Y | **Y** | partial |

### Stratum B: 2024 (top 10 by citations)

| # | Paper (arXiv) | Domain | Dataset | Channel | SNR (dB) | Primary Metric | Secondary | Baseline(s) | Preproc? | DeepJSCC baseline? | SNR overlaps 1–13? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RIS Anti-Jamming SemCom (—) | — | not fetched (no arXiv) | — | — | — | — | — | — | — | — |
| 2 | Goal-Oriented 6G AI-Native / 6G-GOALS (2402.07573) | — | no harness (position) | — | — | — | — | — | — | — | — |
| 3 | Latency-Aware Generative SemCom (2403.17256) | Image(gen) | NR | Rayleigh block-fading | avg-SNR sweep (low region <12) | **multiple: CLIP/cosine, MS-SSIM, LPIPS, FID — no single primary declared** | — | NR | NR | N | partial |
| 4 | Multimodal Multiuser SemCom (—) | — | not fetched (no arXiv) | — | — | — | — | — | — | — | — |
| 5 | Compression-Ratio Alloc. Probabilistic (2403.00434) | Text(opt) | NR | AWGN (MISO) | NR | NR (no metric named in text) | NR | NR | N | N | unknown |
| 6 | Feature-Importance Task-Oriented (—) | — | not fetched (no arXiv) | — | — | — | — | — | — | — | — |
| 7 | Goal-Oriented New Metrics (—) | — | not fetched (no arXiv) | — | — | — | — | — | — | — | — |
| 8 | Diffusion-Driven SemCom (2407.18468) | Image | NR | AWGN + Rayleigh + MIMO | 3 to 12 | PSNR | SSIM, CLIP, FID, LPIPS | CDDM (DeepJSCC-integrated) | NR | **Y** (via CDDM) | Y |
| 9 | Task-Oriented over Rate Splitting (—) | — | not fetched (no arXiv) | — | — | — | — | — | — | — | — |
| 10 | LLM Multi-Task Physical Layer (2412.20772) | PHY/CSI | NR (CSI) | AWGN (MIMO-OFDM) | 5–20 (train), 0–25 (test) | NMSE | — | (NMSE baselines, NR names) | N | N | Y |

### Stratum C: 2025–2026 (top 10 by recency)

| # | Paper (arXiv) | Domain | Dataset | Channel | SNR (dB) | Primary Metric | Secondary | Baseline(s) | Preproc? | DeepJSCC baseline? | SNR overlaps 1–13? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Multi-Branch SemCom in LAWN (—) | — | not fetched (no arXiv) | — | — | — | — | — | — | — | — |
| 2 | Distortion-Aware UAV Placement (2606.04013) | Image | MNIST, FashionMNIST | AWGN (+ fading) | 1 to 16 | end-to-end distortion (MSE) | (PSNR/SSIM noted, not used) | CF-SQP, exhaustive search | NR | **Y** (codec) | Y |
| 3 | Hybrid Bit+Semantic UAV (2606.00668) | Image | NR | Rician + AWGN | NR (inst. SNR formula) | PSNR (+ latency) | — | JPEG | NR | N | unknown |
| 4 | Recursive Vision Transformer (2606.00114) | Image | CIFAR-10 | AWGN + Rayleigh | 2 to 12 | SSIM | — | TDC (BPG+LDPC+16QAM), ViT variants | NR | N | Y |
| 5 | Hybrid Transformer Zero-Shot (—) | — | not fetched (no arXiv) | — | — | — | — | — | — | — | — |
| 6 | TWIST Token Sync (2605.27205) | Token/cls | NR | AWGN + Rayleigh block-fading | wide sweep (fig) | macro-F1 | normalized cost | DeepSC, JPEG, LDPC+16QAM | NR | N | unknown |
| 7 | DSRDM Diffusion Signal Recovery (2605.27730) | Image | LSUN, ImageNet64, CIFAR-10 | AWGN + Rician | 0 to 20 | BER | MSE | 16/256-QAM, noise-match variants | NR | N | Y |
| 8 | Mobility-Aware DeepJSCC (—) | — | not fetched (no arXiv) | — | — | — | — | — | — | — | — |
| 9 | TONIC Token-Centric (2605.21553) | Image(cls) | CIFAR-10, ImageNet-100 | AWGN + Rayleigh + Rician | 0 to 12 | classification accuracy | task loss | separation, DeepJSCC, token-domain, JPEG | NR | **Y** | Y |
| 10 | Perception-Aware Video (PVSC) (2605.19397) | Video | Vimeo-90K, BVI-DVC; test HEVC/UVG/MCL-JCV | AWGN + Rayleigh block-fading | 0 and 10 | LPIPS + DISTS | PSNR | VTM+5G LDPC, H.265/HM, x264/x265, DVST | NR | N | partial |

## Extraction status: COMPLETE
- **Fetched + extracted:** 22 papers (11 Stratum A, 4 Stratum B, 7 Stratum C)
- **No original harness (vision/position):** 4 (A: 2303.10920, 2205.03770, 2202.06471; B: 2402.07573)
- **Not fetched (no arXiv / inaccessible):** 9 (A: 2× ; B: 5× ; C: 3×) — slots preserved, never fabricated
- Total frame slots: 35

### Raw observations (analysis NOT yet performed — pending user go-ahead)
- Preprocessing/normalization **specified in only ~5 of 22** fetched papers; **NR in the rest**.
- DeepJSCC used as an explicit baseline in only **5 of 22** (2304.02317, 2407.18468, 2606.04013, 2605.21553, + de-facto refs) — the "de-facto baseline" is inconsistently adopted even where applicable.
- Primary metric spans: PSNR, SSIM, MS-SSIM, LPIPS, DISTS, FID, CLIP/cosine, BLEU/SBERT, WER, MCD, BER, NMSE, macro-F1, classification accuracy, utility, energy — **>15 distinct primary metrics**.
- SNR ranges (where reported in dB): 1–16, 2–12, 3–12, 0–20, 0–12, −2–14, 0/10, 2–3 (Eb/No), 5–20, 0/21/−3 — heterogeneous, many non-overlapping; several papers report **no SNR in dB at all**.
- Most top-cited 2024 "semantic communication" papers are **not JSCC transmission** (resource allocation, PHY/CSI, vision) — the field's most-cited work does not share a task.

## Pilot extractions (convenience sample — NOT part of ranked Table 1, excluded from analysis)
These 4 were extracted before the Semantic-Scholar ranking was wired up; none fall in the citation frame. Kept only for the record, with user spot-check corrections applied:
- PJSCC (2411.10178): SNR = **variable/adaptive across channel models** (not a fixed 1–13 dB protocol; paper emphasizes channel-adaptive generalization).
- Semantic Channel Equalization (2510.04674): domain = **semantic alignment / heterogeneous TX–RX model interoperability** (different sub-task, not standard image transmission).
- Multi-hop DJSCC (2510.06868): dataset = **NUS-WIDE subset** — RE-VERIFIED verbatim from full text ("a subset of the NUS-WIDE dataset [16], consisting of 9,450:1,050:2,100 ... 256×256"). Original extraction was correct.
- Diffusion-Aided JSCC (2404.17736): primary metric = **multiple (FID, LPIPS, PSNR, MS-SSIM); no single primary declared**.

## Primary-Metric Audit (table-wide, all 21 fetched rows)
Rule: DECLARED = one metric explicitly named as the/primary/principal evaluation metric (or sole metric reported); MULTIPLE = ≥2 metrics, no declared hierarchy; NOT REPORTED = no quality/task metric (system-objective-only papers flagged here pending user ruling).
NOTE: "fetched" count is **21**, not 22 — earlier "11 Stratum A" was a miscount (A has 10 extracted rows).

**DECLARED (5):**
- 2606.04013 — MSE — *"we adopt MSE as the performance metric for analytical tractability"*
- 2606.00114 — SSIM — *"We employ the Structural Similarity Index Measure (SSIM) [24] as the evaluation metric to assess the visual fidelity"*
- 2605.27730 — BER — *"Thus, BER is the evaluation metric in the simulation."*
- 2605.21553 — classification accuracy — *"The principal task metric is classification accuracy"*
- 2309.01249 — transmission accuracy (sole metric, not labeled "primary") — *"we employ BERT and cosine similarity to evaluate the performance of the multimodal SC system"*

**MULTIPLE (13):** 2204.07790 (perceptual-loss/SSIM/AKD) · 2205.13129 (PSNR/MS-SSIM/mIoU) · 2205.12727 (WER/MCD/sent-sim/MOS) · 2208.08239 (MSS/BLEU) · 2211.08747 (PSNR/MS-SSIM/LPIPS/top-1/BLEU) · 2302.13721 (PSNR/compr-ratio/MOS/BER) · 2304.02317 (PSNR/classification-acc) · 2403.17256 (CLIP/MS-SSIM/LPIPS/FID) · 2407.18468 (PSNR/SSIM/LPIPS/FID/CLIP — "all employed") · 2412.20772 (NMSE/BER/accuracy) · 2606.00668 (PSNR/semantic-efficiency/latency) · 2605.27205 (macro-F1/cost) · 2605.19397 (LPIPS/DISTS/PSNR)

**OBJECTIVE-ONLY — optimizes a system objective incommensurable with fidelity/task metrics (3):** 2301.01987 (energy) · 2403.00434 (semantic rate) · 2209.12274 (utility function)

**Counts:** DECLARED 5 · MULTIPLE 13 · OBJECTIVE-ONLY 3 · NOT REPORTED 0 (= 21)

## QC note — systematic error mode identified
Spot-check (seed 20260606) found my recurring error is **over-attributing a single "primary metric"** when a paper declares none / lists several. Confirmed in 2403.17256 and 2403.00434 (corrected above). A table-wide audit of the Primary-Metric column is warranted before analysis. This over-attribution is itself a fragmentation signal (papers rarely declare one primary).
