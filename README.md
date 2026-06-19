# semcom-mvh

Code and paper for **“Fragmented but Not Flipped: A Controlled Re-Evaluation of
Semantic-Communication Methods and a Minimal Viable Harness.”**

## Summary

Semantic-communication (SemCom) papers evaluate methods with mutually incompatible
harnesses. We:

1. **Quantify the fragmentation** across a citation-ranked sample of 21 papers
   (35-paper frame): 31 distinct quality metrics (21 used by a single paper), an
   **empty** intersection of SNR ranges across image-domain papers, no preprocessing
   specified in 15/21, and DeepJSCC used as an explicit baseline in only 3/11 image
   papers. (`analysis/table1_extraction.md`)
2. **Pre-register and run a controlled re-evaluation** of two methods — DeepJSCC (CNN)
   and WITT (Swin-transformer) — under two harness configurations drawn verbatim from
   surveyed papers, one ranked by SSIM and one by downstream classification accuracy,
   through a single shared channel. (`analysis/method_b_preregistration.md`)
3. **Result — rank inversion NOT CONFIRMED.** The transformer dominates the CNN under
   *both* configurations, with margins well beyond the pre-registered noise floor. For
   this pair, fragmentation’s risk is **unverifiable incomparability**, not silent rank
   inversion. (`analysis/results/decision.json`)
4. **Propose and release a Minimal Viable Harness (MVH)** — the `harness/` code — that
   restores comparability and lets others test whether the result generalizes.

## Layout

```
paper/        IEEE conference paper (main.tex, main.bib, IEEEtran.*, main.pdf)
harness/      MVH reference implementation:
              shared_channel.py        one shared channel (AWGN/Rayleigh/Rician, per-symbol
                                       complex fading, ZF equalization) for BOTH methods
              classifier_pipeline.py   frozen CIFAR-10 classifier (task metric)
              ssim_eval.py             plain SSIM (fidelity metric)
              analog_approx_channel_note.md
experiments/  Method-B training/eval/decision scripts + GPU orchestrator
analysis/     35-row fragmentation frame, pre-registration, Step-3 design, results/
```

## Reproduce

1. `pip install -r requirements.txt` (Python 3.11).
2. The two evaluated methods are third-party; clone them and apply the one-line
   shared-channel override per **`experiments/INTEGRATION.md`**.
3. Train: `python experiments/run_methodB_train.py --method {m1,m2} --config {A,B} --seed {0,1,2}`
   (or `experiments/orchestrate_methodB.py` to schedule all 12 across GPUs).
4. Evaluate: `python experiments/run_methodB_eval.py ...`; then
   `python experiments/run_methodB_decision.py` for the verdict.

**Note:** scripts use a project-relative root (`.`); `orchestrate_methodB.py` assumes
`micromamba` on `PATH`. Adjust to your environment.

## Citing

Please cite the paper (see `paper/main.bib`). The two evaluated methods retain their
own licenses; this repository redistributes only our harness, scripts, and results.
