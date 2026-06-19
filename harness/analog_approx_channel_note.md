# PATCH 4 — Analog-Symbol Approximation for Config B (methodology note)

**Status:** documentation only (no code). Records the deliberate simplification
of Config B's physical layer and its justification, for the paper's methodology
and limitations sections.

## What the approximation is
Config B's source paper, TONIC (arXiv:2605.21553), specifies a **digital** PHY:
16-QAM modulation + group-wise LDPC channel coding, with a symbol-budget sweep
(B0 = 4096 symbols; {0.5·B0, B0, 2·B0}). We **do not** reproduce that digital
stack. Instead we use an **analog-symbol approximation**: the encoder's
continuous-valued latent symbols are transmitted directly over the shared
complex channel (`shared_channel.py`) — i.e., the analog joint source–channel
coding (JSCC) regime — with the symbol budget enforced as the latent dimension
(channel-bandwidth ratio) rather than via QAM+LDPC bits. The metric
(classification accuracy, via `classifier_pipeline.py`) is unchanged.

## Why it was chosen over faithful 16-QAM + LDPC
1. **Both methods are natively analog JSCC.** M1 (DeepJSCC) and M2 (WITT) emit
   continuous symbols; neither repo ships a 16-QAM+LDPC pipeline. Bolting on a
   digital stack would require non-trivial new code applied unequally to two
   different codebases — the opposite of the "one identical channel" requirement.
2. **Isolation of the variables under test.** The rank-inversion experiment
   manipulates exactly three harness axes (channel model, SNR range, primary
   metric). Holding the PHY in the analog-JSCC regime for both methods keeps
   those three axes the only differences between Config A and Config B; adding a
   digital constellation/coding layer would introduce confounds.
3. **Comparability with Config A.** Config A (2606.00114) is analog-symbol
   transmission; using the same regime for Config B keeps the two configs
   differing only on the pre-registered axes.

## Generalizability limitation (one sentence)
Because Config B uses analog-symbol transmission rather than TONIC's 16-QAM+LDPC
PHY, any observed rank inversion is established for the analog-JSCC regime and
may not transfer unchanged to digital, constellation-constrained deployments.

## Papers in Table 1 using the same analog-symbol approximation
- **arXiv:2211.08747** (Deep Joint Source-Channel Coding for Semantic
  Communications) — analog JSCC; continuous symbols mapped directly to the
  channel (no digital constellation/coding).
- **arXiv:2606.00114** (Recursive Vision Transformer; our Config A source) —
  analog transmission, `y = g·x + n`, continuous symbols over the channel.
