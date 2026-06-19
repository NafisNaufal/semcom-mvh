"""GATE 3 — decision. Aggregate eval.json across seeds {0,1,2}, apply the pre-registered
decision rule, emit CONFIRMED / NOT CONFIRMED with margins and per-seed values.
NOTHING enters the paper until the user confirms this verdict."""
import sys, json
import numpy as np

ROOT = "."
RUNDIR = ROOT + "/methods/methodB_runs"
SEEDS = [0, 1, 2]
ACC_FLOOR = 0.02      # 2% absolute accuracy (task axis)
PSNR_FLOOR = 0.5      # 0.5 dB PSNR-equivalent (fidelity axis, Config A)

def load(method, config, key):
    vals = []
    for s in SEEDS:
        r = json.load(open(f"{RUNDIR}/{method}_{config}_s{s}/eval.json"))
        vals.append(r[key])
    return np.array(vals, dtype=float)

def ms(a):  # mean, sample std (ddof=1)
    return float(a.mean()), float(a.std(ddof=1))

def pooled_std(a, b):
    return float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))

def main():
    # Config A — primary metric SSIM (+ PSNR for the fidelity-axis floor)
    ssimA = {'M1': load('m1', 'A', 'mean_ssim'), 'M2': load('m2', 'A', 'mean_ssim')}
    psnrA = {'M1': load('m1', 'A', 'mean_psnr'), 'M2': load('m2', 'A', 'mean_psnr')}
    # Config B — primary metric accuracy
    accB = {'M1': load('m1', 'B', 'mean_acc'), 'M2': load('m2', 'B', 'mean_acc')}

    print("=== PER-SEED ===")
    for name, d in [('SSIM-A', ssimA), ('PSNR-A', psnrA), ('ACC-B', accB)]:
        for m in ['M1', 'M2']:
            print(f"  {name} {m}: {np.round(d[m],4).tolist()}  mean={d[m].mean():.4f} std={d[m].std(ddof=1):.4f}")

    winA = 'M1' if ssimA['M1'].mean() > ssimA['M2'].mean() else 'M2'
    winB = 'M1' if accB['M1'].mean() > accB['M2'].mean() else 'M2'

    ssim_margin = abs(ssimA['M1'].mean() - ssimA['M2'].mean())
    psnr_margin = abs(psnrA['M1'].mean() - psnrA['M2'].mean())
    acc_margin = abs(accB['M1'].mean() - accB['M2'].mean())

    floorA = max(2 * pooled_std(ssimA['M1'], ssimA['M2']), 0.0)   # SSIM-axis: 2x pooled seed std
    floorB = max(ACC_FLOOR, 2 * pooled_std(accB['M1'], accB['M2']))

    # Config A significant iff SSIM margin > 2x seed std AND PSNR margin > 0.5 dB (fidelity floor)
    A_sig = (ssim_margin > floorA) and (psnr_margin > PSNR_FLOOR)
    B_sig = (acc_margin > floorB)
    inversion = (winA != winB) and A_sig and B_sig

    print("\n=== DECISION ===")
    print(f"  [SSIM-A-M1]={ssimA['M1'].mean():.4f}+/-{ssimA['M1'].std(ddof=1):.4f}  "
          f"[SSIM-A-M2]={ssimA['M2'].mean():.4f}+/-{ssimA['M2'].std(ddof=1):.4f}  winner_A={winA}")
    print(f"  [ACC-B-M1]={accB['M1'].mean():.4f}+/-{accB['M1'].std(ddof=1):.4f}  "
          f"[ACC-B-M2]={accB['M2'].mean():.4f}+/-{accB['M2'].std(ddof=1):.4f}  winner_B={winB}")
    print(f"  Config A: SSIM margin={ssim_margin:.4f} (floor {floorA:.4f}), PSNR margin={psnr_margin:.3f} dB "
          f"(floor {PSNR_FLOOR}) -> significant={A_sig}")
    print(f"  Config B: ACC margin={acc_margin:.4f} (floor {floorB:.4f}) -> significant={B_sig}")
    print(f"\n  VERDICT: {'INVERSION CONFIRMED' if inversion else 'NOT CONFIRMED'}")
    print(f"  (winner flips A->B: {winA != winB}; both margins significant: {A_sig and B_sig})")

    json.dump(dict(
        winner_A=winA, winner_B=winB, inversion=bool(inversion),
        ssim_A={m: ms(ssimA[m]) for m in ssimA}, psnr_A={m: ms(psnrA[m]) for m in psnrA},
        acc_B={m: ms(accB[m]) for m in accB},
        ssim_margin=ssim_margin, psnr_margin=psnr_margin, acc_margin=acc_margin,
        floorA=floorA, floorB=floorB, A_significant=bool(A_sig), B_significant=bool(B_sig),
    ), open(f"{RUNDIR}/decision.json", "w"), indent=2)
    print("\nWrote decision.json")

if __name__ == "__main__":
    main()
