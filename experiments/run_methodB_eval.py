"""GATE 3 — eval. Load best.pt for (method,config,seed); eval the config metric over the
TEST grid. For Config B, classifier sanity (>=0.91 clean) runs FIRST and aborts on failure.
  python run_methodB_eval.py --method {m1,m2} --config {A,B} --seed {0,1,2} --device cuda:0
"""
import os, sys, json, math, argparse, types
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from itertools import product

ROOT = "."
sys.path.insert(0, ROOT + "/methods/Deep-JSCC-PyTorch")
sys.path.insert(0, ROOT + "/methods/WITT")
sys.path.insert(0, ROOT + "/methods/shared")
from model import DeepJSCC
from net.network import WITT
from ssim_eval import ssim_batch
from classifier_pipeline import FrozenCIFARClassifier, assert_clean_accuracy

DATA, RUNDIR, BATCH = ROOT + "/data", ROOT + "/methods/methodB_runs", 256
CONFIGS = {
    'A': dict(channels=['awgn', 'rayleigh'],            test_snrs=[2, 4, 7, 10, 12], metric='ssim'),
    'B': dict(channels=['awgn', 'rayleigh', 'rician'], test_snrs=[0, 2, 4, 7, 10, 12], metric='acc'),
}

def build_m1(dev): return DeepJSCC(c=8, channel_type='awgn', snr=10).to(dev)

def build_m2(dev):
    C = 16
    args = types.SimpleNamespace(channel_type='awgn', multiple_snr='2', model='WITT',
                                 distortion_metric='MSE', trainset='CIFAR10', C=C)
    cfg = types.SimpleNamespace(
        encoder_kwargs=dict(img_size=(32, 32), patch_size=2, in_chans=3, embed_dims=[128, 256],
            depths=[2, 4], num_heads=[4, 8], C=C, window_size=2, mlp_ratio=4., qkv_bias=True,
            qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True),
        decoder_kwargs=dict(img_size=(32, 32), embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4],
            C=C, window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True),
        downsample=2, pass_channel=True, logger=None, norm=False, CUDA=True, device=dev)
    return WITT(args, cfg).to(dev)

def recon(model, is_m1, x, ch, snr):
    if is_m1:
        model.channel.channel_type = ch; model.channel.snr = float(snr)
        return model(x)
    model.channel.channel_type = ch
    return model(x, given_SNR=float(snr))[0]

def psnr_from_mse(m): return 10.0 * math.log10(1.0 / max(m, 1e-12))

@torch.no_grad()
def eval_grid(model, is_m1, loader, cfg, dev, clf=None):
    model.eval(); per = {}
    for ch, snr in product(cfg['channels'], cfg['test_snrs']):
        ss = mse = nb = 0.0; num = den = 0
        for x, y in loader:
            x = x.to(dev); r = recon(model, is_m1, x, ch, snr).clamp(0, 1)
            ss += ssim_batch(r, x).item() * x.size(0)
            mse += ((r - x) ** 2).mean(dim=[1, 2, 3]).sum().item(); nb += x.size(0)
            if clf is not None:
                num += (clf(r).argmax(1).cpu() == y).sum().item(); den += y.numel()
        per[f"{ch}@{snr}"] = dict(ssim=ss / nb, psnr=psnr_from_mse(mse / nb),
                                  acc=(num / den if clf is not None else None))
    agg = dict(per_point=per,
               mean_ssim=float(np.mean([v['ssim'] for v in per.values()])),
               mean_psnr=float(np.mean([v['psnr'] for v in per.values()])),
               mean_acc=(float(np.mean([v['acc'] for v in per.values()])) if clf is not None else None))
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=['m1', 'm2'])
    ap.add_argument("--config", required=True, choices=['A', 'B'])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); dev = torch.device(a.device)
    cfg = CONFIGS[a.config]; is_m1 = a.method == 'm1'
    tf = transforms.Compose([transforms.ToTensor()])
    te = datasets.CIFAR10(DATA, train=False, download=True, transform=tf)
    tel = DataLoader(te, BATCH, shuffle=False, num_workers=4)
    clf = None
    if cfg['metric'] == 'acc':
        clf = FrozenCIFARClassifier().to(dev)
        assert_clean_accuracy(clf, tel, device=str(dev), floor=0.91)   # ABORTS Config B if < 0.91
    model = build_m1(dev) if is_m1 else build_m2(dev)
    ckpt = f"{RUNDIR}/{a.method}_{a.config}_s{a.seed}/best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    res = eval_grid(model, is_m1, tel, cfg, dev, clf)
    json.dump(res, open(f"{RUNDIR}/{a.method}_{a.config}_s{a.seed}/eval.json", "w"), indent=2)
    key = 'mean_ssim' if cfg['metric'] == 'ssim' else 'mean_acc'
    print(f"EVAL {a.method} {a.config} s{a.seed}: {key}={res[key]:.4f} mean_psnr={res['mean_psnr']:.3f}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
