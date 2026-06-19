"""GATE 2 — Method B full training. One (method, config, seed) per invocation.
D1 constant-MSE[0,255] objective; D2 SNR+channel-adaptive sampling; D4 45k/5k val split,
best-on-val checkpointing. Usage:
  python run_methodB_train.py --method {m1,m2} --config {A,B} --seed {0,1,2} --device cuda:0
"""
import os, sys, json, time, math, random, argparse, types
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from itertools import product

ROOT = "."
sys.path.insert(0, ROOT + "/methods/Deep-JSCC-PyTorch")
sys.path.insert(0, ROOT + "/methods/WITT")
sys.path.insert(0, ROOT + "/methods/shared")
from model import DeepJSCC                       # integrated SharedChannel
from net.network import WITT                     # integrated WITTChannelAdapter
from ssim_eval import ssim_batch
from classifier_pipeline import FrozenCIFARClassifier

DATA, RUNDIR, BATCH, VAL_EVERY = ROOT + "/data", ROOT + "/methods/methodB_runs", 128, 10
CONFIGS = {
    'A': dict(channels=['awgn', 'rayleigh'],            lo=2, hi=12, metric='ssim', val_snrs=[2, 7, 12]),
    'B': dict(channels=['awgn', 'rayleigh', 'rician'], lo=0, hi=12, metric='acc',  val_snrs=[0, 6, 12]),
}
HP = {'m1': dict(lr=1e-3, wd=5e-4, epochs=300), 'm2': dict(lr=1e-4, wd=5e-4, epochs=300)}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def split_train_val():
    tf = transforms.Compose([transforms.ToTensor()])
    full = datasets.CIFAR10(DATA, train=True, download=True, transform=tf)
    g = torch.Generator().manual_seed(1234)            # FIXED split — identical across all runs
    perm = torch.randperm(len(full), generator=g).tolist()
    return Subset(full, perm[5000:]), Subset(full, perm[:5000])   # 45k train, 5k val

def build_m1(dev):
    return DeepJSCC(c=8, channel_type='awgn', snr=10).to(dev)     # c=8 -> k/n=1/6

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
    """One forward through the shared channel at channel `ch`, SNR `snr` (dB)."""
    if is_m1:
        model.channel.channel_type = ch; model.channel.snr = float(snr)
        return model(x)                                # DeepJSCC sigmoid -> [0,1]
    model.channel.channel_type = ch
    return model(x, given_SNR=float(snr))[0]           # WITT decoder output

@torch.no_grad()
def val_metric(model, is_m1, loader, cfg, dev, clf=None):
    model.eval(); grid_vals = []
    for ch, snr in product(cfg['channels'], cfg['val_snrs']):
        ss = nb = 0.0; num = den = 0
        for x, y in loader:
            x = x.to(dev); r = recon(model, is_m1, x, ch, snr).clamp(0, 1)
            if cfg['metric'] == 'ssim':
                ss += ssim_batch(r, x).item() * x.size(0); nb += x.size(0)
            else:
                num += (clf(r).argmax(1).cpu() == y).sum().item(); den += y.numel()
        grid_vals.append(ss / nb if cfg['metric'] == 'ssim' else num / den)
    return float(np.mean(grid_vals))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=['m1', 'm2'])
    ap.add_argument("--config", required=True, choices=['A', 'B'])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); dev = torch.device(a.device)
    cfg, hp = CONFIGS[a.config], HP[a.method]
    set_seed(a.seed)
    tr_ds, val_ds = split_train_val()
    trl = DataLoader(tr_ds, BATCH, shuffle=True, num_workers=4, drop_last=True)
    vall = DataLoader(val_ds, 256, shuffle=False, num_workers=4)
    is_m1 = a.method == 'm1'
    model = build_m1(dev) if is_m1 else build_m2(dev)
    clf = FrozenCIFARClassifier().to(dev) if cfg['metric'] == 'acc' else None
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
    out = f"{RUNDIR}/{a.method}_{a.config}_s{a.seed}"; os.makedirs(out, exist_ok=True)
    print(f"[{a.method} {a.config} s{a.seed}] dev={a.device} cfg={cfg} hp={hp}", flush=True)
    best, best_ep, hist, t0 = -1.0, -1, [], time.time()
    for ep in range(hp['epochs']):
        model.train()
        for x, _ in trl:
            x = x.to(dev)
            ch = random.choice(cfg['channels']); snr = random.uniform(cfg['lo'], cfg['hi'])
            opt.zero_grad()
            r = recon(model, is_m1, x, ch, snr)
            loss = crit(r.clamp(0, 1) * 255.0, x * 255.0)      # D1: MSE on [0,255], both methods
            loss.backward(); opt.step()
        if ep % VAL_EVERY == 0 or ep == hp['epochs'] - 1:
            vm = val_metric(model, is_m1, vall, cfg, dev, clf); hist.append({'epoch': ep, 'val': vm})
            if vm > best:
                best, best_ep = vm, ep; torch.save(model.state_dict(), f"{out}/best.pt")
            print(f"  ep{ep} val_{cfg['metric']}={vm:.4f} best={best:.4f}@{best_ep} {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), f"{out}/last.pt")
    json.dump(dict(method=a.method, config=a.config, seed=a.seed, best_epoch=best_ep,
                   best_val=best, metric=cfg['metric'], history=hist,
                   minutes=round((time.time() - t0) / 60, 1)), open(f"{out}/run.json", "w"), indent=2)
    print(f"DONE best_val={best:.4f}@{best_ep}", flush=True)

if __name__ == "__main__":
    main()
