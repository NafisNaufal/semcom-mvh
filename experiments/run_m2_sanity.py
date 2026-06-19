"""GATE 1 — M2 (WITT) sanity. AWGN, SNR-adaptive 2-12 dB, C=16 (CBR=1/6),
[0,255] MSE loss, lr=1e-4, 300 epochs. Eval PSNR at {2,7,12} + noiseless."""
import sys, types, time, math, random, json, argparse
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

WITT_DIR = "./methods/WITT"
sys.path.insert(0, WITT_DIR)
from net.network import WITT

DEVICE, DATA = "cuda:0", "./data"
EPOCHS, BATCH, SEED, LR, WD, C = 300, 128, 42, 1e-4, 5e-4, 16
SNRS = list(range(2, 13))   # SNR-adaptive over 2..12 dB
torch.backends.cudnn.benchmark = True

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def psnr(m): return 10.0 * math.log10(1.0 / max(m, 1e-12))

def make_net():
    args = types.SimpleNamespace(channel_type='awgn', multiple_snr=",".join(map(str, SNRS)),
                                 model='WITT', distortion_metric='MSE', trainset='CIFAR10', C=C)
    cfg = types.SimpleNamespace(
        encoder_kwargs=dict(img_size=(32, 32), patch_size=2, in_chans=3, embed_dims=[128, 256],
            depths=[2, 4], num_heads=[4, 8], C=C, window_size=2, mlp_ratio=4., qkv_bias=True,
            qk_scale=None, norm_layer=nn.LayerNorm, patch_norm=True),
        decoder_kwargs=dict(img_size=(32, 32), embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4],
            C=C, window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True),
        downsample=2, pass_channel=True, logger=None, norm=False, CUDA=True, device=torch.device(DEVICE))
    return WITT(args, cfg).to(DEVICE)

@torch.no_grad()
def evaluate(net, loader, snr, noiseless=False):
    net.eval(); orig = net.pass_channel
    if noiseless: net.pass_channel = False
    mse = 0.0; n = 0
    for x, _ in loader:
        x = x.to(DEVICE)
        recon = net(x, given_SNR=snr)[0].clamp(0, 1)
        mse += ((recon - x) ** 2).mean(dim=[1, 2, 3]).sum().item(); n += x.size(0)
    net.pass_channel = orig
    return psnr(mse / n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--tag", default="m2_sanity")
    a = ap.parse_args()
    set_seed(SEED)
    tf = transforms.Compose([transforms.ToTensor()])
    tr = datasets.CIFAR10(DATA, train=True,  download=True, transform=tf)
    te = datasets.CIFAR10(DATA, train=False, download=True, transform=tf)
    trl = DataLoader(tr, BATCH, shuffle=True,  num_workers=4, drop_last=True)
    tel = DataLoader(te, 256,   shuffle=False, num_workers=4)
    net = make_net()
    print(f"WITT C={C}, expected CBR=C/96={C/96:.4f} (target 1/6={1/6:.4f}); epochs={a.epochs}", flush=True)
    opt = optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        for x, _ in trl:
            x = x.to(DEVICE); opt.zero_grad()
            recon, CBR, SNR, mse, loss_G = net(x)        # SNR sampled internally from SNRS
            loss_G.backward(); opt.step()
        if ep == 0:
            print(f"epoch0 {time.time()-t0:.0f}s | measured CBR={float(CBR):.4f}", flush=True)
    res = {}
    for s in [2, 7, 12]:
        res[str(s)] = round(evaluate(net, tel, s), 3)
        print(f"PSNR_RESULT snr{s} {res[str(s)]}", flush=True)
    res["noiseless"] = round(evaluate(net, tel, 12, noiseless=True), 3)
    print(f"PSNR_RESULT noiseless {res['noiseless']}", flush=True)
    print(f"total {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(f"./methods/{a.tag}.json", "w"), indent=2)
    print("SWEEP " + json.dumps(res), flush=True); print("DONE", flush=True)

main()
