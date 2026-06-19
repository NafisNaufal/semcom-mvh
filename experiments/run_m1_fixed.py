"""M1 with the reference-faithful training recipe: MSE on [0,255], Adam lr=1e-3, wd=5e-4.
--mode noiseless : channel disabled (identity), one model.
--mode sweep     : AWGN SNR {2,4,7,10,12} via the integrated shared channel."""
import sys, json, time, math, random, argparse
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO = "./methods/Deep-JSCC-PyTorch"
sys.path.insert(0, REPO)
from model import DeepJSCC, ratio2filtersize

DEVICE, DATA = "cuda:0", "./data"
RATIO, EPOCHS, BATCH, SEED, LR, WD = 1/6, 120, 128, 42, 1e-3, 5e-4

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def psnr(m): return 10.0 * math.log10(1.0 / max(m, 1e-12))

def loaders():
    tf = transforms.Compose([transforms.ToTensor()])
    tr = datasets.CIFAR10(DATA, train=True,  download=True, transform=tf)
    te = datasets.CIFAR10(DATA, train=False, download=True, transform=tf)
    return (DataLoader(tr, BATCH, shuffle=True, num_workers=4, drop_last=True),
            DataLoader(te, 256, shuffle=False, num_workers=4), tr[0][0])

def train_eval(model, trl, tel, epochs):
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.MSELoss(); t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, _ in trl:
            x = x.to(DEVICE); opt.zero_grad()
            out = model(x)
            loss = crit(out * 255.0, x * 255.0)        # [0,255] MSE — matches reference repo
            loss.backward(); opt.step()
    model.eval(); mse = 0.0; n = 0
    with torch.no_grad():
        for x, _ in tel:
            x = x.to(DEVICE); out = model(x).clamp(0, 1)
            mse += ((out - x) ** 2).mean(dim=[1, 2, 3]).sum().item(); n += x.size(0)
    return mse / n, time.time() - t0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=["noiseless", "sweep"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args(); set_seed(SEED)
    trl, tel, sample = loaders()
    c = ratio2filtersize(sample, RATIO)
    print(f"FIXED recipe: [0,255] MSE, lr={LR}, wd={WD}, epochs={a.epochs}, c={c} (k/n=1/6)", flush=True)
    if a.mode == "noiseless":
        set_seed(SEED)
        mse, dt = train_eval(DeepJSCC(c=c), trl, tel, a.epochs)   # snr=None -> identity
        ps = psnr(mse)
        print(f"PSNR_RESULT noiseless {ps:.3f}  (MSE {mse:.6f}, {dt:.0f}s)", flush=True)
    else:
        res = {}
        for snr in [2, 4, 7, 10, 12]:
            set_seed(SEED)
            mse, dt = train_eval(DeepJSCC(c=c, channel_type='AWGN', snr=snr), trl, tel, a.epochs)
            ps = psnr(mse); res[str(snr)] = round(ps, 3)
            print(f"PSNR_RESULT snr{snr} {ps:.3f}  (MSE {mse:.6f}, {dt:.0f}s)", flush=True)
        json.dump(res, open("./methods/m1_sanity_fixed300.json", "w"), indent=2)
        print("SWEEP " + json.dumps(res), flush=True)
    print("DONE", flush=True)

main()
