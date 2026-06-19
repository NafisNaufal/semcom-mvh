"""GATE 2 orchestrator. Dispatches the remaining Method-B training jobs under the user's
gating rules: M1 needs >=2GB free + <=10% util; M2 needs >=4GB free + <=10% util; <=2 concurrent
M2; M1 jobs first. m1_A_s0 is excluded (launched/verified manually). Logs every launch/complete.
Polls every 5 min (more responsive than 30 min so device 0 dispatches promptly between ~25-min M1
jobs; mem+util thresholds still strictly enforced). Exits when all managed jobs finish."""
import subprocess, time, os
from datetime import datetime

ROOT = "."
RUNDIR = ROOT + "/methods/methodB_runs"
MM = "micromamba"
ENV = {**os.environ, "MAMBA_ROOT_PREFIX": "${MAMBA_ROOT_PREFIX}"}
POLL = 300
M1 = [('m1','A',1), ('m1','A',2), ('m1','B',0), ('m1','B',1), ('m1','B',2)]   # m1_A_s0 done manually
M2 = [('m2','A',0), ('m2','A',1), ('m2','A',2), ('m2','B',0), ('m2','B',1), ('m2','B',2)]
QUEUE = M1 + M2

def tag(j): return f"{j[0]}_{j[1]}_s{j[2]}"
def thresh(j): return 4096 if j[0] == 'm2' else 2048
def done(j): return os.path.exists(f"{RUNDIR}/{tag(j)}/run.json")
def log(m): print(f"[{datetime.now().isoformat(timespec='seconds')}] {m}", flush=True)

def gpus():
    out = subprocess.check_output(["nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]).decode()
    res = []
    for ln in out.strip().splitlines():
        i, f, u = [x.strip() for x in ln.split(',')]
        res.append((int(i), int(f), int(u)))
    return res

def launch(j, gpu, free, util):
    t = tag(j); os.makedirs(f"{RUNDIR}/{t}", exist_ok=True)
    lf = open(f"{RUNDIR}/{t}.log", "w")
    env = {**ENV, "CUDA_VISIBLE_DEVICES": str(gpu)}
    p = subprocess.Popen([MM, "run", "-n", "ictc", "python", "methods/run_methodB_train.py",
        "--method", j[0], "--config", j[1], "--seed", str(j[2]), "--device", "cuda:0"],
        cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    log(f"[LAUNCH] {t} device={gpu} free={free}MiB util={util}% pid={p.pid}")
    return p

def main():
    running = {}                                   # tag -> (proc, gpu)
    pending = [j for j in QUEUE if not done(j)]
    log(f"orchestrator start; pending={[tag(j) for j in pending]}")
    while pending or running:
        for t, (p, gpu) in list(running.items()):
            if p.poll() is not None:
                ok = os.path.exists(f"{RUNDIR}/{t}/run.json")
                log(f"[COMPLETE] {t} rc={p.returncode} best_ok={ok}")
                del running[t]
        if not pending and not running:
            break
        occupied = {gpu for (_, gpu) in running.values()}
        m2_running = sum(1 for t in running if t.startswith('m2'))
        gl = gpus()
        for j in list(pending):
            if tag(j) in running:
                continue
            if j[0] == 'm2' and m2_running >= 2:
                continue
            for (i, f, u) in gl:
                if i in occupied:
                    continue
                if f >= thresh(j) and u <= 10:
                    p = launch(j, i, f, u); running[tag(j)] = (p, i)
                    pending.remove(j); occupied.add(i)
                    if j[0] == 'm2':
                        m2_running += 1
                    break
        time.sleep(POLL)
    log("[ALL_DONE] orchestrator finished")

if __name__ == "__main__":
    main()
