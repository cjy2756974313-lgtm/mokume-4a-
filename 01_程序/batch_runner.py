# batch_runner.py — 对含斜切面的样本批量运行 Step 2/3/4a/eval（r740 服务器版）
# 用法: python batch_runner.py [start_idx] [end_idx] [workers]
# 由本地 D:\CODE\moke\mokumeproject-main\batch_runner.py 移植：
#   - PY / DATASET_DIR / TORCH_HOME 改为服务器路径
#   - 每个 worker 通过 CUDA_VISIBLE_DEVICES 轮询分配到 GPU 0/1（3090/4090），
#     并设 CUDA_DEVICE_ORDER=PCI_BUS_ID 使编号与 nvidia-smi 一致
# 其余断点续跑逻辑与本地版逐行一致。
import os, sys, glob, json, shutil, subprocess, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/gast/wzy/anaconda3/envs/mokume/bin/python"
DATASET_DIR = "/home/gast/cjy/project1/moku/data/dataset/MokumeDataset"
SAMPLES_DIR = os.path.join(HERE, "Samples")
LOG_DIR = os.path.join(HERE, "batch_logs")
os.makedirs(LOG_DIR, exist_ok=True)
TORCH_HOME = "/home/gast/.cache/torch"
GPUS = ["0", "1"]  # nvidia-smi: 0=RTX3090(14G free), 1=RTX4090(12G free)；2=4090D 已被他人占满，勿用

def find_cut_samples():
    out = []
    for d in sorted(os.listdir(DATASET_DIR)):
        sd = os.path.join(DATASET_DIR, d)
        if not os.path.isdir(sd):
            continue
        for cut in (1, 2):
            if os.path.isfile(os.path.join(sd, f"cut{cut}_col.png")):
                out.append((d, cut))
                break
    return out

def stage_inputs(sample_id):
    src = os.path.join(DATASET_DIR, sample_id)
    dst = os.path.join(SAMPLES_DIR, sample_id)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith("_col.png") or f.endswith("_knot.png"):
            tgt = os.path.join(dst, f)
            if not os.path.isfile(tgt):
                shutil.copy2(os.path.join(src, f), tgt)

def run_step(cmd, log_path, gpu=None):
    env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3",
               CUDA_DEVICE_ORDER="PCI_BUS_ID")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("\n=== " + " ".join(cmd) + f"  [GPU {gpu}] ===\n")
        lf.flush()
        r = subprocess.run(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return r.returncode

def done(path):
    return os.path.isfile(path)

def eval_done(path):
    # 仅当是 4a_eval_cut.py 写的完整报告（含 metrics 键）才算完成；
    # 4a_proc.py 内置 cut2 自检写的是扁平指标，不能跳过正式评价
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            return "metrics" in json.load(f)
    except Exception:
        return False

_sum_lock = threading.Lock()

def process_one(sid, cut, gpu):
    folder = os.path.join(SAMPLES_DIR, sid)
    log = os.path.join(LOG_DIR, f"{sid}.log")
    t0 = time.time()
    rec = {"id": sid, "cut": cut}
    try:
        stage_inputs(sid)
        if not done(os.path.join(folder, "A_arl-unet.png")):
            if run_step([PY, "2_apply_unet.py", "-sample", sid], log, gpu) != 0:
                raise RuntimeError("step2 failed")
        if not done(os.path.join(folder, "gf_params.pkl")):
            if run_step([PY, "3_infer_gf.py", "-id", sid, "-show", "0"], log, gpu) != 0:
                raise RuntimeError("step3 failed")
        if not done(os.path.join(folder, "proc_cube.npz")):
            if run_step([PY, "4a_proc.py", "-id", sid, "-res", "128",
                         "-show", "0", "-torch_home", TORCH_HOME], log, gpu) != 0:
                raise RuntimeError("step4a failed")
        if not eval_done(os.path.join(folder, "proc_eval.json")):
            if run_step([PY, "4a_eval_cut.py", "-id", sid, "-cut", str(cut),
                         "-torch_home", TORCH_HOME], log, gpu) != 0:
                raise RuntimeError("eval failed")
        with open(os.path.join(folder, "proc_eval.json"), encoding="utf-8") as f:
            rec["eval"] = json.load(f)
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = f"FAIL: {e}"
    rec["minutes"] = round((time.time() - t0) / 60, 1)
    with _sum_lock:
        sp = os.path.join(LOG_DIR, "summary.json")
        summary = []
        if os.path.isfile(sp):
            with open(sp, encoding="utf-8") as f:
                summary = json.load(f)
        summary = [r for r in summary if r["id"] != sid] + [rec]
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[{sid}] {rec['status']}  {rec['minutes']} min", flush=True)
    return rec

def main():
    samples = find_cut_samples()
    print(f"共发现 {len(samples)} 个含斜切面样本: {[s for s,_ in samples]}", flush=True)
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else len(samples)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    todo = samples[start:end]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_one, sid, cut, GPUS[i % len(GPUS)])
                for i, (sid, cut) in enumerate(todo)]
        for fu in as_completed(futs):
            fu.result()

if __name__ == "__main__":
    main()
