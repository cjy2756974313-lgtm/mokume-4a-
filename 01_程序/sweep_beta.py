# -*- coding: utf-8 -*-
"""
sweep_beta.py —— 扫描黑色噪点衰减强度 beta，记录每个 beta 下的斜切面指标。

对每个 beta：在原拟合参数上缩放 pore_color / ray_multiply_color → 重建体纹理 → 评价。
原始体（beta=1）备份为 proc_cube_beta1_backup.npz。

用法：
  python sweep_beta.py -id BW06 -cut 2 -betas 1,0.75,0.5,0.25,0
"""
import os
import sys
import json
import time
import argparse
import pickle
import shutil
import subprocess

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "COMMON")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import importlib.util
spec = importlib.util.spec_from_file_location(
    "proc4a", os.path.join(HERE, "4a_proc.py"))
proc4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proc4a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-id", type=str, required=True)
    ap.add_argument("-cut", type=int, default=2, choices=[1, 2])
    ap.add_argument("-betas", type=str, default="1,0.75,0.5,0.25,0")
    ap.add_argument("-torch_home", type=str, default="D:/CODE/moke/hub")
    args = ap.parse_args()

    folder = os.path.join(HERE, "Samples", args.id)
    params_path = os.path.join(folder, "proc_params.pkl")
    cube_path = os.path.join(folder, "proc_cube.npz")
    backup_path = os.path.join(folder, "proc_cube_beta1_backup.npz")

    with open(params_path, "rb") as f:
        base = pickle.load(f)
    pc0 = base.pore_color.detach().clone()
    rc0 = base.ray_multiply_color.detach().clone()

    # 备份原始体
    if not os.path.isfile(backup_path):
        shutil.copy2(cube_path, backup_path)

    out = {"id": args.id, "cut": args.cut,
           "pore_color": pc0.tolist(), "ray_multiply_color": rc0.tolist(),
           "sweep": []}
    betas = [float(b) for b in args.betas.split(",")]

    for b in betas:
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        with torch.no_grad():
            params.pore_color = (pc0 * b).requires_grad_(False)
            params.ray_multiply_color = (rc0 * b).requires_grad_(False)
        t0 = time.time()
        with torch.no_grad():
            vol = proc4a.build_volume(params, dim=256)
        vm = float(np.asarray(vol, dtype=np.float32).reshape(-1, 3).mean())
        if abs(vm - 0.5) > 0.02:
            vol = np.clip(np.asarray(vol, dtype=np.float32) * (0.5 / vm),
                          0, 1).astype(np.float16)
        np.savez_compressed(cube_path, vol)
        subprocess.run([sys.executable, os.path.join(HERE, "4a_eval_cut.py"),
                        "-id", args.id, "-cut", str(args.cut),
                        "-torch_home", args.torch_home],
                       cwd=HERE, stdout=subprocess.DEVNULL)
        with open(os.path.join(folder, "proc_eval.json"), encoding="utf-8") as f:
            rep = json.load(f)
        m = rep["metrics"]["PROC (4a 程序纹理)"]
        col = rep["metrics"]["col_cube (仅颜色)"]
        rec = {"beta": b,
               "L1": round(m["L1"], 4), "RMSE": round(m["RMSE"], 4),
               "Gram": round(m["Gram-VGG19"], 6),
               "SWD2": round(m["VGG-SWD2"], 4),
               "col_SWD2": round(col["VGG-SWD2"], 4),
               "sec": round(time.time() - t0, 1)}
        out["sweep"].append(rec)
        print(f"  beta={b:<5} SWD2={rec['SWD2']:.4f}  Gram={rec['Gram']:.5f}"
              f"  L1={rec['L1']:.4f}  (col SWD2={rec['col_SWD2']:.4f})", flush=True)

    best = min(out["sweep"], key=lambda r: r["SWD2"])
    out["best_beta_by_SWD2"] = best["beta"]
    print(f"\n最优 beta（按 SWD2）= {best['beta']}  SWD2={best['SWD2']:.4f}")

    # 扫描循环会把体数据和 proc_eval.json 留在最后一个 beta 上。
    # 这里统一还原到「基准 beta=1」（纯拟合结果），避免污染后续评价与批量汇总。
    print("  还原体数据到基准 beta=1 ...", flush=True)
    with torch.no_grad():
        vol = proc4a.build_volume(base, dim=256)
    vm = float(np.asarray(vol, dtype=np.float32).reshape(-1, 3).mean())
    if abs(vm - 0.5) > 0.02:
        vol = np.clip(np.asarray(vol, dtype=np.float32) * (0.5 / vm),
                      0, 1).astype(np.float16)
    np.savez_compressed(cube_path, vol)
    subprocess.run([sys.executable, os.path.join(HERE, "4a_eval_cut.py"),
                    "-id", args.id, "-cut", str(args.cut),
                    "-torch_home", args.torch_home],
                   cwd=HERE, stdout=subprocess.DEVNULL)

    sp = os.path.join(HERE, "beta_sweep.json")
    data = []
    if os.path.isfile(sp):
        with open(sp, encoding="utf-8") as f:
            data = json.load(f)
    data = [d for d in data if d["id"] != args.id] + [out]
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"扫描结果已记录 -> {sp}")


if __name__ == "__main__":
    main()
