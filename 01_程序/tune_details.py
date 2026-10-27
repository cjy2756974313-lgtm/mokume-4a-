# -*- coding: utf-8 -*-
"""
tune_details.py —— 在已拟合的 4a 参数上衰减「黑色噪点」（管孔/射线暗点）强度，
重新生成体纹理。不重新优化，仅对细节暗化做后处理缩放，秒级完成。

原理（procedural_wood_function.py）：
  pore:  factor = clamp(1 - pore_color * mask, 0.5, 1.0)   —— pore_color 越大越黑
  ray :  factor = clamp(1 - ray_multiply_color * mask, 0.5, 1.0)
本脚本把这两个颜色乘以 beta（0<beta<1），黑色噪点按比例变浅；beta=1 还原。

用法：
  python tune_details.py -id BW06 -cut 2 -beta 0.5
"""
import os
import sys
import argparse
import pickle
import shutil

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "COMMON")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 直接加载 4a_proc（其 argparse 在 main() 里，顶层 import 安全）
import importlib.util
spec = importlib.util.spec_from_file_location(
    "proc4a", os.path.join(HERE, "4a_proc.py"))
proc4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proc4a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-id", type=str, required=True)
    ap.add_argument("-cut", type=int, default=2, choices=[1, 2])
    ap.add_argument("-beta", type=float, default=0.5,
                    help="黑色噪点暗度缩放（0=完全去掉管孔/射线暗点，1=原样）")
    ap.add_argument("-torch_home", type=str, default="D:/CODE/moke/hub")
    args = ap.parse_args()

    folder = os.path.join(HERE, "Samples", args.id)
    params_path = os.path.join(folder, "proc_params.pkl")
    cube_path = os.path.join(folder, "proc_cube.npz")
    backup_path = os.path.join(folder, "proc_cube_beta1_backup.npz")

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    pc = params.pore_color.detach().clone()
    rc = params.ray_multiply_color.detach().clone()
    print(f"原始 pore_color = {pc.tolist()}")
    print(f"原始 ray_multiply_color = {rc.tolist()}")

    with torch.no_grad():
        params.pore_color = (pc * args.beta).requires_grad_(False)
        params.ray_multiply_color = (rc * args.beta).requires_grad_(False)

    # 备份原始体（首次）
    if not os.path.isfile(backup_path):
        shutil.copy2(cube_path, backup_path)
        print(f"已备份原始体 -> {backup_path}")

    print(f"\n按 beta={args.beta} 重新生成 256^3 体纹理 ...")
    import time
    t0 = time.time()
    with torch.no_grad():
        vol = proc4a.build_volume(params, dim=256)
    print(f"  用时 {time.time() - t0:.1f}s")

    vm = float(np.asarray(vol, dtype=np.float32).reshape(-1, 3).mean())
    print(f"  白平衡域均值 = {vm:.4f}（目标 0.5）")
    if abs(vm - 0.5) > 0.02:
        vol = np.clip(np.asarray(vol, dtype=np.float32) * (0.5 / vm), 0, 1).astype(np.float16)
        print("  [已锚定校正]")

    np.savez_compressed(cube_path, vol)
    print(f"  已保存 {cube_path}（原始体备份在 {os.path.basename(backup_path)}）")

    # 保存调整版参数副本（不动原 proc_params.pkl）
    tuned_path = os.path.join(folder, f"proc_params_beta{args.beta:g}.pkl")
    params.detach_tensors()
    with open(tuned_path, "wb") as f:
        pickle.dump(params, f)
    print(f"  已保存调整版参数 -> {tuned_path}")

    # 立即重跑斜切面评价，刷新对比图与指标
    cmd = [sys.executable, os.path.join(HERE, "4a_eval_cut.py"),
           "-id", args.id, "-cut", str(args.cut),
           "-torch_home", args.torch_home]
    import subprocess
    subprocess.run(cmd, cwd=HERE)


if __name__ == "__main__":
    main()
