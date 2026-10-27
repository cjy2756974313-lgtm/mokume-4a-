# -*- coding: utf-8 -*-
"""check_rot180.py — 全 38 块 180° 朝向核查（评价几何 A4 项）。

背景：2026-10-22 发现 H11 节疤在渲染切面上与照片错位 180°；结构张量法
（check_orient.py）对年轮走向的 180° 不敏感，无法发现这类翻转。
本脚本对每块现有 proc_cube.npz 用与 4a_eval_cut.py 完全相同的渲染管线
输出 cut 切面，分别计算 rot0 与 rot180（二面群中 L1 最优的翻转）的白平衡
L1，报告差值。属评价几何核查：只用 GT 判定渲染朝向约定，不回传生成端。

输出：code/rot180_check.json（每块 rot0_L1 / rot180_L1 / delta / 建议）
"""
import os
import sys
import json

import numpy as np
import cv2

if os.name != "nt":
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: 0
    cv2.destroyAllWindows = lambda *a, **k: None

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "COMMON")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_utils

DIM = 256


def load_npz(folder, name):
    p = os.path.join(folder, name)
    return np.load(p)["arr_0"] if os.path.isfile(p) else None


def render_cut(vol, cut_coords, dim=256, rotate90=0):
    px = cut_coords.reshape(-1, 3).detach().cpu().numpy()
    idx = np.clip(255.0 * (px + 0.5), 0, dim - 1).astype(np.int64)
    s = vol[idx[:, 0], idx[:, 1], idx[:, 2]]
    if s.ndim == 1:
        s = np.repeat(s[:, None], 3, axis=1)
    img = s.astype(np.float32).reshape(dim, dim, 3)
    img = cv2.flip(img, 0)
    for _ in range(rotate90):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def white_balance(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    if x.max() > 2.0:
        x = x / 255.0
    mean = x.reshape(-1, 3).mean(axis=0)
    return np.clip(x / (2.0 * (mean + eps)), 0.0, 1.0)


def main():
    samples_root = os.path.join(HERE, "Samples")
    ids = sorted(d for d in os.listdir(samples_root)
                 if os.path.isdir(os.path.join(samples_root, d)))
    cutA, cutD = data_utils.generate_cuboid_coordinates_cutsAD(DIM)
    out = {}
    for sid in ids:
        folder = os.path.join(samples_root, sid)
        vol = load_npz(folder, "proc_cube.npz")
        cut = 2 if os.path.isfile(os.path.join(folder, "cut2_col.png")) else 1
        gt_path = os.path.join(folder, f"cut{cut}_col.png")
        if vol is None or not os.path.isfile(gt_path):
            out[sid] = {"skip": True}
            continue
        gt = cv2.imread(gt_path)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = cv2.resize(gt, (DIM, DIM), interpolation=cv2.INTER_AREA)
        gt_wb = white_balance(gt)

        coords = cutA if cut == 1 else cutD
        rot90 = 1 if cut == 1 else 0
        rend = render_cut(vol, coords, DIM, rotate90=rot90)
        a = white_balance(rend)
        l1_rot0 = float(np.abs(a - gt_wb).mean())
        a180 = np.rot90(a, 2)
        l1_rot180 = float(np.abs(a180 - gt_wb).mean())
        out[sid] = {
            "cut": cut,
            "rot0_L1": round(l1_rot0, 5),
            "rot180_L1": round(l1_rot180, 5),
            "delta": round(l1_rot0 - l1_rot180, 5),
            "rot180_better": bool(l1_rot180 < l1_rot0 - 0.002),
        }
        print(sid, out[sid])
    n_better = sum(1 for v in out.values() if v.get("rot180_better"))
    print(f"\nrot180 更优（delta>0.002）的样本数：{n_better}/{len(out)}")
    with open(os.path.join(HERE, "rot180_check.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
