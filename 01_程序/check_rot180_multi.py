# -*- coding: utf-8 -*-
"""check_rot180_multi.py — 多立方体交叉验证 180° 朝向偏好（方法无关性检查）。

对每块样本，分别计算 gf_cube / col_cube / proc_cube / nca_cube 在目标切面上
rot0 vs rot180 的白平衡 L1 差（delta>0 表示 rot180 更优），并给出共识：
- consensus: 所有可用立方体 delta 符号一致且 |delta|>0.001 的多数方向
- 意义：若结构来源（gf，仅含生长场无颜色细节）与颜色来源（col）与 PROC
  偏好一致，说明翻转是"体坐标系 vs 照片拍摄朝向"的几何问题，与方法无关，
  朝向表可以安全地被所有方法共用。
输出：code/rot180_multi.json
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
        cut = 2 if os.path.isfile(os.path.join(folder, "cut2_col.png")) else 1
        gt_path = os.path.join(folder, f"cut{cut}_col.png")
        if not os.path.isfile(gt_path):
            continue
        gt = cv2.imread(gt_path)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = cv2.resize(gt, (DIM, DIM), interpolation=cv2.INTER_AREA)
        gt_wb = white_balance(gt)

        coords = cutA if cut == 1 else cutD
        rot90 = 1 if cut == 1 else 0
        rec = {"cut": cut}
        prefs = []
        for nm, fn in [("gf", "gf_cube.npz"), ("col", "col_cube.npz"),
                       ("proc", "proc_cube.npz"), ("nca", "nca_cube.npz")]:
            vol = load_npz(folder, fn)
            if vol is None:
                continue
            rend = render_cut(vol, coords, DIM, rotate90=rot90)
            a = white_balance(rend)
            l0 = float(np.abs(a - gt_wb).mean())
            l180 = float(np.abs(np.rot90(a, 2) - gt_wb).mean())
            rec[nm] = {"rot0": round(l0, 5), "rot180": round(l180, 5),
                       "delta": round(l0 - l180, 5)}
            if abs(l0 - l180) > 0.001:
                prefs.append("r180" if l180 < l0 else "r0")
        if prefs:
            n180 = prefs.count("r180")
            rec["votes"] = f"{n180}/{len(prefs)} r180"
            if n180 == len(prefs):
                rec["consensus"] = "rot180"
            elif n180 == 0:
                rec["consensus"] = "rot0"
            else:
                rec["consensus"] = "mixed"
        out[sid] = rec
        print(sid, rec)
    with open(os.path.join(HERE, "rot180_multi.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    n180 = sum(1 for v in out.values() if v.get("consensus") == "rot180")
    print(f"\n共识 rot180 的样本：{n180}")


if __name__ == "__main__":
    main()
