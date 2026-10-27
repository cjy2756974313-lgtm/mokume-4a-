# -*- coding: utf-8 -*-
"""
check_orient.py —— 用结构张量测量木纹走向，检验生成结果与 GT 切面照片的方向一致性。

输出每个样本：GT / col_cube(Step3 纯颜色) / proc_cube(4a) 的年轮级与细节级走向，
以及 PROC 相对 GT 的方向偏差（0-90 度）。

用法：python check_orient.py
"""
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "COMMON")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import torch  # noqa
import data_utils  # noqa


def st_angle(img, sigma_g=2.0, sigma_w=9.0):
    """返回主纹理走向（streak 方向，0-179 度）。"""
    g = img.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(2)
    g = cv2.GaussianBlur(g, (0, 0), sigma_g)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma_w)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma_w)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma_w)
    A = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)          # 梯度方向 mod pi
    E = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    idx = ((A + np.pi / 2) / np.pi * 180).astype(np.int32) % 180
    hist = np.bincount(idx.ravel(), weights=E.ravel(), minlength=180)
    return int(np.argmax(hist))


def render_cut(vol, coords, dim=256, rotate90=0):
    px = coords.reshape(-1, 3).detach().cpu().numpy()
    i = np.clip(255.0 * (px + 0.5), 0, dim - 1).astype(np.int64)
    s = vol[i[:, 0], i[:, 1], i[:, 2]]
    if s.ndim == 1:
        s = np.repeat(s[:, None], 3, axis=1)
    img = cv2.flip(s.astype(np.float32).reshape(dim, dim, 3), 0)
    for _ in range(rotate90):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def angdiff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def main():
    cutA, cutD = data_utils.generate_cuboid_coordinates_cutsAD(256)
    # 自动发现所有已完成（含 proc_cube 与切面照片）的样本
    samples = []
    for sid in sorted(os.listdir(os.path.join(HERE, "Samples"))):
        folder = os.path.join(HERE, "Samples", sid)
        if not os.path.isdir(folder):
            continue
        if not os.path.isfile(os.path.join(folder, "proc_cube.npz")):
            continue
        for cut in (1, 2):
            if os.path.isfile(os.path.join(folder, f"cut{cut}_col.png")):
                samples.append((sid, cut))
    print(f"{'样本':6}{'cut':4} | {'GT年轮':>7}{'col年轮':>8}{'PROC年轮':>9}"
          f" | {'GT细节':>7}{'col细节':>8}{'PROC细节':>9} | 偏差(年轮/细节)")
    print("-" * 88)
    for sid, cut in samples:
        folder = os.path.join(HERE, "Samples", sid)
        p = os.path.join(folder, f"cut{cut}_col.png")
        if not os.path.isfile(p):
            continue
        gt = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        gt = cv2.resize(gt, (256, 256))
        coords = cutA if cut == 1 else cutD
        rot = 1 if cut == 1 else 0
        out = {}
        for tag, fn in (("col", "col_cube.npz"), ("proc", "proc_cube.npz")):
            fp = os.path.join(folder, fn)
            if os.path.isfile(fp):
                v = np.load(fp)["arr_0"]
                out[tag] = render_cut(v, coords, rotate90=rot)
        a_gt_r = st_angle(gt, 6.0, 25.0)
        a_gt_f = st_angle(gt, 1.5, 5.0)
        a_col_r = st_angle(out["col"], 6.0, 25.0) if "col" in out else -1
        a_col_f = st_angle(out["col"], 1.5, 5.0) if "col" in out else -1
        a_p_r = st_angle(out["proc"], 6.0, 25.0) if "proc" in out else -1
        a_p_f = st_angle(out["proc"], 1.5, 5.0) if "proc" in out else -1
        print(f"{sid:6}{cut:<4} | {a_gt_r:>7}{a_col_r:>8}{a_p_r:>9}"
              f" | {a_gt_f:>7}{a_col_f:>8}{a_p_f:>9}"
              f" | {angdiff(a_gt_r, a_p_r):>3}/{angdiff(a_gt_f, a_p_f):<3}")


if __name__ == "__main__":
    main()
