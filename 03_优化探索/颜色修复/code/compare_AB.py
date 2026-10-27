# -*- coding: utf-8 -*-
"""compare_AB.py — 同一批数据上交叉对比 A（两侧白平衡域）与 B（两侧同管线）两套显示方案。

三个维度：
  1) 收敛：dE_low(GT 显示, PROC 显示)          —— 越小越"看起来一致"
  2) 保真：dE_low(GT 显示, GT 原始照片)        —— 越小越"没改原图"
  3) 木质感：平均饱和度 / 平均色度 相对原始照片的保留率
只用六外表面估计白点，不读任何目标切面统计做定色。
"""
import os
import sys
import csv
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import colorlib as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", default="/home/gast/cjy/project1/moke_4a_holdout_v2_20261021")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "out"))
    ap.add_argument("--sigma", type=float, default=16.0)
    args = ap.parse_args()

    code = os.path.join(args.prod, "code")
    for _p in (code, os.path.join(code, "COMMON")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import data_utils  # noqa: E402
    os.makedirs(args.out, exist_ok=True)

    met = {r["sample"]: r for r in csv.DictReader(
        open(os.path.join(code, "reeval_orient_summary.csv")))}
    cutA, cutD = data_utils.generate_cuboid_coordinates_cutsAD(C.DIM)

    rows = []
    for sid in sorted(met):
        folder = os.path.join(code, "Samples", sid)
        cut = int(met[sid]["cut"])
        gt = C.imread_rgb(os.path.join(folder, f"cut{cut}_col.png"), C.DIM)
        mu = C.face_chan_mean_fast(folder)
        proc = C.load_proc_slice(folder, sid, cut, cutA, cutD)
        gt_wb = C.white_balance(gt)

        gtA, prA = gt_wb, proc
        gtB = np.clip(C.unwhitebalance(gt_wb, mu), 0, 1)
        prB = np.clip(C.unwhitebalance(proc, mu), 0, 1)

        r = {
            "sample": sid, "cut": cut,
            # 1) 收敛
            "dE_AB_A": C.de_lowfreq(gtA, prA, args.sigma),
            "dE_AB_B": C.de_lowfreq(gtB, prB, args.sigma),
            # 2) 保真（显示出的 GT 偏离真实照片多少）
            "dE_gtA_raw": C.de_lowfreq(gtA, gt, args.sigma),
            "dE_gtB_raw": C.de_lowfreq(gtB, gt, args.sigma),
            # 3) 饱和度/色度保留率（显示 GT 相对原始照片）
            "sat_raw": C.mean_sat(gt),
            "sat_A": C.mean_sat(gtA),
            "sat_B": C.mean_sat(gtB),
            "chr_raw": C.chroma_hue(gt)[0],
            "chr_A": C.chroma_hue(gtA)[0],
            "chr_B": C.chroma_hue(gtB)[0],
            # PROC 侧显示亮度/饱和度
            "proc_sat_A": C.mean_sat(prA),
            "proc_sat_B": C.mean_sat(prB),
        }
        r["sat_ret_A"] = r["sat_A"] / max(r["sat_raw"], 1e-6)
        r["sat_ret_B"] = r["sat_B"] / max(r["sat_raw"], 1e-6)
        r["chr_ret_A"] = r["chr_A"] / max(r["chr_raw"], 1e-6)
        r["chr_ret_B"] = r["chr_B"] / max(r["chr_raw"], 1e-6)
        rows.append(r)
        print(f"  {sid}: conv A {r['dE_AB_A']:.2f} / B {r['dE_AB_B']:.2f} | "
              f"fidelity A {r['dE_gtA_raw']:.2f} / B {r['dE_gtB_raw']:.2f} | "
              f"sat_ret A {r['sat_ret_A']:.2f} / B {r['sat_ret_B']:.2f}", flush=True)

    p = os.path.join(args.out, "AB_compare.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    def s(k):
        v = np.array([r[k] for r in rows], np.float64)
        return v.mean(), np.median(v), v.max()

    print("\n== 收敛 dE_low（越小越好）")
    for k in ("dE_AB_A", "dE_AB_B"):
        m, md, mx = s(k)
        print(f"  {k}: mean {m:.3f} / median {md:.3f} / max {mx:.3f}")
    winB = sum(1 for r in rows if r["dE_AB_B"] < r["dE_AB_A"])
    print(f"  逐块胜出：B {winB}/38，A {38 - winB}/38")

    print("\n== GT 保真 dE_low（显示 GT vs 原始照片，越小越好）")
    for k in ("dE_gtA_raw", "dE_gtB_raw"):
        m, md, mx = s(k)
        print(f"  {k}: mean {m:.3f} / median {md:.3f} / max {mx:.3f}")

    print("\n== 饱和度/色度保留率（相对原始照片，越接近 1 越好）")
    for k in ("sat_ret_A", "sat_ret_B", "chr_ret_A", "chr_ret_B"):
        m, md, mx = s(k)
        print(f"  {k}: mean {m:.3f} / median {md:.3f} / min {'%.3f' % min(r[k] for r in rows)}")
    print("\nsaved", p)


if __name__ == "__main__":
    main()
