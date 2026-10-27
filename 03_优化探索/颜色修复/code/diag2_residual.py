# -*- coding: utf-8 -*-
"""diag2_residual.py — 残余色差拆解（针对 B 方案后仍偏大的样本）。

把低频 ΔE 拆成亮度分量与色度分量，判断残余是"色相/饱和"还是"明暗结构"，
并对每个样本导出一张 4 联图（GT 原图 | PROC 旧 | GT 对齐 | PROC 对齐）供目视核对。
"""
import os
import sys
import csv
import argparse

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import colorlib as C  # noqa: E402


def dE_split(x, y, sigma=16.0):
    xb = C.lowfreq(x, sigma)
    yb = C.lowfreq(y, sigma)
    lx, ly = C.lab(xb), C.lab(yb)
    dL = np.abs(lx[..., 0] - ly[..., 0])
    dAB = np.sqrt((lx[..., 1] - ly[..., 1]) ** 2 + (lx[..., 2] - ly[..., 2]) ** 2)
    dE = np.sqrt(dL ** 2 + dAB ** 2)
    # 色度差在总差里的能量占比（按平方和分解）
    return (float(dE.mean()), float(dL.mean()), float(dAB.mean()),
            float(dAB.mean() / (dE.mean() + 1e-9)))


def panel(imgs, heads, tile=384):
    h = tile + 22
    w = tile * len(imgs)
    out = 255 * np.ones((h, w, 3), np.uint8)
    for i, (im, hd) in enumerate(zip(imgs, heads)):
        t = cv2.resize(cv2.cvtColor((np.clip(im, 0, 1) * 255).astype(np.uint8),
                                    cv2.COLOR_RGB2BGR), (tile, tile))
        out[22:, i * tile:(i + 1) * tile] = t
        cv2.putText(out, hd, (i * tile + 6, 16), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    (0, 0, 0), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", default="/home/gast/cjy/project1/moke_4a_holdout_v2_20261021")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "out"))
    ap.add_argument("--samples", default="H11,K08,MZ03,B08,H06")
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
    for sid in [s.strip() for s in args.samples.split(",") if s.strip()]:
        if sid not in met:
            print("skip", sid)
            continue
        folder = os.path.join(code, "Samples", sid)
        cut = int(met[sid]["cut"])
        gt = C.imread_rgb(os.path.join(folder, f"cut{cut}_col.png"), C.DIM)
        mu_face = C.face_chan_mean_fast(folder)
        proc = C.load_proc_slice(folder, sid, cut, cutA, cutD)
        gt_wb = C.white_balance(gt)
        proc_old = np.clip(C.unwhitebalance(proc, mu_face), 0, 1)
        gt_al = np.clip(C.unwhitebalance(gt_wb, mu_face), 0, 1)

        e_cur = dE_split(gt, proc_old)
        e_al = dE_split(gt_al, proc_old)
        r = {"sample": sid, "cut": cut,
             "gt_Lmean": float(C.lab(gt)[..., 0].mean()),
             "proc_Lmean": float(C.lab(proc_old)[..., 0].mean()),
             "gt_Lstd": C.contrast_L(gt), "proc_Lstd": C.contrast_L(proc_old),
             "cur_dE": e_cur[0], "cur_dL": e_cur[1], "cur_dAB": e_cur[2],
             "al_dE": e_al[0], "al_dL": e_al[1], "al_dAB": e_al[2],
             "al_ABshare": e_al[3],
             "mu_face_R": mu_face[0] * 255, "mu_face_G": mu_face[1] * 255,
             "mu_face_B": mu_face[2] * 255}
        rows.append(r)
        fn = os.path.join(args.out, f"panel_{sid}_colorfix.png")
        cv2.imwrite(fn, panel([gt, proc_old, gt_al, proc_old],
                              ["GT (raw)", "PROC (old)", "GT (aligned)", "PROC (aligned)"]))
        print(f"  {sid}: cur dE={e_cur[0]:.2f} (dL {e_cur[1]:.2f}/dAB {e_cur[2]:.2f}) | "
              f"aligned dE={e_al[0]:.2f} (dL {e_al[1]:.2f}/dAB {e_al[2]:.2f}, "
              f"AB share {e_al[3] * 100:.0f}%) | Lstd gt {r['gt_Lstd']:.1f} "
              f"proc {r['proc_Lstd']:.1f}", flush=True)

    if rows:
        p = os.path.join(args.out, "residual_split.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
        print("saved", p)


if __name__ == "__main__":
    main()
