# -*- coding: utf-8 -*-
"""make_montage38_colorfix.py — 38 块 GT vs PROC 对比图（显示层对称对齐版）。

与生产版 make_montage38.py 的唯一区别是"显示定色"这一段：
  生产版：GT 直接贴原始照片；PROC 用 unwhitebalance(proc, mu_face)。
  本版  ：两侧走同一条颜色管线
          A 模式 —— 两侧都只显示白平衡域；
          B 模式 —— 两侧都先 white_balance，再用同一样本 mu_face 还原（默认）。
指标栏数字直接沿用 reeval_orient_summary.csv（白平衡域口径，显示层改动不影响）。

用法：
  python make_montage38_colorfix.py --mode B --tag fixed
  python make_montage38_colorfix.py --mode compare --tag before_after
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

if os.name != "nt":
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: 0
    cv2.destroyAllWindows = lambda *a, **k: None

DIM, TILE = 256, 192
ANAT_EN = {"环孔材": "Ring-porous", "半环孔材": "Semi-ring-porous",
           "散孔材": "Diffuse-porous", "针叶材": "Conifer"}
LABEL_W, MET_W = 300, 290


def load_meta(code):
    met = {r["sample"]: r for r in csv.DictReader(
        open(os.path.join(code, "reeval_orient_summary.csv")))}
    spec = {}
    p = os.path.join(code, "species_map.csv")
    if os.path.isfile(p):
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            sp = (r["species"], r["anatomy"])
            if r["sample"] in met and (r["sample"], sp[0]) not in spec:
                spec[r["sample"]] = sp
    return met, spec


def fit_font(txt, width, base=0.52):
    """标题自适应缩放：长标题不越过画布宽度。"""
    s = base
    while s > 0.30 and cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, s, 1)[0][0] > width - 20:
        s -= 0.02
    return s


def row_canvas(imgs, label, label2, met_lines, heads, hdr_h=26):
    """imgs: list of float RGB (192x192) 已缩放；heads: 列标题。"""
    n = len(imgs)
    W = LABEL_W + TILE * n + MET_W
    H = TILE + 4
    row = 255 * np.ones((H, W, 3), np.uint8)
    x = LABEL_W
    for im, hd in zip(imgs, heads):
        t = cv2.resize(cv2.cvtColor((np.clip(im, 0, 1) * 255).astype(np.uint8),
                                    cv2.COLOR_RGB2BGR), (TILE, TILE))
        t = cv2.rectangle(t, (0, 0), (TILE - 1, TILE - 1), (0, 0, 0), 1)
        row[2:2 + TILE, x:x + TILE] = t
        cv2.putText(row, hd, (x + TILE // 2 - len(hd) * 5, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 3)
        cv2.putText(row, hd, (x + TILE // 2 - len(hd) * 5, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 0, 0), 1)
        x += TILE
    cv2.putText(row, label, (8, 40), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(row, label2, (8, 66), cv2.FONT_HERSHEY_DUPLEX, 0.40, (90, 90, 90), 1)
    for i, txt in enumerate(met_lines):
        cv2.putText(row, txt, (LABEL_W + TILE * n + 10, 52 + 26 * i),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 0, 0), 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", default="/home/gast/cjy/project1/moke_4a_holdout_v2_20261021")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "out"))
    ap.add_argument("--mode", default="B", choices=["B", "A", "cur", "compare"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--diag", default="", help="color_diag.csv，用于在图上标注 ΔE 收敛")
    args = ap.parse_args()

    code = os.path.join(args.prod, "code")
    for _p in (code, os.path.join(code, "COMMON")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import data_utils  # noqa: E402
    os.makedirs(args.out, exist_ok=True)

    met, spec = load_meta(code)
    diag = {}
    if args.diag and os.path.isfile(args.diag):
        for r in csv.DictReader(open(args.diag, encoding="utf-8")):
            diag[r["sample"]] = r

    cutA, cutD = data_utils.generate_cuboid_coordinates_cutsAD(DIM)
    ids = sorted(met.keys())

    mode = args.mode
    if mode == "compare":
        heads = ["GT (raw)", "PROC (old)", "GT (aligned)", "PROC (aligned)"]
    else:
        heads = ["GT", "PROC"]

    rows = []
    for sid in ids:
        folder = os.path.join(code, "Samples", sid)
        cut = int(met[sid]["cut"])
        gt = C.imread_rgb(os.path.join(folder, f"cut{cut}_col.png"), DIM)
        mu_face = C.face_chan_mean_fast(folder)
        proc = C.load_proc_slice(folder, sid, cut, cutA, cutD)
        gt_wb = C.white_balance(gt)
        proc_cur = np.clip(C.unwhitebalance(proc, mu_face), 0, 1)
        gt_align = np.clip(C.unwhitebalance(gt_wb, mu_face), 0, 1)
        proc_align = proc_cur  # unwhitebalance 已把均值锚到 mu_face

        if mode == "B":
            imgs = [gt_align, proc_align]
        elif mode == "A":
            imgs = [gt_wb, proc]
        elif mode == "cur":
            imgs = [gt, proc_cur]
        else:
            imgs = [gt, proc_cur, gt_align, proc_align]

        m = met[sid]
        f1 = m.get("proc_RingF1", "nan")
        f1s = f"{float(f1):.3f}" if f1 and f1 != "nan" else " n/a"
        sp = spec.get(sid, ("", ""))
        label = f"{sid} {sp[0]} [{ANAT_EN.get(sp[1], '')}]"
        label2 = f"cut{cut} {'rot180' if sid in C.FLIP180 else 'rot0'}"
        ml = [f"L1 {float(m['proc_L1']):.3f}  SWD2 {float(m['proc_VGG-SWD2']):.3f}",
              f"DISTS {float(m['proc_DISTS']):.3f}  F1 {f1s}"]
        if sid in diag:
            d = diag[sid]
            try:
                ml.append(f"dE_low {float(d['dE_low_cur']):.1f} -> {float(d['dE_low_B']):.1f}")
            except (KeyError, ValueError):
                pass
        rows.append((sid, sp[1] or "zz", row_canvas(imgs, label, label2, ml, heads)))
        print(f"  [{len(rows)}/{len(ids)}] {sid}", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    head = 255 * np.ones((44, rows[0][2].shape[1], 3), np.uint8)
    ttl = {"B": "one colour pipeline for both columns (white-balance + per-sample outer-face white point)",
           "A": "both columns shown in the white-balance domain",
           "cur": "legacy: raw GT photo vs unwhitebalance(PROC, face mean)",
           "compare": "left pair = legacy display; right pair = aligned display"}[mode]
    cv2.putText(head, f"Mokume 4a - 38 test blocks - {ttl}",
                (10, 30), cv2.FONT_HERSHEY_DUPLEX,
                fit_font(f"Mokume 4a - 38 test blocks - {ttl}", head.shape[1]),
                (0, 0, 0), 1)
    full = np.vstack([head] + [r[2] for r in rows])
    out = os.path.join(args.out, f"montage38_gt_vs_proc{suffix}.png")
    cv2.imwrite(out, full)
    print("saved", out, full.shape)

    for anat in ["环孔材", "半环孔材", "散孔材", "针叶材"]:
        sel = [r[2] for r in rows if r[1] == anat]
        if not sel:
            continue
        head = 255 * np.ones((44, sel[0].shape[1], 3), np.uint8)
        ttl_g = f"Mokume 4a - {ANAT_EN[anat]} ({len(sel)} blocks) - {ttl}"
        cv2.putText(head, ttl_g, (10, 30), cv2.FONT_HERSHEY_DUPLEX,
                    fit_font(ttl_g, head.shape[1], 0.50), (0, 0, 0), 1)
        fn = os.path.join(args.out, f"montage_{ANAT_EN[anat]}{suffix}.png")
        cv2.imwrite(fn, np.vstack([head] + sel))
        print("saved", fn)


if __name__ == "__main__":
    main()
