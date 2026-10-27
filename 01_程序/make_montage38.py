# -*- coding: utf-8 -*-
"""make_montage38.py — 38 块统一口径 GT vs PROC 评价对比总览图。

每块一行：[样本ID/材种 | GT 斜切面 | PROC 渲染（朝向表口径，H11 用 knot 体）]
底部标注该块 L1 / VGG-SWD2 / DISTS / 生长带F1。
输出：code/montage38_gt_vs_proc.png（38 行大图）+ 分组织类型分组的 4 张子图。

显示口径（2026-10-25 起，B 方案）：GT 与 PROC 走同一条颜色管线——
两侧都先 white_balance，再用**同一样本**的外表面均值 mu_face 还原。
动机与指标：低频 ΔE 11.07→1.93（−82.6%，36/38 块 ≤3），38 块指标逐位不变。
详见 03_优化探索/颜色修复/color_fix_summary.md。
"""
import os
import sys
import csv

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
TILE = 192
FLIP180 = {"H11", "H14", "MP07", "N07", "SG15", "SG16"}


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


def unwhitebalance(img, chan_mean):
    img = np.asarray(img, dtype=np.float32)
    m = img.reshape(-1, 3).mean(axis=0)
    return img / (m + 1e-6) * np.asarray(chan_mean, dtype=np.float32)


def face_chan_mean(folder, letters="ABCDEF"):
    means = []
    for ltr in letters:
        fp = os.path.join(folder, f"{ltr}_col.png")
        if os.path.isfile(fp):
            im = cv2.imread(fp)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            means.append(im.astype(np.float32).reshape(-1, 3).mean(axis=0) / 255.0)
    return np.mean(means, axis=0) if means else np.array([0.5, 0.5, 0.5], np.float32)


def main():
    root = os.path.join(HERE, "Samples")
    # 指标与材种
    met = {r["sample"]: r for r in csv.DictReader(
        open(os.path.join(HERE, "reeval_orient_summary.csv")))}
    spec = {}
    try:
        for r in csv.DictReader(open(os.path.join(HERE, "species_map.csv"),
                                     encoding="utf-8-sig")):
            spec[r["sample"]] = (r["species"], r["anatomy"])
    except FileNotFoundError:
        pass

    ids = sorted(met.keys())
    cutA, cutD = data_utils.generate_cuboid_coordinates_cutsAD(DIM)

    rows = []
    for sid in ids:
        folder = os.path.join(root, sid)
        cut = int(met[sid]["cut"])
        gt = cv2.imread(os.path.join(folder, f"cut{cut}_col.png"))
        gt = cv2.resize(gt, (DIM, DIM), interpolation=cv2.INTER_AREA)  # BGR
        fn = ("proc_cube_knot.npz"
              if sid == "H11" and os.path.isfile(
                  os.path.join(folder, "proc_cube_knot.npz"))
              else "proc_cube.npz")
        vol = load_npz(folder, fn)
        rend = render_cut(vol, cutA if cut == 1 else cutD, DIM,
                          rotate90=1 if cut == 1 else 0)
        if sid in FLIP180:
            rend = np.ascontiguousarray(np.rot90(rend, 2))
        # —— B 方案：两侧共用同一个 mu_face ——
        mu_face = face_chan_mean(folder)
        shown = unwhitebalance(rend, mu_face)          # PROC：行为与旧版一致
        shown = np.clip(255 * shown, 0, 255).astype(np.uint8)
        # GT：先白平衡，再用同一 mu_face 还原到显示域（RGB 空间），再转回 BGR
        gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt_al = unwhitebalance(white_balance(gt_rgb), mu_face)
        gt_disp = cv2.cvtColor(np.clip(255 * gt_al, 0, 255).astype(np.uint8),
                               cv2.COLOR_RGB2BGR)

        m = met[sid]
        f1 = m.get("proc_RingF1", "nan")
        f1s = f"{float(f1):.3f}" if f1 and f1 != "nan" else " n/a"
        sp = spec.get(sid, ("", ""))
        ANAT_EN = {"环孔材": "ring-porous", "半环孔材": "semi-ring",
                   "散孔材": "diffuse", "针叶材": "conifer"}
        label = (f"{sid} {sp[0]} [{ANAT_EN.get(sp[1], '')}]")
        label2 = f"cut{cut} {'rot180' if sid in FLIP180 else 'rot0'}"
        sub1 = (f"L1 {float(m['proc_L1']):.3f}   SWD2 {float(m['proc_VGG-SWD2']):.3f}")
        sub2 = (f"DISTS {float(m['proc_DISTS']):.3f}   F1 {f1s}")

        # 行画布: [标签240 | GT 192 | PROC 192 | 指标栏260]
        W = 230 + TILE * 2 + 200
        H = TILE + 4
        row = 255 * np.ones((H, W, 3), np.uint8)
        gt_t = cv2.resize(gt_disp, (TILE, TILE))
        pr_t = cv2.resize(cv2.cvtColor(shown, cv2.COLOR_RGB2BGR), (TILE, TILE))
        # 分隔线
        gt_t = cv2.rectangle(gt_t, (0, 0), (TILE - 1, TILE - 1), (0, 0, 0), 1)
        pr_t = cv2.rectangle(pr_t, (0, 0), (TILE - 1, TILE - 1), (0, 0, 0), 1)
        X0 = 230
        row[2:2 + TILE, X0:X0 + TILE] = gt_t
        row[2:2 + TILE, X0 + TILE:X0 + 2 * TILE] = pr_t
        cv2.putText(row, label, (8, 52), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1)
        cv2.putText(row, label2, (8, 76), cv2.FONT_HERSHEY_DUPLEX, 0.42, (90, 90, 90), 1)
        for dx, txt in ((X0 + TILE // 2 - 14, "GT"), (X0 + TILE + TILE // 2 - 26, "PROC")):
            cv2.putText(row, txt, (dx, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 3)
            cv2.putText(row, txt, (dx, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
        # 指标（两行）
        XM = X0 + 2 * TILE + 10
        cv2.putText(row, sub1, (XM, TILE // 2 - 12), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 0, 0), 1)
        cv2.putText(row, sub2, (XM, TILE // 2 + 14), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 0, 0), 1)
        rows.append((sid, sp[1] if sp[1] else "zz", row))
        print(f"  [{len(rows)}/{len(ids)}] {sid} done", flush=True)

    # 总图：按样本 ID 排序
    head = 255 * np.ones((44, rows[0][2].shape[1], 3), np.uint8)
    cv2.putText(head, "Mokume 4a PROC vs GT - 38 test blocks (orientation-corrected, H11=knot)",
                (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
    full = np.vstack([head] + [r[2] for r in rows])
    out = os.path.join(HERE, "montage38_gt_vs_proc.png")
    cv2.imwrite(out, full)
    print("saved", out, full.shape)

    # 分组子图（按组织类型）
    ANAT_EN2 = {"环孔材": "Ring-porous", "半环孔材": "Semi-ring-porous",
                "散孔材": "Diffuse-porous", "针叶材": "Conifer"}
    for anat in ["环孔材", "半环孔材", "散孔材", "针叶材"]:
        sel = [r[2] for r in rows if r[1] == anat]
        if not sel:
            continue
        head = 255 * np.ones((44, sel[0].shape[1], 3), np.uint8)
        cv2.putText(head, f"Mokume 4a PROC vs GT - {ANAT_EN2[anat]} ({len(sel)} blocks)",
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        sub_full = np.vstack([head] + sel)
        fn = os.path.join(HERE, f"montage_{anat}.png")
        cv2.imwrite(fn, sub_full)
        print("saved", fn, sub_full.shape)


if __name__ == "__main__":
    main()
