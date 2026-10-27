# -*- coding: utf-8 -*-
"""diag_color.py — 38 块 GT 斜切面照片 vs PROC 渲染 的显示色差诊断（不重跑 GPU）。

做三件事：
1) 量化"GT 原始照片的通道均值"与"六外表面均值"的错位（现行 unwhitebalance 的锚点）；
2) 量化 PROC 切片在白平衡域的饱和度/对比与 GT 的差距（洗白问题）；
3) 在显示层试算 4 种对齐方案，用低频 ΔE（对结构免疫）给出收敛幅度。

输出：out/color_diag.csv + out/color_diag_summary.txt
"""
import os
import sys
import csv
import json
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

    prod = args.prod
    code = os.path.join(prod, "code")
    for _p in (code, os.path.join(code, "COMMON")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import data_utils  # noqa: E402
    os.makedirs(args.out, exist_ok=True)

    met = {r["sample"]: r for r in csv.DictReader(
        open(os.path.join(code, "reeval_orient_summary.csv")))}
    spec = {}
    sp_path = os.path.join(code, "species_map.csv")
    if os.path.isfile(sp_path):
        for r in csv.DictReader(open(sp_path, encoding="utf-8-sig")):
            spec[r["sample"]] = (r["species"], r["anatomy"])

    ids = sorted(met.keys())
    cutA, cutD = data_utils.generate_cuboid_coordinates_cutsAD(C.DIM)

    rows = []
    for sid in ids:
        folder = os.path.join(code, "Samples", sid)
        cut = int(met[sid]["cut"])
        gt = C.imread_rgb(os.path.join(folder, f"cut{cut}_col.png"), C.DIM)
        faces = C.face_images(folder)
        proc = C.load_proc_slice(folder, sid, cut, cutA, cutD)

        mu_gt = gt.reshape(-1, 3).mean(axis=0)
        mu_face = C.face_chan_mean(faces)
        face_means = np.array([im.reshape(-1, 3).mean(axis=0) for im in faces.values()])
        face_spread = face_means.std(axis=0)

        gt_wb = C.white_balance(gt)
        gt_chr, gt_hue = C.chroma_hue(gt)
        gtwb_chr, gtwb_hue = C.chroma_hue(gt_wb)

        # ---------- 现行显示路径 ----------
        proc_disp = np.clip(C.unwhitebalance(proc, mu_face), 0, 1)
        # ---------- 候选 A：两侧都只显示白平衡域 ----------
        refA, canA = gt_wb, proc
        # ---------- 候选 B：对称管线（GT 也白平衡后用同一 mu_face 还原） ----------
        refB = np.clip(C.unwhitebalance(gt_wb, mu_face), 0, 1)
        canB = proc_disp
        # ---------- 候选 C：外表面拟合"白平衡域→相机域"仿射 ----------
        M_d, b_d, mae_d, mu_all = C.fit_face_affine(faces, mode="diag")
        M_f, b_f, mae_f, _ = C.fit_face_affine(faces, mode="full")
        canC1 = C.apply_affine(proc, M_d, b_d)
        canC2 = C.apply_affine(proc, M_f, b_f)

        # ---------- 量化 ----------
        mu_d = (mu_face - mu_gt) * 255.0
        l_gt, l_face = C.mean_color_lab(gt), C.mean_color_lab(
            np.clip(mu_face[None, None, :], 0, 1))
        C_gt = float(np.hypot(l_gt[1], l_gt[2]))
        C_fc = float(np.hypot(l_face[1], l_face[2]))
        h_gt = float(np.degrees(np.arctan2(l_gt[2], l_gt[1])) % 360)
        h_fc = float(np.degrees(np.arctan2(l_face[2], l_face[1])) % 360)

        proc_chr, proc_hue = C.chroma_hue(proc)
        proc_disp_chr, _ = C.chroma_hue(proc_disp)

        # 指标域自检：与 reeval 的 proc_L1 对比（无需 GPU）
        l1_check = float(np.abs(proc - gt_wb).mean())

        r = {
            "sample": sid,
            "species": spec.get(sid, ("", ""))[0],
            "anatomy": spec.get(sid, ("", ""))[1] or "zz",
            "cut": cut,
            "orient": met[sid].get("orient", ""),
            # 一阶错位
            "mu_gt_R": mu_gt[0] * 255, "mu_gt_G": mu_gt[1] * 255, "mu_gt_B": mu_gt[2] * 255,
            "mu_face_R": mu_face[0] * 255, "mu_face_G": mu_face[1] * 255,
            "mu_face_B": mu_face[2] * 255,
            "dRGB_R": mu_d[0], "dRGB_G": mu_d[1], "dRGB_B": mu_d[2],
            "dRGB_norm": float(np.linalg.norm(mu_d)),
            "dRGB_spread": float(mu_d.max() - mu_d.min()),
            "face_mean_spread": float(face_spread.mean() * 255),
            "dE_mean_color": C.de76(l_gt, l_face),
            "dC_mean_color": C_gt - C_fc,
            "dh_mean_color": C.hue_diff_deg(h_gt, h_fc),
            # 饱和度 / 对比
            "gt_sat": C.mean_sat(gt), "gt_chroma": gt_chr,
            "gt_contrast": C.contrast_L(gt),
            "gtwb_chroma": gtwb_chr,
            "proc_chroma_wb": proc_chr,
            "proc_sat_wb": C.mean_sat(proc),
            "proc_contrast_wb": C.contrast_L(proc),
            "procdisp_chroma": proc_disp_chr,
            "procdisp_sat": C.mean_sat(proc_disp),
            # 显示层色差（低频 ΔE）
            "dE_low_cur": C.de_lowfreq(gt, proc_disp, args.sigma),
            "dE_low_A": C.de_lowfreq(refA, canA, args.sigma),
            "dE_low_B": C.de_lowfreq(refB, canB, args.sigma),
            "dE_low_C1": C.de_lowfreq(gt, canC1, args.sigma),
            "dE_low_C2": C.de_lowfreq(gt, canC2, args.sigma),
            # 候选 B 之后的残余差异细节
            "B_dSat": C.mean_sat(canB) - C.mean_sat(refB),
            "B_dChroma": C.chroma_hue(canB)[0] - C.chroma_hue(refB)[0],
            "B_dContrast": C.contrast_L(canB) - C.contrast_L(refB),
            # 外表面映射拟合质量
            "fit_mae_diag": mae_d, "fit_mae_full": mae_f,
            # 自检
            "l1_wb_check": l1_check,
            "l1_csv": float(met[sid]["proc_L1"]),
        }
        rows.append(r)
        print(f"  [{len(rows)}/{len(ids)}] {sid} cur={r['dE_low_cur']:.2f} "
              f"A={r['dE_low_A']:.2f} B={r['dE_low_B']:.2f} "
              f"C1={r['dE_low_C1']:.2f} C2={r['dE_low_C2']:.2f}", flush=True)

    # ---------- 导出 CSV ----------
    csv_path = os.path.join(args.out, "color_diag.csv")
    cols = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---------- 汇总 ----------
    def agg(k):
        v = np.array([r[k] for r in rows], np.float64)
        return float(v.mean()), float(np.median(v)), float(v.max())

    L = []
    p = L.append
    p(f"# 色差诊断（38 块，n={len(rows)}）")
    p("")
    p("## 1. 一阶错位：GT 切面均色 vs 六外表面均值（单位 0-255）")
    for k, nm in [("dRGB_norm", "|ΔRGB| 欧氏"), ("dRGB_spread", "ΔRGB 通道间极差"),
                  ("dE_mean_color", "均色 ΔE76"), ("dC_mean_color", "均色色度差 ΔC"),
                  ("dh_mean_color", "均色色相角差(°)")]:
        m, md, mx = agg(k)
        p(f"- {nm}: mean {m:.2f} / median {md:.2f} / max {mx:.2f}")
    p("")
    p("## 2. 洗白问题：PROC 白平衡域 vs GT（白平衡域）色彩能量")
    m_gt = np.mean([r["gtwb_chroma"] for r in rows])
    m_pr = np.mean([r["proc_chroma_wb"] for r in rows])
    p(f"- GT 白平衡域平均色度 C*: {m_gt:.2f}")
    p(f"- PROC 白平衡域平均色度 C*: {m_pr:.2f}  （比值 {m_pr / max(m_gt, 1e-6):.3f}）")
    m_gts = np.mean([r["gt_sat"] for r in rows])
    m_prs = np.mean([r["procdisp_sat"] for r in rows])
    p(f"- GT 原始照片平均饱和度 S: {m_gts:.3f}；PROC 显示后 S: {m_prs:.3f}")
    p(f"- 平均对比度(L std): GT {np.mean([r['gt_contrast'] for r in rows]):.2f} / "
      f"PROC {np.mean([r['proc_contrast_wb'] for r in rows]):.2f}")
    p("")
    p("## 3. 显示层低频 ΔE（越小越好，σ=%g）" % args.sigma)
    base = np.mean([r["dE_low_cur"] for r in rows])
    for k, nm in [("dE_low_cur", "现行（GT 原图 vs unwb(PROC)）"),
                  ("dE_low_A", "A 两侧都白平衡域"),
                  ("dE_low_B", "B 对称管线（同一 mu_face 还原）"),
                  ("dE_low_C1", "C1 外表面逐通道仿射"),
                  ("dE_low_C2", "C2 外表面 3x3 仿射")]:
        m, md, mx = agg(k)
        p(f"- {nm}: mean {m:.2f} / median {md:.2f} / max {mx:.2f}"
          f"  （较现行 {(m - base) / base * 100:+.1f}%）")
    p("")
    p("## 4. 逐块残余（B 方案后仍 >3 的样本）")
    worst = sorted(rows, key=lambda r: -r["dE_low_B"])[:12]
    p("| sample | 材种 | cur | B | C2 | B_dSat | B_dChroma | B_dContrast |")
    p("|---|---|---|---|---|---|---|---|")
    for r in worst:
        p(f"| {r['sample']} | {r['species']} | {r['dE_low_cur']:.2f} | {r['dE_low_B']:.2f} "
          f"| {r['dE_low_C2']:.2f} | {r['B_dSat']:+.3f} | {r['B_dChroma']:+.2f} "
          f"| {r['B_dContrast']:+.2f} |")
    p("")
    p("## 5. 自检")
    diffs = [abs(r["l1_wb_check"] - r["l1_csv"]) for r in rows]
    p(f"- 白平衡域 L1 复算 vs reeval_orient_summary.csv：max|Δ| = {max(diffs):.2e} "
      f"（渲染/朝向路径一致性校验）")
    p(f"- 外表面仿射拟合 MAE：diag {np.mean([r['fit_mae_diag'] for r in rows]):.4f} / "
      f"full {np.mean([r['fit_mae_full'] for r in rows]):.4f}")
    p("")
    p("## 6. 最差 10 块（按现行方案）")
    for r in sorted(rows, key=lambda r: -r["dE_low_cur"])[:10]:
        p(f"- {r['sample']} {r['species']}: cur {r['dE_low_cur']:.2f} → "
          f"|ΔRGB| {r['dRGB_norm']:.1f} (通道极差 {r['dRGB_spread']:.1f}), "
          f"B {r['dE_low_B']:.2f}")

    txt = "\n".join(L)
    with open(os.path.join(args.out, "color_diag_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    with open(os.path.join(args.out, "color_diag_raw.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
