# -*- coding: utf-8 -*-
"""reeval_orient.py — 按朝向表统一重评 38 块（全部指标）。

口径（notes/eval_geometry.md）：
  · 朝向表：H11/H14/MP07/N07/SG15/SG16 评价时渲染图旋转 180°，其余 rot0；
  · H11 的 PROC 用 proc_cube_knot.npz（节疤开启，正式配置）；
  · 指标：L1 / RMSE / Gram-VGG19 / VGG-SWD2 / DISTS / 生长带 P/R/F1，
    全部白平衡域，管线与 4a_eval_cut.py / 4c_eval_dists_f1.py 一致；
  · 所有立方体（col/gf/nca）与 flat 基线使用同一朝向表。

输出：code/reeval_orient_summary.csv + reeval_orient_summary.json
      每样本 Samples/<id>/proc_eval_orient.json
"""
import os
import sys
import csv
import json

import numpy as np
import cv2

if os.name != "nt":
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: 0
    cv2.destroyAllWindows = lambda *a, **k: None

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "COMMON")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_utils
import loss_utils
from DISTS_local import DISTS
from unet import UNet_2D
import importlib.util as _ilu

# 复用 4c 的 RingExtractor / ring_prf1 / 工具函数
_spec = _ilu.spec_from_file_location("m4c", os.path.join(HERE, "4c_eval_dists_f1.py"))
m4c = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(m4c)

DIM = 256
FLIP180 = {"H11", "H14", "MP07", "N07", "SG15", "SG16"}  # 朝向表（eval_geometry.md §2.3）


def metrics_all(a_wb, b_wb, dists, device):
    ta = torch.from_numpy(a_wb).permute(2, 0, 1).unsqueeze(0).to(device).float()
    tb = torch.from_numpy(b_wb).permute(2, 0, 1).unsqueeze(0).to(device).float()
    out = {"L1": float((ta - tb).abs().mean().item()),
           "RMSE": float(torch.sqrt(((ta - tb) ** 2).mean()).item())}
    tm = loss_utils.VGGStyleLoss().to(device).eval()
    out["Gram-VGG19"] = float(tm(ta, tb).item())
    with torch.no_grad():
        out["DISTS"] = float(dists(ta, tb).item())

    from torchvision.models import vgg16, VGG16_Weights
    vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval().to(device)
    for p in vgg.parameters():
        p.requires_grad_(False)
    style_layers = (1, 6, 11, 18, 25)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def feats(imgs):
        x = (imgs - mean) / std
        o = []
        for i, layer in enumerate(vgg[:max(style_layers) + 1]):
            x = layer(x)
            if i in style_layers:
                b_, c_, h_, w_ = x.shape
                o.append(x.reshape(b_, c_, h_ * w_))
        return o

    g = torch.Generator(device=device).manual_seed(777)
    d = 0.0
    with torch.no_grad():
        fa, fb = feats(ta), feats(tb)
    for x, y in zip(fa, fb):
        b_, c_, n_ = x.shape
        P = torch.randn(c_, 128, generator=g, device=device)
        P = P / (P.norm(dim=0, keepdim=True) + 1e-8)
        px = torch.matmul(x.transpose(1, 2), P).sort(dim=1)[0]
        py = torch.matmul(y.transpose(1, 2), P).sort(dim=1)[0]
        d += float((px - py).abs().mean().item())
    out["VGG-SWD2"] = d / len(fa)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples_root = os.path.join(HERE, "Samples")
    ids = sorted(d for d in os.listdir(samples_root)
                 if os.path.isdir(os.path.join(samples_root, d)))
    cutA, cutD = data_utils.generate_cuboid_coordinates_cutsAD(DIM)

    dists = DISTS().to(device).eval()
    extractor = m4c.RingExtractor(os.path.join(HERE, "unet_trained_model.pt"), device)

    rows = []
    for sid in ids:
        folder = os.path.join(samples_root, sid)
        cut = 2 if os.path.isfile(os.path.join(folder, "cut2_col.png")) else 1
        gt = cv2.imread(os.path.join(folder, f"cut{cut}_col.png"))
        if gt is None:
            continue
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = cv2.resize(gt, (DIM, DIM), interpolation=cv2.INTER_AREA)
        gt_wb = m4c.white_balance(gt)

        # 真值生长带图（4c 已缓存）
        gt_ring_path = os.path.join(folder, f"cut{cut}_arl-unet.png")
        if os.path.isfile(gt_ring_path):
            gt_ring = cv2.imread(gt_ring_path, cv2.IMREAD_GRAYSCALE)
        else:
            gt_ring = extractor.ring_map(cv2.cvtColor(
                np.clip(255 * gt, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(gt_ring_path, gt_ring)

        coords = cutA if cut == 1 else cutD
        rot90 = 1 if cut == 1 else 0
        flip = sid in FLIP180
        chan_mean = m4c.face_chan_mean(folder)

        cubes = {
            "gf": "gf_cube.npz",
            "col": "col_cube.npz",
            # H11 正式配置：节疤开启的体（KNOT_FIX.md）
            "proc": ("proc_cube_knot.npz" if sid == "H11"
                     and os.path.isfile(os.path.join(folder, "proc_cube_knot.npz"))
                     else "proc_cube.npz"),
            "nca": "nca_cube.npz",
        }

        row = {"sample": sid, "cut": cut,
               "orient": "rot180" if flip else "rot0"}
        report = {"sample": sid, "cut": cut, "orient": row["orient"],
                  "proc_cube": cubes["proc"], "metrics": {}}
        for nm, fn in cubes.items():
            vol = m4c.load_npz(folder, fn)
            if vol is None:
                continue
            rend = m4c.render_cut(vol, coords, DIM, rotate90=rot90)
            if flip:
                rend = np.ascontiguousarray(np.rot90(rend, 2))
            a_wb = m4c.white_balance(rend)
            m = metrics_all(a_wb, gt_wb, dists, device)

            shown = m4c.unwhitebalance(rend, chan_mean)
            u8 = np.clip(255 * shown, 0, 255).astype(np.uint8)
            pred_ring = extractor.ring_map(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
            p, r, f1 = m4c.ring_prf1(pred_ring, gt_ring)
            m.update({"RingP": p, "RingR": r, "RingF1": f1})

            report["metrics"][nm] = m
            for k, v in m.items():
                row[f"{nm}_{k}"] = v

        # flat 基线（与朝向无关）
        proc = m4c.load_npz(folder, "proc_cube.npz")
        if proc is not None:
            mean_col = proc.reshape(-1, 3).mean(axis=0)
            flat = np.ones((DIM, DIM, 3), dtype=np.float32) * mean_col
            m = metrics_all(m4c.white_balance(flat), gt_wb, dists, device)
            p, r, f1 = m4c.ring_prf1(np.zeros((DIM, DIM), np.uint8), gt_ring)
            m.update({"RingP": p, "RingR": r, "RingF1": f1})
            report["metrics"]["flat"] = m
            for k, v in m.items():
                row[f"flat_{k}"] = v

        with open(os.path.join(folder, "proc_eval_orient.json"), "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        rows.append(row)
        pm = report["metrics"].get("proc", {})
        print(f"[{sid}] cut{cut} {row['orient']} PROC L1={pm.get('L1', float('nan')):.4f} "
              f"SWD2={pm.get('VGG-SWD2', float('nan')):.4f} "
              f"DISTS={pm.get('DISTS', float('nan')):.4f} "
              f"F1={pm.get('RingF1', float('nan'))}")

    csv_path = os.path.join(HERE, "reeval_orient_summary.csv")
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(HERE, "reeval_orient_summary.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n汇总 {len(rows)} 行 -> {csv_path}")


if __name__ == "__main__":
    main()
