# -*- coding: utf-8 -*-
"""eval_knot_h11.py — H11 节疤消融评价：knot OFF/ON × rot0/rot180 四格对比。

读取：
  Samples/H11/proc_cube.npz        （knot OFF，主表原体）
  Samples/H11/proc_cube_knot.npz   （knot ON，2026-10-23 重建）
对每个体在 cut2 的 rot0 / rot180 两种评价朝向下计算 L1 / RMSE / DISTS /
Gram-VGG19 / VGG-SWD2（白平衡域，与 4a_eval_cut.py 同管线）。
输出：code/eval_knot_h11.json + 四联对比图 proc_vis/knot_ablation.png。
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

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "COMMON")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_utils
import loss_utils
from DISTS_local import DISTS

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


def metrics(a_wb, b_wb, dists, device):
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
        fa = feats(ta)
        fb = feats(tb)
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
    folder = os.path.join(HERE, "Samples", "H11")
    gt = cv2.imread(os.path.join(folder, "cut2_col.png"))
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt = cv2.resize(gt, (DIM, DIM), interpolation=cv2.INTER_AREA)
    gt_wb = white_balance(gt)

    _, cutD = data_utils.generate_cuboid_coordinates_cutsAD(DIM)

    dists = DISTS().to(device).eval()

    cubes = {
        "PROC knot OFF": "proc_cube.npz",
        "PROC knot ON": "proc_cube_knot.npz",
    }
    res = {}
    tiles = [np.clip(255 * gt_wb, 0, 255).astype(np.uint8)]
    titles = ["GT cut2"]
    for nm, fn in cubes.items():
        vol = load_npz(folder, fn)
        if vol is None:
            print(f"缺 {fn}，跳过 {nm}")
            continue
        rend = render_cut(vol, cutD, DIM)
        for rot in (0, 180):
            a = rend if rot == 0 else np.rot90(rend, 2).copy()
            a_wb = white_balance(a)
            m = metrics(a_wb, gt_wb, dists, device)
            res[f"{nm} rot{rot}"] = m
            print(f"{nm} rot{rot}: L1={m['L1']:.4f} SWD2={m['VGG-SWD2']:.4f} "
                  f"DISTS={m['DISTS']:.4f} Gram={m['Gram-VGG19']:.5f}")
            tiles.append(np.clip(255 * a_wb, 0, 255).astype(np.uint8))
            titles.append(f"{nm.replace('PROC ', '')} r{rot}")

    out_path = os.path.join(HERE, "eval_knot_h11.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("saved", out_path)

    # 四联图（GT + 4）
    scale = 2
    T = [cv2.resize(t, (DIM * scale, DIM * scale),
                    interpolation=cv2.INTER_CUBIC) for t in tiles]
    canvas = np.hstack(T)
    h, w = canvas.shape[:2]
    bar = 255 * np.ones((40, w, 3), dtype=np.uint8)
    canvas = np.vstack([bar, canvas])
    foot = 255 * np.ones((30, w, 3), dtype=np.uint8)
    canvas = np.vstack([canvas, foot])
    for i, t in enumerate(titles):
        x = 10 + i * DIM * scale
        m = res.get(f"PROC {t.replace(' r', ' rot')}") if i > 0 else None
        cv2.putText(canvas, t, (x, 26), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0), 2)
        if m:
            txt = "L1 %.3f SWD %.3f" % (m["L1"], m["VGG-SWD2"])
            cv2.putText(canvas, txt, (x, h + 40 + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0), 1)
    vis = os.path.join(folder, "proc_vis")
    os.makedirs(vis, exist_ok=True)
    cv2.imwrite(os.path.join(vis, "knot_ablation.png"),
                cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print("saved", os.path.join(vis, "knot_ablation.png"))


if __name__ == "__main__":
    main()
