# -*- coding: utf-8 -*-
"""
4c_eval_dists_f1.py — 指标重放：为 4a 评估补齐 DISTS 与生长带 F1（表 3 第 1、3 列）。

只重放指标，不重新生成体：
  · 体纹理直接从磁盘 proc_cube.npz / col_cube.npz / gf_cube.npz / nca_cube.npz 读取；
  · 切面渲染、白平衡、cut1 的 90° 修正与 4a_eval_cut.py 完全一致；
  · DISTS 在白平衡后的 [0,1] RGB 上计算（与论文 §4.2 “DISTS 使用 [0,1] RGB”一致）；
  · 生长带 F1：用冻结的 U-Net（unet_trained_model.pt，训练清单与 38 块测试集
    交集为 0，见 notes/a2_unet_split_check.md）分别提取真值切面照片与各渲染图的
    生长带图，二值化（>127）+ 骨架化，按距离容差 tau=2px（256x256 下）做双向匹配，
    报告精确率 / 召回率 / F1。真值侧同样由该冻结 U-Net 从切面照片提取（数据集中
    切面无人工生长带标注，*ann.png 仅存在于六外表面；U-Net 只作评价用，不回传生成端）。

结果写回每个样本的 proc_eval.json（metrics[模型名] 增加 DISTS / RingF1 / RingP / RingR），
并汇总到 code/reeval_dists_f1_summary.csv 与 reeval_dists_f1_summary.json。

用法：
    python 4c_eval_dists_f1.py -id B08            # 单样本
    python 4c_eval_dists_f1.py -all               # 全部 38 块（读 Samples/ 目录）
    python 4c_eval_dists_f1.py -all -skip_done    # 已有 DISTS 的样本跳过
"""

import os
import sys
import csv
import json
import math
import argparse
from itertools import product
from datetime import datetime

import numpy as np
import cv2

if os.name != "nt":  # r740 无头服务器补丁
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: 0
    cv2.destroyAllWindows = lambda *a, **k: None

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "COMMON")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_utils
from unet import UNet_2D
from DISTS_local import DISTS

LETTERS = ["A", "B", "C", "D", "E", "F"]
DIM = 256
PATCH_SIZE = 64
RING_THRESHOLD = 127      # U-Net 生长带图二值化阈值（uint8 域）
MATCH_TOL_PX = 2.0        # 生长带匹配距离容差（像素，256x256 下）


# ----------------------------------------------------------------------
# 与 4a_eval_cut.py 一致的基础函数
# ----------------------------------------------------------------------
def load_npz(folder, name):
    p = os.path.join(folder, name)
    if not os.path.isfile(p):
        return None
    return np.load(p)["arr_0"]


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


def face_chan_mean(folder):
    means = []
    for ltr in LETTERS:
        fp = os.path.join(folder, f"{ltr}_col.png")
        if os.path.isfile(fp):
            im = cv2.imread(fp)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            means.append(im.astype(np.float32).reshape(-1, 3).mean(axis=0) / 255.0)
    if means:
        return np.mean(means, axis=0)
    return np.array([0.5, 0.5, 0.5], dtype=np.float32)


# ----------------------------------------------------------------------
# 生长带提取（冻结 U-Net，与 2_apply_unet.py 同一推理管线）
# ----------------------------------------------------------------------
class RingExtractor:
    def __init__(self, weights_path, device):
        self.device = device
        self.unet = UNet_2D(in_dim=3, out_dim=1).to(device)
        self.unet.load_state_dict(torch.load(weights_path, map_location=device))
        self.unet.eval()
        mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        cx = cy = (PATCH_SIZE - 1) / 2
        sigma2 = 8.0 * 8.0
        for y in range(PATCH_SIZE):
            for x in range(PATCH_SIZE):
                mask[y, x] = math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / sigma2)
        self.mask = mask

    @torch.no_grad()
    def ring_map(self, src_img):
        """src_img: uint8 (H,W,3) BGR 或 RGB（U-Net 对通道顺序与原管线一致地吃 cv2 读入的图）。
        返回 uint8 (H,W) 生长带灰度图。"""
        H, W, _ = src_img.shape
        img = np.zeros((H, W), dtype=np.float32)
        img_count = np.zeros((H, W), dtype=np.float32)
        for yi, xi in product(range(0, H, PATCH_SIZE // 2),
                              range(0, W, PATCH_SIZE // 2)):
            y = yi if yi < H - PATCH_SIZE else H - PATCH_SIZE
            x = xi if xi < W - PATCH_SIZE else W - PATCH_SIZE
            in_patch = src_img[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :]
            in_patch = data_utils.numpy_image_to_norm_torch_data(
                in_patch, PATCH_SIZE, src=True, lst_out=True)
            out_patch = self.unet(in_patch.to(self.device))[0]
            out_patch = data_utils.norm_torch_data_to_numpy_image(out_patch)
            out_patch = out_patch.astype(np.float32).reshape(PATCH_SIZE, PATCH_SIZE)
            img[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += self.mask * out_patch
            img_count[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += self.mask
        img = img / img_count
        return np.uint8(np.clip(img, 0, 255))


def ring_prf1(pred_u8, gt_u8, thr=RING_THRESHOLD, tol=MATCH_TOL_PX):
    """骨架化 + 距离容差双向匹配，返回 (P, R, F1)。两侧空图的退化情形返回 nan。"""
    from skimage.morphology import skeletonize
    from scipy.ndimage import distance_transform_edt
    pb = skeletonize(pred_u8 > thr)
    gb = skeletonize(gt_u8 > thr)
    np_, ng = int(pb.sum()), int(gb.sum())
    if ng == 0:
        return float("nan"), float("nan"), float("nan")  # 真值无标注：指标未定义
    if np_ == 0:
        return float("nan"), 0.0, 0.0  # 有标注但零检出：召回与 F1 为 0，精确率未定义
    dt_g = distance_transform_edt(~gb)
    dt_p = distance_transform_edt(~pb)
    prec = float((dt_g[pb] <= tol).mean())
    rec = float((dt_p[gb] <= tol).mean())
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


# ----------------------------------------------------------------------
def eval_sample(folder, sample_id, cut, dists, extractor, device):
    gt_path = os.path.join(folder, f"cut{cut}_col.png")
    if not os.path.isfile(gt_path):
        print(f"[{sample_id}] 缺 cut{cut}_col.png，跳过")
        return None

    gt = cv2.imread(gt_path)
    gt = cv2.resize(gt, (DIM, DIM), interpolation=cv2.INTER_AREA)  # BGR, 供 U-Net
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    # 真值生长带图（缓存到磁盘，复跑不重算）
    gt_ring_path = os.path.join(folder, f"cut{cut}_arl-unet.png")
    if os.path.isfile(gt_ring_path):
        gt_ring = cv2.imread(gt_ring_path, cv2.IMREAD_GRAYSCALE)
    else:
        gt_ring = extractor.ring_map(gt)
        cv2.imwrite(gt_ring_path, gt_ring)

    gt_wb = white_balance(gt_rgb)
    tb = torch.from_numpy(gt_wb).permute(2, 0, 1).unsqueeze(0).to(device).float()

    cutA, cutD = data_utils.generate_cuboid_coordinates_cutsAD(DIM)
    cut_coords = cutA if cut == 1 else cutD
    rot = 1 if cut == 1 else 0
    chan_mean = face_chan_mean(folder)

    cubes = {
        "gf_cube (生长场切片)": "gf_cube.npz",
        "col_cube (仅颜色)": "col_cube.npz",
        "PROC (4a 程序纹理)": "proc_cube.npz",
        "NCA (4b)": "nca_cube.npz",
    }

    eval_json = os.path.join(folder, "proc_eval.json")
    report = {}
    if os.path.isfile(eval_json):
        with open(eval_json) as f:
            report = json.load(f)
    metrics = report.setdefault("metrics", {})

    row = {"sample": sample_id, "cut": cut}
    for nm, fn in cubes.items():
        vol = load_npz(folder, fn)
        if vol is None:
            continue
        rendered = render_cut(vol, cut_coords, DIM, rotate90=rot)
        a_wb = white_balance(rendered)
        ta = torch.from_numpy(a_wb).permute(2, 0, 1).unsqueeze(0).to(device).float()
        with torch.no_grad():
            d = float(dists(ta, tb).item())

        # U-Net 吃 cv2 风格 uint8 图：先做外表面定色的颜色还原（与展示管线一致）
        shown = unwhitebalance(rendered, chan_mean)
        shown_u8 = np.clip(255.0 * shown, 0, 255).astype(np.uint8)  # RGB
        pred_ring = extractor.ring_map(cv2.cvtColor(shown_u8, cv2.COLOR_RGB2BGR))
        p, r, f1 = ring_prf1(pred_ring, gt_ring)

        m = metrics.setdefault(nm, {})
        m["DISTS"] = d
        m["RingP"] = p
        m["RingR"] = r
        m["RingF1"] = f1
        short = nm.split(" ")[0]
        row[f"{short}_DISTS"] = d
        row[f"{short}_F1"] = f1

    # flat 基线
    proc = load_npz(folder, "proc_cube.npz")
    if proc is not None:
        mean_col = proc.reshape(-1, 3).mean(axis=0)
        flat = np.ones((DIM, DIM, 3), dtype=np.float32) * mean_col
        a_wb = white_balance(flat)
        ta = torch.from_numpy(a_wb).permute(2, 0, 1).unsqueeze(0).to(device).float()
        with torch.no_grad():
            d = float(dists(ta, tb).item())
        p, r, f1 = ring_prf1(np.zeros((DIM, DIM), np.uint8), gt_ring)
        m = metrics.setdefault("baseline: flat mean color", {})
        m["DISTS"] = d
        m["RingP"], m["RingR"], m["RingF1"] = p, r, f1
        row["flat_DISTS"] = d
        row["flat_F1"] = f1

    report["dists_f1_note"] = (
        f"DISTS 于白平衡 [0,1] RGB 计算（DISTS_local，官方 weights.pt）；"
        f"生长带 P/R/F1 由冻结 U-Net 双边提取 + 骨架化 + 容差 {MATCH_TOL_PX}px 匹配，"
        f"阈值 {RING_THRESHOLD}；重放时间 {datetime.now().isoformat(timespec='seconds')}")
    with open(eval_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[{sample_id}] cut{cut} 完成：{row}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-id", type=str, default="")
    ap.add_argument("-all", action="store_true")
    ap.add_argument("-cut", type=int, default=0, choices=[0, 1, 2],
                    help="0=按 RESULTS_38 主表逐样本的 cut 自动判定（有 cut2 用 2）")
    ap.add_argument("-skip_done", action="store_true")
    ap.add_argument("-torch_home", type=str, default="/home/gast/.cache/torch")
    ap.add_argument("-device", type=str, default="")
    args = ap.parse_args()

    os.environ["TORCH_HOME"] = args.torch_home
    torch.hub.set_dir(os.path.join(args.torch_home, "hub"))
    device = args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu")

    samples_root = os.path.join(HERE, "Samples")
    if args.all:
        ids = sorted(d for d in os.listdir(samples_root)
                     if os.path.isdir(os.path.join(samples_root, d)))
    else:
        ids = [args.id]

    dists = DISTS().to(device).eval()
    extractor = RingExtractor(os.path.join(HERE, "unet_trained_model.pt"), device)

    rows = []
    for sid in ids:
        folder = os.path.join(samples_root, sid)
        cut = args.cut
        if cut == 0:
            cut = 2 if os.path.isfile(os.path.join(folder, "cut2_col.png")) else 1
        if args.skip_done:
            ej = os.path.join(folder, "proc_eval.json")
            try:
                with open(ej) as f:
                    mm = json.load(f)["metrics"]
                if "DISTS" in mm.get("PROC (4a 程序纹理)", {}):
                    print(f"[{sid}] 已有 DISTS，跳过")
                    continue
            except Exception:
                pass
        row = eval_sample(folder, sid, cut, dists, extractor, device)
        if row:
            rows.append(row)

    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        csv_path = os.path.join(HERE, "reeval_dists_f1_summary.csv")
        # 追加写：如已存在则按 sample 覆盖
        old = {}
        if os.path.isfile(csv_path):
            with open(csv_path, newline="") as f:
                for r in csv.DictReader(f):
                    old[r["sample"]] = r
        for r in rows:
            old[r["sample"]] = {k: r.get(k, "") for k in keys}
        all_keys = sorted({k for r in old.values() for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for sid in sorted(old):
                w.writerow(old[sid])
        print(f"汇总已写 -> {csv_path}（{len(old)} 行）")


if __name__ == "__main__":
    main()
