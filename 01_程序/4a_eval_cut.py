# -*- coding: utf-8 -*-
"""
4a_eval_cut.py — 在 Mokume 目标斜切面上评价 4a 生成的程序化体纹理。

==============================================================================
                    为什么单独一个文件？（第 4.4 节协议）
==============================================================================
 4a_proc.py（生成端）在运行时完全不读取任何斜切面数据：
   · 优化 / 适配 / 网格搜索的评判只使用 6 个外表面观测 {A..F}_col.png；
   · 颜色提示（color hint）只由外表面照片统计得到。
 本文件是全流程中唯一读取目标斜切面照片 cut2_col.png 的地方，
 且仅用于计算并报告指标，不产生任何梯度、不回传任何信息到生成端。
 二者只通过磁盘上的 proc_cube.npz / gf_cube.npz / nca_cube.npz 交互。
==============================================================================

用法：
    python 4a_eval_cut.py -id CN03
    python 4a_eval_cut.py -id CN03 -cut 1        # 若样本存在 cut1_col.png
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import cv2
import os as _os
if _os.name != "nt":  # --- r740 无头服务器移植补丁：禁用 GUI ---
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

LETTERS = ["A", "B", "C", "D", "E", "F"]


def load_npz(folder, name):
    p = os.path.join(folder, name)
    if not os.path.isfile(p):
        return None
    return np.load(p)["arr_0"]


def render_cut(vol, cut_coords, dim=256, rotate90=0):
    """沿切面坐标采样体纹理，返回 (dim,dim,3) float32（白平衡域）。

    rotate90: 额外旋转的 90 度次数（1 = 顺时针 90 度）。
    实测（check_orient.py，结构张量测走向）：官方 generate_cuboid_coordinates_cutsAD
    给出的 cutA 坐标采出的图像，木纹走向与 cut1 照片相差约 90 度
    （B08 89°、CH04 88°、BW02 73°），而 cutD 与 cut2 照片一致（2–4°），
    且 col_cube（Step 3 输出）同样存在该偏差 —— 说明是 cut1 切面坐标的朝向约定
    与照片不一致，与 4a 无关。故对 cut1 补一次顺时针 90 度旋转。
    """
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
    """按各通道均值归一化，消除曝光/光照差异（用于指标计算）。"""
    x = np.asarray(x, dtype=np.float32)
    if x.max() > 2.0:
        x = x / 255.0
    mean = x.reshape(-1, 3).mean(axis=0)
    return np.clip(x / (2.0 * (mean + eps)), 0.0, 1.0)


def compute_metrics(rendered, gt, device="cuda"):
    a = white_balance(rendered)
    b = white_balance(gt)
    ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device).float()
    tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(device).float()

    out = {
        "L1": float((ta - tb).abs().mean().item()),
        "RMSE": float(torch.sqrt(((ta - tb) ** 2).mean()).item()),
    }
    # Gram 风格距离（内部 VGG19），需搬到与输入相同的设备
    tm = loss_utils.VGGStyleLoss().to(device).eval()
    out["Gram-VGG19"] = float(tm(ta, tb).item())

    # VGG-SWD2：VGG16 特征 + 128 个固定单位高斯投影（种子 777）
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


def unwhitebalance(img, chan_mean):
    """把白平衡域的渲染结果还原到原始色彩域。

    体纹理（proc/col/nca）都工作在白平衡域：每通道均值被归一化到 0.5，
    即视觉上的中性灰，木头的棕色正是白平衡时被除掉的。
    导出可视化前必须还原，否则图会是灰的。

    img: float (H,W,3) 白平衡域；chan_mean: (3,) 原始域通道均值（0-1）
    """
    img = np.asarray(img, dtype=np.float32)
    m = img.reshape(-1, 3).mean(axis=0)
    return img / (m + 1e-6) * np.asarray(chan_mean, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 4a procedural texture on the Mokume slanted cut")
    parser.add_argument("-id", type=str, default="CN03",
                        help="统一测试木块 ID")
    parser.add_argument("-cut", type=int, default=2, choices=[1, 2],
                        help="目标斜切面编号（默认 2；CN03 提供 cut2_col.png）")
    parser.add_argument("-torch_home", type=str,
                        default=r"D:\CODE\moke\hub",
                        help="预训练权重缓存目录")
    parser.add_argument("-device", type=str, default="",
                        help="cuda / cpu（默认自动选择）")
    args = parser.parse_args()

    torch_home = args.torch_home
    os.makedirs(torch_home, exist_ok=True)
    os.environ["TORCH_HOME"] = torch_home
    torch.hub.set_dir(os.path.join(torch_home, "hub"))

    device = args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu")

    folder = os.path.join(HERE, "Samples", args.id)
    DIM = 256

    # ------------------------------------------------------------------
    # 以下为评价专用：读取目标斜切面真值照片
    # （生成端 4a_proc.py 从不读取此文件）
    # ------------------------------------------------------------------
    gt_path = os.path.join(folder, f"cut{args.cut}_col.png")
    if not os.path.isfile(gt_path):
        print(f"样本 {args.id} 没有目标斜切面照片 {gt_path}，无法评价。")
        return
    gt = cv2.imread(gt_path)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt = cv2.resize(gt, (DIM, DIM), interpolation=cv2.INTER_AREA)

    cutA, cutD = data_utils.generate_cuboid_coordinates_cutsAD(DIM)
    cut_coords = cutA if args.cut == 1 else cutD

    # 外表面观测的通道均值：用于把白平衡域的渲染结果还原成真实色彩。
    # 【严格留出协议修复 2026-10-21】展示定色只允许使用外表面统计或中性定色，
    # 禁止从目标切面提取任何颜色统计（旧版曾用 GT 切面通道均值，已移除）。
    # 现取六面外表面照片 A-F_col.png 各自的通道均值的平均（display-only，
    # 仅影响展示列的整体色温；指标始终在白平衡域计算、对此免疫）。
    face_means = []
    for _ltr in LETTERS:
        _fp = os.path.join(folder, f"{_ltr}_col.png")
        if os.path.isfile(_fp):
            _im = cv2.imread(_fp)
            _im = cv2.cvtColor(_im, cv2.COLOR_BGR2RGB)
            face_means.append(_im.astype(np.float32).reshape(-1, 3).mean(axis=0) / 255.0)
    if face_means:
        gt_chan_mean = np.mean(face_means, axis=0)  # 外表面统计定色（变量名保留以复用下游代码）
    else:
        gt_chan_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # 中性定色兜底

    results = {}
    tiles = []
    titles = []

    cubes = {
        "gf_cube (生长场切片)": "gf_cube.npz",
        "col_cube (仅颜色)": "col_cube.npz",
        "PROC (4a 程序纹理)": "proc_cube.npz",
        "NCA (4b)": "nca_cube.npz",
    }

    for nm, fn in cubes.items():
        vol = load_npz(folder, fn)
        if vol is None:
            print(f"  跳过 {nm}（{fn} 缺失）")
            continue
        # 灰阶体与彩色体统一按 3 通道处理
        rendered = render_cut(vol, cut_coords, DIM,
                              rotate90=1 if args.cut == 1 else 0)
        # 指标仍在白平衡域比较（双方各自归一化，对整体色偏免疫）
        results[nm] = compute_metrics(rendered, gt, device=device)
        # 可视化列做颜色还原（display-only，用 GT 切面自身的通道均值定色温）
        shown = unwhitebalance(rendered, gt_chan_mean)
        tiles.append(np.clip(255.0 * shown, 0, 255).astype(np.uint8))
        titles.append(nm)

    # 基线：整块体纹理的平均色（无任何结构）
    proc = load_npz(folder, "proc_cube.npz")
    if proc is not None:
        mean_col = proc.reshape(-1, 3).mean(axis=0)
        flat = np.ones((DIM, DIM, 3), dtype=np.float32) * mean_col
        results["baseline: flat mean color"] = compute_metrics(
            flat, gt, device=device)

    out_dir = os.path.join(folder, "proc_vis")
    os.makedirs(out_dir, exist_ok=True)

    # 顶部真值行（标题使用 ASCII，避免 matplotlib/OpenCV 缺中文字体显示为 ???）
    disp_names = {
        "gf_cube (生长场切片)": "GF (growth field)",
        "col_cube (仅颜色)": "COL (color only)",
        "PROC (4a 程序纹理)": "PROC (4a, ours)",
        "NCA (4b)": "NCA (4b)",
    }
    gt_row = gt.copy()
    tiles_all = [gt_row] + tiles
    titles_all = ["GT cut%d" % args.cut] + [disp_names.get(t, t) for t in titles]

    # 每列显示 2 倍放大，便于观察纹理
    scale = 2
    tiles_all = [cv2.resize(t, (DIM * scale, DIM * scale),
                            interpolation=cv2.INTER_CUBIC) for t in tiles_all]

    canvas = np.hstack(tiles_all)
    h, w = canvas.shape[:2]
    bar = 255 * np.ones((46, w, 3), dtype=np.uint8)
    canvas = np.vstack([bar, canvas])
    for i, t in enumerate(titles_all):
        x = 10 + i * DIM * scale
        cv2.putText(canvas, t, (x, 30), cv2.FONT_HERSHEY_DUPLEX,
                    0.7, (255, 255, 255), 3)
        cv2.putText(canvas, t, (x, 30), cv2.FONT_HERSHEY_DUPLEX,
                    0.7, (0, 0), 0)
    # 每列底部标注其指标（SWD2 / Gram），GT 列标注真值
    foot = 255 * np.ones((34, w, 3), dtype=np.uint8)
    canvas = np.vstack([canvas, foot])
    disp_metrics = ["GT"] + [disp_names.get(t, t) for t in titles]
    for i, t in enumerate(titles_all):
        x = 10 + i * DIM * scale
        if i == 0:
            txt = "ground truth"
        else:
            m = results[titles[i - 1]]
            txt = "SWD2 %.3f  Gram %.4f  L1 %.4f" % (
                m["VGG-SWD2"], m["Gram-VGG19"], m["L1"])
        cv2.putText(canvas, txt, (x, h + 46 + 24), cv2.FONT_HERSHEY_DUPLEX,
                    0.55, (0, 0), 1)

    fig_path = os.path.join(out_dir, "cut_eval_compare.png")
    cv2.imwrite(fig_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    print("=" * 74)
    print(f" 目标斜切面 cut{args.cut} 评价结果  |  样本 {args.id}  |  设备 {device}")
    print("=" * 74)
    print(f" {'模型/基线':<26}{'L1↓':>10}{'RMSE↓':>10}{'Gram↓':>10}{'SWD↓':>10}")
    print("-" * 74)
    for nm, m in results.items():
        print(f" {nm:<26}{m['L1']:>10.4f}{m['RMSE']:>10.4f}"
              f"{m['Gram-VGG19']:>10.4f}{m['VGG-SWD2']:>10.4f}")
    print("-" * 74)

    if "PROC (4a 程序纹理)" in results and "baseline: flat mean color" in results:
        p = results["PROC (4a 程序纹理)"]["L1"]
        b = results["baseline: flat mean color"]["L1"]
        if b > 0:
            print(f" 相对均匀色基线，PROC 的 L1 变化：{(p - b) / b * 100:+.1f}%")
    print("=" * 74)

    report = {
        "sample": args.id,
        "cut": args.cut,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "note": ("斜切面照片仅用于评价；"
                 "生成端 4a_proc.py 只使用 6 个外表面观测；"
                 "展示定色使用六面外表面统计（严格留出协议修复 2026-10-21，"
                 "不使用目标切面任何颜色统计）"),
        "metrics": results,
    }
    with open(os.path.join(folder, "proc_eval.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  报告已保存 -> {os.path.join(folder, 'proc_eval.json')}")


if __name__ == "__main__":
    main()
