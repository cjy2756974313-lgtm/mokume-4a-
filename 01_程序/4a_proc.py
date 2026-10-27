# -*- coding: utf-8 -*-
"""
4a_proc.py — Step 4a：反求程序化体纹理
              (Approach 1: inverse 3D procedural texturing)

论文依据
--------
Larsson et al. 2025, "The Mokume Dataset and Inverse Modeling of Solid Wood
Textures", ACM TOG 44(4), Article 162.
  · Sec. 5.3    —— F(G(w_hat), A, phi) 的求值方式
  · Sec. 5.3.1  —— Approach 1: inverse 3D procedural texturing
  · App. A.2.3  —— 详细程序化模型参数清单（无节子 436 / 含一个节子 630）

本脚本做什么
------------
阶段 3（3_infer_gf.py）已从 6 面外表面照片反求出生长场 G(w_hat) 与年轮
等值 A，并给出 1D 色条的颜色初始化。本脚本在此基础上反求「详细程序化
纹理模型」的全部参数 phi，产出可直接点求值（point-wise evaluation）的
实体木体纹理。

优化流程（严格按 Sec. 5.3.1 原文四步）
--------------------------------------
  Stage A  连续优化 + L1 图像损失    —— 初始化年轮颜色（无细节）
  Stage B  切换到 VGG 风格损失       —— 继续优化颜色（仍无细节）
  Stage C  纤维 / 管孔 / 射线尺寸粗网格搜索（开细节，风格损失评判）
  Stage D  全部 phi 连续优化（开细节 + 风格损失）

参数分组（App. A.2.3）
---------------------
  早材颜色   base(3) + 1D 径向色图(64x3)              195
  晚材颜色   base(3) + 1D 径向色图(64x3)              195
  侧面颜色   每外表面一色（早/晚各一套）                 36
  早晚材过渡 start,end (+smooth/linear blend)           2 (+2)
  纤维       size, color, darkness, latewood            4
  管孔       radius, cell(w,l,h), color, aniso,
            occ, occ_lw, occ-ring, scale-ring           14
  射线       radius, cell(w,l,h), color, occ            15
  ----------------------------------------------------
  合计（无节子）                                     ≈463
  含一个节子（+ 节子色图 32x3 与 aniso 等）           ≈568

  注：论文正文把「侧面色」按 6x3=18 计，合计 436；代码实现里侧面色对
  早/晚材各存一份（36），故实际可训练参数略多。两者只是记账口径差异，
  参数语义与论文一致。

关于设备
--------
COMMON/procedural_wood_function.py 内部有硬编码的 CPU 张量（如
closest_angle() 中的 norm = torch.zeros(3)），原作者的 3_infer_gf.py
同样全程在 CPU 上运行。为了不改作者源码、保持与官方行为一致，
本脚本让 params 与程序化求值固定在 CPU；只有 VGG 前向/风格损失
放到 GPU（跨设备计算图可正常反向传播）。

=============================================================================
                  实验协议（严格按第 4.4 节要求）
=============================================================================
  1) 使用统一的 Mokume 测试木块：默认 CN03。
     CN03 同时具备 (a) 6 面外表面照片、(b) 目标斜切面照片 cut2_col.png，
     是数据集中可直接做「斜切面评价」的样本。
  2) 外表面观测 = 唯一输入：Samples/<ID>/{A..F}_col.png。
     所有损失、梯度、网格搜索的评价都只在这 6 个外表面上进行。
  3) 生长场 G(w_hat) 与年轮等值 A 来自阶段 3 产物 gf_params.pkl，
     本阶段全程冻结（requires_grad=False），不参与优化。
  4) 目标斜切面 cut2_col.png —— 只在 evaluate_on_slanted_cut() 中读取一次，
     用于报告指标。它不参与训练、不参与适配、不参与网格搜索/参数选择，
     也不参与颜色提示（color hint）提取。
     颜色提示只取自外表面观测。
=============================================================================

用法：
    python 4a_proc.py -id CN03
    python 4a_proc.py -id CN03 -res 128 -iters 400
"""

import os
import sys
import argparse
import pickle
import json
import time
from datetime import datetime

import numpy as np
import cv2
import os as _os
if _os.name != "nt":  # --- r740 无头服务器移植补丁：禁用 GUI ---
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: 0
    cv2.destroyAllWindows = lambda *a, **k: None
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

# --------------------------------------------------------------------------
# 路径设置：让 COMMON 下的模块可以被直接 import（与 3_infer_gf.py 一致）
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "COMMON")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch.pi = torch.acos(torch.zeros(1)).item() * 2

# 导入项目公共模块
# 注意：ProceduralParameters 模块导入时会把全局默认 dtype 设为 float64，
# 而 DataInstance 会设回 float32。这里与 3_infer_gf.py 保持相同导入顺序，
# 使最终全局默认 dtype 为 float32。
from procedural_wood_function import *          # noqa: F401,F403
import data_utils                                # noqa: E402
import loss_utils                                # noqa: E402
import opti_utils                                # noqa: E402
from ProceduralParameters import ProceduralParameters  # noqa: E402
from DataInstance import DataInstance            # noqa: E402

torch.set_default_dtype(torch.float32)

LETTERS = ["A", "B", "C", "D", "E", "F"]
AXES = [2, 1, 0, 1, 0, 2]   # 每个外表面的法线轴（与 3_infer_gf.py 一致）

# 程序化求值固定在 CPU（原因见文件头「关于设备」）
PROC_DEVICE = torch.device("cpu")


# ==========================================================================
#                          参数组 phi
# ==========================================================================
class PhiParams(nn.Module):
    """详细程序化纹理模型的待优化参数 phi。

    生长场参数（髓心 O/V、变形 R、灰阶图 M）不在此类中 —— 它们来自阶段 3
    并且在本阶段被冻结。

    各组参数通过 ProceduralParameters 的 update_* 接口写入 params 实例；
    由于这些接口内部是纯算术运算，梯度可回流到本类的 nn.Parameter。
    """

    N_SIDE = 6       # 立方体面数 A-F
    N_MAP = 64       # 1D 径向色图分辨率（论文 App. A.2.3）
    N_KNOT_MAP = 32  # 节子色图分辨率（现有接口为 32）

    def __init__(self, knot=False):
        super().__init__()
        self.knot = knot

        # --- 年轮颜色：早材 / 晚材 ---
        self.ew_lw_cols = nn.Parameter(torch.zeros(3, 2))        # [RGB, 早/晚]
        self.ew_lw_side_cols = nn.Parameter(torch.zeros(self.N_SIDE, 3, 2))
        self.ew_lw_col_bars = nn.Parameter(torch.zeros(self.N_MAP, 3, 2))

        # --- 早晚材过渡 ---
        self.lw_end_start = nn.Parameter(torch.zeros(2))
        self.extra_ew_lw_cols = nn.Parameter(torch.zeros(3, 2))
        self.extra_lw_end_start = nn.Parameter(torch.zeros(2))
        self.lw_end_start_linear = nn.Parameter(torch.zeros(2))      # smooth/linear blend

        # --- 纤维 ---
        self.fiber_disc = nn.Parameter(torch.zeros(1))                # 尺寸（网格搜索）
        self.fiber_cont = nn.Parameter(torch.zeros(3))

        # --- 管孔 ---
        self.pore_disc = nn.Parameter(torch.zeros(5))                 # 基元尺寸/发生率（网格搜索）
        self.pore_cont = nn.Parameter(torch.zeros(9))

        # --- 射线 ---
        self.ray_disc = nn.Parameter(torch.zeros(4))                   # 基元尺寸/发生率（网格搜索）
        self.ray_cont = nn.Parameter(torch.zeros(11))

        if knot:
            self.knot_deform = nn.Parameter(torch.zeros(8))
            self.knot_col_bar = nn.Parameter(torch.zeros(self.N_KNOT_MAP, 3))
            self.knot_ani = nn.Parameter(torch.zeros(1))

    def groups(self):
        """返回 {组名: [Parameter, ...]}，供分阶段优化使用。"""
        g = {
            "ring_color": [self.ew_lw_cols, self.ew_lw_side_cols,
                           self.ew_lw_col_bars, self.lw_end_start],
            "ring_color_fine": [self.extra_ew_lw_cols, self.extra_lw_end_start],
            "transition_blend": [self.lw_end_start_linear],
            "fiber": [self.fiber_cont],
            "pore": [self.pore_cont],
            "ray": [self.ray_cont],
        }
        if self.knot:
            g["knot"] = [self.knot_deform, self.knot_col_bar, self.knot_ani]
        return g

    def named_groups(self):
        return {k: sum(p.numel() for p in v) for k, v in self.groups().items()}

    def apply(self, params):
        """把当前 phi 写入 params，供程序化函数求值。"""
        params.update_annual_ring_colors(
            self.ew_lw_cols, self.ew_lw_side_cols,
            self.ew_lw_col_bars, self.lw_end_start)
        params.update_base_detailed_annual_ring_colors()
        params.update_detailed_annual_ring_colors(
            self.extra_ew_lw_cols, self.extra_lw_end_start,
            self.lw_end_start_linear)
        params.update_discontinous_fiber_parameters(self.fiber_disc)
        params.update_continous_fiber_parameters(self.fiber_cont)
        params.update_discontinous_pore_parameters(self.pore_disc)
        params.update_continous_pore_parameters(self.pore_cont)
        params.update_discontinous_ray_parameters(self.ray_disc)
        params.update_continous_ray_parameters(self.ray_cont)
        if self.knot:
            params.update_knot_deform_parameters(self.knot_deform)
        if self.knot:
            params.update_knot_colors(self.knot_col_bar, self.knot_ani)
        return params


# ==========================================================================
#                              VGG 特征 / 风格损失
# ==========================================================================
class VGGFeat(nn.Module):
    """VGG16 特征提取器（ImageNet 预训练，冻结）。

    用 register_buffer 保存 mean/std，使其随 .to(device) 一起搬运，
    规避库内实现中「CPU 常量 vs GPU 输入」的 device 不匹配。
    """

    STYLE_LAYERS = (1, 6, 11, 18, 25)

    def __init__(self, device="cuda"):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        layers = []
        for l in vgg:
            if isinstance(l, nn.MaxPool2d):
                l = nn.AvgPool2d(l.kernel_size, l.stride, l.padding, l.ceil_mode)
            layers.append(l)
        self.vgg = nn.Sequential(*layers)
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)

    def forward(self, imgs):
        x = (imgs - self.mean) / self.std
        out = []
        for i, layer in enumerate(self.vgg[:max(self.STYLE_LAYERS) + 1]):
            x = layer(x)
            if i in self.STYLE_LAYERS:
                b, c, h, w = x.shape
                out.append(x.reshape(b, c, h * w))
        return out


class StyleLoss(nn.Module):
    """可微风格损失：Gram 相关矩阵距离 + Relaxed OT（style + moment）。

    与论文 / 4b_nca.ipynb 的 Relaxed-OT 风格损失一致，并额外加入 Gram 项
    以稳定优化。计算在 device（默认 GPU）上进行；梯度会回流到 CPU 上的 phi。
    """

    def __init__(self, device="cuda", n_samples=1024):
        super().__init__()
        self.feat = VGGFeat(device)
        self.n_samples = n_samples

    @staticmethod
    def pairwise_distances_cos(x, y):
        x_norm = torch.norm(x, dim=2, keepdim=True)
        y_t = y.transpose(1, 2)
        y_norm = torch.norm(y_t, dim=1, keepdim=True)
        return 1. - torch.matmul(x, y_t) / (x_norm * y_norm + 1e-10)

    @staticmethod
    def relaxed_ot(x, y):
        d = StyleLoss.pairwise_distances_cos(x, y)
        m1, _ = d.min(1)
        m2, _ = d.min(2)
        return torch.max(m1.mean(dim=1), m2.mean(dim=1))

    @staticmethod
    def moment_loss(x, y):
        mu_x, mu_y = torch.mean(x, 1, keepdim=True), torch.mean(y, 1, keepdim=True)
        mu_diff = torch.abs(mu_x - mu_y).mean(dim=(1, 2))
        x_c, y_c = x - mu_x, y - mu_y
        x_cov = torch.matmul(x_c.transpose(1, 2), x_c) / (x.shape[1] - 1)
        y_cov = torch.matmul(y_c.transpose(1, 2), y_c) / (y.shape[1] - 1)
        cov_diff = torch.abs(x_cov - y_cov).mean(dim=(1, 2))
        return mu_diff + cov_diff

    def forward(self, generated_image, target_image):
        gf = self.feat(generated_image.float())
        with torch.no_grad():
            tf = self.feat(target_image.float())

        gram = 0.0
        ot = 0.0
        for x, y in zip(gf, tf):
            b, c, n = x.shape
            gx = torch.bmm(x, x.transpose(1, 2)) / n
            gy = torch.bmm(y, y.transpose(1, 2)) / n
            gram = gram + (gx - gy).abs().mean()

            n_s = min(n, y.shape[2], self.n_samples)
            ix = torch.argsort(torch.rand(b, 1, n, device=x.device), dim=-1)[..., :n_s]
            xg = x.gather(-1, ix.expand(b, c, n_s))
            iy = torch.argsort(torch.rand(b, 1, y.shape[2], device=y.device), dim=-1)[..., :n_s]
            yg = y.gather(-1, iy.expand(b, c, n_s))
            xg, yg = xg.transpose(1, 2), yg.transpose(1, 2)
            # relaxed_ot / moment_loss 返回逐样本 [B]，这里取均值得到标量
            ot = ot + StyleLoss.relaxed_ot(xg, yg).mean() \
                    + StyleLoss.moment_loss(xg, yg).mean()

        return gram / len(gf) + ot / len(gf)


class TextureMetrics(nn.Module):
    """评价指标（不参与梯度）：Gram-VGG19 距离与 VGG-SWD2 距离。

    参照论文 Sec. 4.2：
      · Gram-VGG19：取前三个最大池化层特征，按 F F^T / HW 计算相关矩阵并取
        平均绝对差；
      · VGG-SWD2：使用相同特征，每层 128 个固定单位高斯投影，种子 777。
    """

    def __init__(self, device="cuda", n_proj=128, seed=777):
        super().__init__()
        self.feat = VGGFeat(device)
        self.n_proj = n_proj
        self.seed = seed
        self.eval()

    @torch.no_grad()
    def _feats(self, img):
        return self.feat(img.float())

    @torch.no_grad()
    def gram_distance(self, a, b):
        fa, fb = self._feats(a), self._feats(b)
        d = 0.0
        for x, y in zip(fa, fb):
            gx = torch.bmm(x, x.transpose(1, 2)) / x.shape[2]
            gy = torch.bmm(y, y.transpose(1, 2)) / y.shape[2]
            d = d + (gx - gy).abs().mean()
        return d / len(fa)

    @torch.no_grad()
    def swd_distance(self, a, b):
        fa, fb = self._feats(a), self._feats(b)
        g = torch.Generator(device=fa[0].device).manual_seed(self.seed)
        d = 0.0
        for x, y in zip(fa, fb):
            b, c, n = x.shape
            P = torch.randn(c, self.n_proj, generator=g, device=x.device)
            P = P / (P.norm(dim=0, keepdim=True) + 1e-8)
            px = torch.matmul(x.transpose(1, 2), P)   # [B, HW, n_proj]
            py = torch.matmul(y.transpose(1, 2), P)
            px = px.sort(dim=1)[0]
            py = py.sort(dim=1)[0]
            d = d + (px - py).abs().mean()
        return d / len(fa)


# ==========================================================================
#                          数据加载（仅外表面）
# ==========================================================================
def move_params_to_device(params, device, freeze_gf=True):
    """把 params 内所有张量统一为 float32 并搬到 device。

    ProceduralParameters 不是 nn.Module，只能逐张量搬运；且各组 init_*/update_*
    随时可能新建 CPU 张量，故在初始化之后需要再调用一次。
    """
    for k, v in list(params.__dict__.items()):
        if isinstance(v, torch.Tensor):
            params.__dict__[k] = v.detach().to(device=device, dtype=torch.float32)

    if freeze_gf:
        for k in ("pith_origin", "pith_direction", "ref_vec",
                  "height_levels", "spoke_angs", "ring_rads", "ring_dists"):
            v = params.__dict__.get(k, None)
            if isinstance(v, torch.Tensor):
                v.requires_grad_(False)
    return params


def load_sample(root, sample_id, dim=256):
    """读取统一测试木块的『外表面观测』与阶段 3 的生长场。

    本函数不读取斜切面照片 —— 那是评价函数的专属数据源。
    """
    folder = os.path.join(root, "Samples", sample_id)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"样本目录不存在: {folder}")

    rgb_imgs, arl_imgs, _ = data_utils.get_cube_side_images(
        folder + os.sep, H=dim, W=dim)

    target = DataInstance(dim, dim, dim, TARGET=True)
    target.update_rgb_imgs_from_numpy(rgb_imgs)
    target.create_white_balanced_rgb_imgs()
    target.update_average_wb_rgb_color()
    target.update_arl_imgs_from_numpy(arl_imgs)
    target.update_average_arl_color()

    params_path = os.path.join(folder, "gf_params.pkl")
    if not os.path.isfile(params_path):
        raise FileNotFoundError(
            f"未找到生长场参数 {params_path}，请先运行: python 3_infer_gf.py -id {sample_id}")
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    # 年轮颜色初始化 —— 只使用外表面观测
    params.get_init_early_late_wood_colors(target.rgb_imgs_np)

    # 统一精度，并冻结生长场相关张量
    move_params_to_device(params, PROC_DEVICE, freeze_gf=True)

    return params, target


def get_coords(dim, device):
    """立方体 6 个面的 3D 采样坐标（搬到 device，float32）。"""
    return [c.to(device=device, dtype=torch.float32).contiguous()
            for c in data_utils.generate_cuboid_coordinates(dim, dim, dim)]


def get_face_targets(target, device):
    """6 个外表面的白平衡目标图像 —— 唯一的监督信号。"""
    return [t.to(device).float().contiguous() for t in target.wb_rgb_imgs_torch]


# ==========================================================================
#                     程序化纹理求值  F(G, A, phi)
# ==========================================================================
def render_faces(params, coords, dim, axes, show_fiber, show_pore, show_ray,
                 show_knot, use_color_map=False):
    """对 6 个外表面求值程序化纹理，返回 list[Tensor(dim, dim, 3)]。"""
    imgs = []
    for j, px_coords in enumerate(coords):
        px = px_coords.reshape(-1, 3)
        img = procedural_wood_function_refined_and_colors_and_details(
            params, px, side_index=j, side_axis=axes[j],
            A=dim, B=dim,
            show_fiber=show_fiber, show_pore=show_pore,
            show_ray=show_ray, show_knot=show_knot,
            color_map=use_color_map, return_reshaped=True)
        imgs.append(img)
    return imgs


def l1_image_loss(out_imgs, tgt_imgs):
    loss = 0.0
    for o, t in zip(out_imgs, tgt_imgs):
        loss = loss + (o - t).abs().mean()
    return loss / max(len(out_imgs), 1)


def mean_anchor_loss(out_imgs, target=0.5):
    """把每个外表面、每个通道的均值锚定到白平衡域的约定值 0.5。

    为什么需要：白平衡的定义是 `wb = orig / (2 * chan_mean)`，即每个面每个通道
    归一化后均值恒为 0.5。Stage A 用 L1 损失时这一约束天然满足；但切换到
    风格损失（Gram 矩阵 + Relaxed OT）后，这类统计对**整体亮度**几乎不敏感，
    颜色参数会整体漂移。若不显式锚定，还原（orig = wb * 2 * chan_mean）
    就会整体偏亮——实测漂移约 +15%。
    """
    d = 0.0
    for o in out_imgs:
        m = o.reshape(-1, 3).mean(dim=0)
        d = d + (m - target).abs().mean()
    return d / max(len(out_imgs), 1)


def style_loss_on_faces(loss_fn, out_imgs, tgt_imgs, device):
    """VGG 风格损失：把 6 个面拼成 batch 输入，与外表面观测比对。

    程序化求值得出的 outs 在 CPU 上，这里统一搬到 device 再过 VGG；
    反向传播时梯度会自动回流到 CPU 上的 phi。
    """
    # [H,W,3] -> [B,3,H,W]
    batch = torch.stack([o.permute(2, 0, 1) for o in out_imgs], dim=0
                                ).float().to(device)
    tbatch = torch.stack([t.permute(2, 0, 1) for t in tgt_imgs], dim=0
                                 ).float().to(device)
    return loss_fn(batch, tbatch)


# ==========================================================================
#                              体纹理生成
# ==========================================================================
def build_volume(params, dim=256, show_knot=False):
    """对整个立方体求值程序化纹理，返回 (dim,dim,dim,3) float16 numpy 体。

    沿用 3_infer_gf.py 的体构建方式：沿 z 轴逐层扫描并对齐朝向。
    """
    coords = get_coords(dim, PROC_DEVICE)
    vol = np.zeros((dim, dim, dim, 3), dtype=np.float16)
    base = coords[0].clone()
    for j in range(dim):
        px = base.reshape(-1, 3).clone()
        z_val = -0.5 + j * (1.0 / (dim - 1))
        px[:, 2] = z_val
        img = procedural_wood_function_refined_and_colors_and_details(
            params, px, side_index=0, side_axis=AXES[0],
            A=dim, B=dim, show_fiber=True, show_pore=True,
            show_ray=True, show_knot=show_knot,
            color_map=False, return_reshaped=True)
        img = img.detach().float().cpu().numpy()
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        vol[:, :, j] = img.astype(np.float16)
    return vol


def render_cut(vol, cut_coords, dim=256):
    """沿给定切面坐标对体纹理采样，返回 (dim,dim,3) uint8 图（白平衡域）。"""
    px = cut_coords.reshape(-1, 3).detach().cpu().numpy()
    idx = np.clip(255.0 * (px + 0.5), 0, dim - 1).astype(np.int64)
    img = vol[idx[:, 0], idx[:, 1], idx[:, 2]]
    img = img.reshape(dim, dim, 3)
    img = cv2.flip(img, 0)
    return img


def normalize_for_metric(img, eps=1e-6):
    """评价前对图像做白平衡归一化，消除光照/曝光差异。"""
    img = img.astype(np.float32)
    if float(img.max()) > 2.0:
        img = img / 255.0
    mean = img.reshape(-1, 3).mean(axis=0)
    out = img / (2.0 * (mean + eps))
    return np.clip(out, 0.0, 1.0)


def compare_metrics(a, b, device="cuda", tag=""):
    """比较两张图（均已在 [0,1] 且做过白平衡归一化）的若干指标。"""
    ta = torch.from_numpy(np.asarray(a)).float()
    tb = torch.from_numpy(np.asarray(b)).float()
    if ta.dim() == 3:
        ta = ta.permute(2, 0, 1).unsqueeze(0)
    if tb.dim() == 3:
        tb = tb.permute(2, 0, 1).unsqueeze(0)
    ta, tb = ta.to(device), tb.to(device)

    out = {
        "L1": float((ta - tb).abs().mean().item()),
        "RMSE": float(torch.sqrt(((ta - tb) ** 2).mean()).item()),
    }
    tm = TextureMetrics(device=device).eval()
    out["Gram-VGG19"] = float(tm.gram_distance(ta, tb).item())
    out["VGG-SWD2"] = float(tm.swd_distance(ta, tb).item())
    if tag:
        out = {f"{tag}/{k}": v for k, v in out.items()}
    return out


# ==========================================================================
#                      优化阶段（只用外表面观测）
# ==========================================================================
def run_optimize(stage_name, params, phi, coords, dim, tgt_faces, loss_fn,
                 device, iters, lr, group_names, use_style, show_details,
                 log, w_anchor=2.0, show_knot=False):
    """通用的分阶段优化循环。

    所有损失都在 6 个外表面上计算 —— 目标斜切面从不参与。

    w_anchor：亮度锚定权重。风格损失不约束整体亮度，故在 use_style 阶段
    叠加一项"每个面每个通道均值 → 0.5"的约束，防止颜色整体漂移。
    """
    ps = []
    for g in group_names:
        ps.extend(phi.groups()[g])
    opt = Adam(ps, lr=lr)

    kind = "风格损失" if use_style else "L1 图像损失"
    det = "含纤维/管孔/射线细节" if show_details else "仅年轮颜色"
    print(f"\n[{stage_name}] {kind}（{det}）  iters={iters}, lr={lr}")

    for i in tqdm(range(iters), desc=stage_name):
        opt.zero_grad()
        phi.apply(params)
        outs = render_faces(params, coords, dim, AXES,
                            show_fiber=show_details,
                            show_pore=show_details,
                            show_ray=show_details,
                            show_knot=show_knot)
        if use_style:
            loss = style_loss_on_faces(loss_fn, outs, tgt_faces, device)
            if w_anchor > 0.0:
                anchor = mean_anchor_loss(outs)
                loss = loss + w_anchor * anchor
        else:
            loss = l1_image_loss(outs, tgt_faces)
        loss.backward()
        opt.step()
        log.append((stage_name, i, float(loss.item())))

    final = float(loss.item())
    print(f"  {stage_name} 最终损失 = {final:.5f}")
    # 报告白平衡域均值是否仍锚定在 0.5（颜色还原正确性的关键）
    with torch.no_grad():
        phi.apply(params)
        outs = render_faces(params, coords, dim, AXES,
                            show_fiber=show_details,
                            show_pore=show_details,
                            show_ray=show_details,
                            show_knot=show_knot)
        m = torch.stack([o.reshape(-1, 3).mean(dim=0) for o in outs]).mean()
        print(f"  {stage_name} 白平衡域均值 = {float(m):.4f}"
              f"（目标 0.5000，偏差 {(float(m) / 0.5 - 1) * 100:+.1f}%）")
    return final


def grid_search_sizes(params, phi, coords, dim, tgt_faces, loss_fn, device,
                      grid=(-1.0, 0.0, 1.0), show_knot=False):
    """纤维 / 管孔 / 射线尺寸的粗网格搜索（Sec. 5.3.1 第三步）。

    评判同样只使用外表面观测。
    """
    print("\n[Stage C] 纤维 / 管孔 / 射线尺寸粗网格搜索")

    best = None
    for fv in tqdm(grid, desc="Stage C"):
        for pv in grid:
            for rv in grid:
                with torch.no_grad():
                    phi.fiber_disc.fill_(fv)
                    phi.pore_disc.fill_(pv)
                    phi.ray_disc.fill_(rv)
                    phi.apply(params)
                    outs = render_faces(params, coords, dim, AXES,
                                        show_fiber=True, show_pore=True,
                                        show_ray=True, show_knot=show_knot)
                    val = float(style_loss_on_faces(
                        loss_fn, outs, tgt_faces, device).item())
                if best is None or val < best[0]:
                    best = (val, fv, pv, rv)

    val, fv, pv, rv = best
    print(f"  网格最优: fiber={fv:+.1f} pore={pv:+.1f} ray={rv:+.1f}  "
          f"风格损失={val:.5f}")
    with torch.no_grad():
        phi.fiber_disc.fill_(fv)
        phi.pore_disc.fill_(pv)
        phi.ray_disc.fill_(rv)
        phi.apply(params)
    return (float(fv), float(pv), float(rv)), val


# ==========================================================================
#                      斜切面评价 —— 唯一的切面读取点
# ==========================================================================
def compute_metrics(rendered_u8, gt_u8, device="cuda"):
    """对已渲染好的 uint8 图与真值图计算指标。

    两张图都会各自做白平衡归一化，以消除拍摄曝光差异，
    使指标聚焦于纹理结构本身。
    """
    a = normalize_for_metric(rendered_u8)
    b = normalize_for_metric(gt_u8)
    ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device).float()
    tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(device).float()

    results = {
        "L1": float((ta - tb).abs().mean().item()),
        "RMSE": float(torch.sqrt(((ta - tb) ** 2).mean()).item()),
    }
    tm = TextureMetrics(device=device).eval()
    results["Gram-VGG19"] = float(tm.gram_distance(ta, tb).item())
    results["VGG-SWD2"] = float(tm.swd_distance(ta, tb).item())
    return results


def evaluate_on_slanted_cut(vol, sample_folder, cut_index=2, dim=256,
                            device="cuda"):
    """在目标斜切面上评价生成的体纹理。

    ============================ 协议边界 ============================
    这是本脚本中唯一读取目标斜切面照片（cut1/cut2_col.png）的地方。
    该图像仅用于计算并报告指标，不参与任何优化、适配、网格搜索或
    颜色提示（color hint）提取。
    ==================================================================
    """
    cutA_coords, cutD_coords = data_utils.generate_cuboid_coordinates_cutsAD(dim)
    cut_coords = cutA_coords if cut_index == 1 else cutD_coords

    # vol 为 float16 [0,1]，转成 uint8 便于 OpenCV 写出与后续比较
    rendered = np.clip(
        255.0 * render_cut(vol, cut_coords, dim).astype(np.float32),
        0, 255).astype(np.uint8)

    gt_path = os.path.join(sample_folder, f"cut{cut_index}_col.png")
    if not os.path.isfile(gt_path):
        return None
    gt = cv2.imread(gt_path)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt = cv2.resize(gt, (dim, dim), interpolation=cv2.INTER_AREA)

    return compute_metrics(rendered, gt, device=device), rendered


# ==========================================================================
#                              主流程
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Step 4a: inverse procedural texturing (Mokume)")
    parser.add_argument("-id", type=str, default="CN03",
                        help="统一测试木块 ID（默认 CN03，含目标斜切面照片）")
    parser.add_argument("-show", type=int, default=1,
                        help="是否显示优化过程")
    parser.add_argument("-res", type=int, default=128,
                        help="优化阶段每个面的分辨率（默认 128）")
    parser.add_argument("-iters", type=int, default=0,
                        help="各阶段迭代次数（0 = 使用默认 A100/B100/D200）")
    parser.add_argument("-knot", type=int, default=0, choices=[0, 1],
                        help="是否启用节疤（0=关，默认；1=开，要求 gf_params "
                             "已由 Step3 带掩膜拟合出 knot 参数；输出文件加 "
                             "_knot 后缀，不覆盖主结果）")
    parser.add_argument("-torch_home", type=str,
                        default=r"D:\CODE\moke\hub",
                        help="预训练权重缓存目录")
    parser.add_argument("-device", type=str, default="",
                        help="风格损失所用设备：cuda / cpu（默认自动选择）")
    args = parser.parse_args()

    torch_home = args.torch_home
    os.makedirs(torch_home, exist_ok=True)
    os.environ["TORCH_HOME"] = torch_home
    torch.hub.set_dir(os.path.join(torch_home, "hub"))

    device = args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print(" Step 4a — inverse 3D procedural texturing (Mokume)")
    print("=" * 78)
    print(f"  测试木块(统一)  : {args.id}")
    print(f"  输入(仅外表面)  : Samples/{args.id}/{{A..F}}_col.png")
    print(f"  生长场(冻结)    : Samples/{args.id}/gf_params.pkl")
    print(f"  评价(仅切面)    : Samples/{args.id}/cut2_col.png  <- 只用于报指标")
    print(f"  权重缓存目录    : {torch_home}")
    print(f"  风格损失设备    : {device}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if device.startswith("cuda") else ""))
    print(f"  程序化求值设备  : cpu（与 3_infer_gf.py 一致）")
    print("=" * 78)

    dim = args.res
    # 注意：这里只加载外表面观测；斜切面照片在评价函数中才被读取
    params, target = load_sample(HERE, args.id, dim=dim)
    tgt_faces = get_face_targets(target, PROC_DEVICE)
    coords = get_coords(dim, PROC_DEVICE)

    use_knot = bool(args.knot)
    if use_knot and not hasattr(params, "knot_deformations"):
        raise RuntimeError(
            f"样本 {args.id} 的 gf_params.pkl 不含节疤参数（Step3 未带掩膜拟合）。"
            f"需先将官方 *_knot.png 拷入样本目录并重跑 3_infer_gf.py。")
    phi = PhiParams(knot=use_knot)
    # 顺序：先 init_* 建立 base_* 备份，再 apply() 写入 phi
    with torch.no_grad():
        params.init_fiber_parameters()
        params.init_pore_parameters()
        params.init_ray_parameters()
        if use_knot:
            # 接续生长场已拟合的节疤几何/颜色（不从零开始）：
            # update_knot_* 是 base + delta 形式，把 phi 初值设为 gf 现值的
            # 逆映射，保证首次 apply 后体渲染与 gf 拟合结果一致。
            phi.knot_deform.data.copy_(params.knot_deformations)
            if hasattr(params, "knot_color_bar") and params.knot_color_bar is not None:
                phi.knot_col_bar.data.copy_(params.knot_color_bar - 0.5)
            if (hasattr(params, "knot_color_anisotrophy_factor")
                    and params.knot_color_anisotrophy_factor is not None):
                phi.knot_ani.data.copy_(
                    (params.knot_color_anisotrophy_factor - 0.5) / 5.0)
            print("  [knot] 已启用节疤：接续 gf_params 的节疤参数")
        move_params_to_device(params, PROC_DEVICE, freeze_gf=True)
        phi.apply(params)

    print("\nphi 参数分组：")
    tot = 0
    for k, v in phi.named_groups().items():
        print(f"  {k:18s}: {v:4d}")
        tot += v
    print(f"  {'合计':18s}: {tot:4d}")

    # 风格损失在 device（GPU）上跑；梯度会回流到 CPU 上的 phi
    loss_fn = StyleLoss(device=device).eval()

    A_ITERS = args.iters if args.iters > 0 else 100
    B_ITERS = args.iters if args.iters > 0 else 100
    D_ITERS = args.iters if args.iters > 0 else 200
    LR = 0.02

    log = []

    # Stage A：连续优化 + L1 图像损失 —— 初始化年轮颜色（无细节）
    run_optimize("Stage A", params, phi, coords, dim, tgt_faces, loss_fn,
                 device, A_ITERS, LR, ["ring_color"],
                 use_style=False, show_details=False, log=log,
                 show_knot=use_knot)

    # Stage B：切换到 VGG 风格损失，继续优化颜色（仍无细节）
    run_optimize("Stage B", params, phi, coords, dim, tgt_faces, loss_fn,
                 device, B_ITERS, LR,
                 ["ring_color", "ring_color_fine", "transition_blend"],
                 use_style=True, show_details=False, log=log,
                 show_knot=use_knot)

    # Stage C：纤维 / 管孔 / 射线尺寸的粗网格搜索（开细节）
    sizes, val = grid_search_sizes(params, phi, coords, dim, tgt_faces,
                                   loss_fn, device, show_knot=use_knot)

    # Stage D：全部 phi 连续优化（开细节 + 风格损失）
    d_groups = ["ring_color", "ring_color_fine", "transition_blend",
                "fiber", "pore", "ray"] + (["knot"] if use_knot else [])
    run_optimize("Stage D", params, phi, coords, dim, tgt_faces, loss_fn,
                 device, D_ITERS, LR, d_groups,
                 use_style=True, show_details=True, log=log,
                 show_knot=use_knot)

    # ---------------- 生成体纹理并保存 ----------------
    print("\n生成 256^3 程序化体纹理 ...")
    t0 = time.time()
    with torch.no_grad():
        phi.apply(params)
        vol = build_volume(params, dim=256, show_knot=use_knot)
    print(f"  用时 {time.time() - t0:.1f}s")

    # 颜色还原正确性检查：白平衡域均值必须锚定在 0.5，
    # 否则按 orig = wb * 2 * chan_mean 还原时会整体偏亮/偏暗。
    vm = float(vol.reshape(-1, 3).mean())
    print(f"  白平衡域均值 = {vm:.4f}（目标 0.5000，偏差 {(vm / 0.5 - 1) * 100:+.1f}%）")
    if abs(vm - 0.5) > 0.02:
        print("  [警告] 均值偏离超过 2%，颜色还原会整体偏色；"
              "已按实际均值做锚定校正。")
        vol = np.clip(vol * (0.5 / vm), 0.0, 1.0).astype(np.float16)
        print(f"  校正后均值 = {float(vol.reshape(-1, 3).mean()):.4f}")

    folder = os.path.join(HERE, "Samples", args.id)
    sfx = "_knot" if use_knot else ""
    np.savez_compressed(os.path.join(folder, f"proc_cube{sfx}.npz"), vol)
    print(f"  已保存 proc_cube{sfx}.npz -> {folder}")

    params.detach_tensors()
    with open(os.path.join(folder, f"proc_params{sfx}.pkl"), "wb") as f:
        pickle.dump(params, f)
    print(f"  已保存 proc_params{sfx}.pkl -> {folder}")

    # 外表面渲染（含细节）快速查看图
    #
    # 注意：体纹理本身工作在白平衡域（每通道均值 0.5，即视觉上的中性灰），
    # 木头的棕色是在白平衡时被"除掉"的。因此导出可视化图之前必须先做颜色还原
    # orig = wb / wb.mean() * chan_mean，否则图会是灰的，
    # 容易被误读成"模型生成不出木材颜色"。
    with torch.no_grad():
        coords256 = get_coords(256, PROC_DEVICE)
        outs = render_faces(params, coords256, 256, AXES,
                            show_fiber=True, show_pore=True,
                            show_ray=True, show_knot=use_knot)
        chan_means = np.asarray(target.channel_means_np, dtype=np.float32) / 255.0
        tiles = []
        for o, cm in zip(outs, chan_means):
            img = o.detach().float().cpu().numpy().astype(np.float32)
            # 按实际均值锚定还原（等价推广：wb.mean()==0.5 时即 orig = wb*2*cm）
            img = img / (img.reshape(-1, 3).mean(axis=0) + 1e-6) * cm
            tiles.append(np.clip(255.0 * img, 0, 255).astype(np.uint8))
        strip = np.hstack(tiles)
        cv2.imwrite(os.path.join(folder, f"proc_faces{sfx}.png"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"  已保存 proc_faces{sfx}.png（已还原颜色）-> {folder}")

    # ---------------- 外表面拟合误差（仅作对照，非评价指标） ----------------
    print("\n[对照] 外表面重建误差（训练/适配实际使用的观测，仅作拟合优度参考）")
    with torch.no_grad():
        phi.apply(params)
        # 用与优化阶段相同的分辨率渲染，保证与目标图尺寸一致
        outs = render_faces(params, coords, dim, AXES,
                            show_fiber=True, show_pore=True,
                            show_ray=True, show_knot=use_knot)
        ext = []
        for o, t in zip(outs, tgt_faces):
            ta = o.permute(2, 0, 1).unsqueeze(0).float().to(device)
            tb = t.permute(2, 0, 1).unsqueeze(0).float().to(device)
            ext.append(float((ta - tb).abs().mean().item()))
        print("  外表面 L1 = " + ", ".join(
            f"{LETTERS[i]}:{v:.4f}" for i, v in enumerate(ext)))
        print(f"  外表面 L1 均值 = {float(np.mean(ext)):.5f}")

    # ---------------- 斜切面评价（唯一读取切面的地方） ----------------
    print("\n[评价] 目标斜切面 cut2（仅用于报指标）")
    out = evaluate_on_slanted_cut(vol, folder, cut_index=2, dim=256,
                          device=device)
    if out is None:
        print("  该样本没有目标斜切面照片，跳过评价。")
    else:
        metrics, rendered = out
        cv2.imwrite(os.path.join(folder, f"proc_cut2_render{sfx}.png"),
                    cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        with open(os.path.join(folder, f"proc_eval{sfx}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print("  ---------------------------------------------")
        print("  目标斜切面 cut2 上的指标：")
        for k, v in metrics.items():
            print(f"    {k:14s}: {v:.5f}")
        print("  ---------------------------------------------")
        print(f"  已保存 proc_cut2_render{sfx}.png / proc_eval{sfx}.json -> {folder}")

    print("\n完成。")


if __name__ == "__main__":
    print(torch.__version__)
    main()
