# -*- coding: utf-8 -*-
"""colorlib.py — 色差诊断/显示对齐共用工具（纯 numpy + cv2，不碰 GPU）。

严格留出协议：本文件所有统计只接受"六外表面 A-F_col.png"与"proc_cube.npz"
（其定色参数 gf_params.pkl 已冻结）。函数本身不读取 cut*_col.png 以外的
任何目标切面信息 —— 目标切面仅在评价端作为参考图出现。
"""
import os

import numpy as np
import cv2

DIM = 256
LETTERS = "ABCDEF"
FLIP180 = {"H11", "H14", "MP07", "N07", "SG15", "SG16"}


# ---------------------------------------------------------------- 基础 IO
def load_npz(folder, name):
    p = os.path.join(folder, name)
    return np.load(p)["arr_0"] if os.path.isfile(p) else None


def imread_rgb(path, size=None):
    im = cv2.imread(path)
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if size:
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return im.astype(np.float32) / 255.0


def render_cut(vol, cut_coords, dim=DIM, rotate90=0):
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


def load_proc_slice(folder, sid, cut, cutA, cutD, dim=DIM):
    fn = ("proc_cube_knot.npz"
          if sid == "H11" and os.path.isfile(os.path.join(folder, "proc_cube_knot.npz"))
          else "proc_cube.npz")
    vol = load_npz(folder, fn)
    if vol is None:
        return None
    r = render_cut(vol, cutA if cut == 1 else cutD, dim,
                   rotate90=1 if cut == 1 else 0)
    if sid in FLIP180:
        r = np.ascontiguousarray(np.rot90(r, 2))
    return r


# ---------------------------------------------------------------- 白平衡域
def white_balance(x, eps=1e-6):
    """按通道均值归一化（与 4a_eval_cut.py / make_montage38.py 逐字一致）。"""
    x = np.asarray(x, dtype=np.float32)
    if x.max() > 2.0:
        x = x / 255.0
    mean = x.reshape(-1, 3).mean(axis=0)
    return np.clip(x / (2.0 * (mean + eps)), 0.0, 1.0)


def unwhitebalance(img, chan_mean, eps=1e-6):
    """白平衡域→显示域（现行 make_montage38.py 的口径）。"""
    img = np.asarray(img, dtype=np.float32)
    m = img.reshape(-1, 3).mean(axis=0)
    return img / (m + eps) * np.asarray(chan_mean, dtype=np.float32)


# ---------------------------------------------------------------- 外表面统计
def face_images(folder, letters=LETTERS):
    out = {}
    for ltr in letters:
        im = imread_rgb(os.path.join(folder, f"{ltr}_col.png"))
        if im is not None:
            out[ltr] = im
    return out


def face_chan_mean(faces):
    """现行 montage 口径：六面各自通道均值，再对 6 面取平均（一阶 3 维统计）。"""
    if not faces:
        return np.array([0.5, 0.5, 0.5], np.float32)
    means = [im.reshape(-1, 3).mean(axis=0) for im in faces.values()]
    return np.mean(means, axis=0).astype(np.float32)


def face_chan_mean_fast(folder, letters=LETTERS):
    """与 face_chan_mean 等价但只读 uint8、用 cv2.mean 求均值（不转 float32 整图）。

    用于只需一阶统计的场合，省内存/时间；结果与原口径差异 < 1e-5。
    """
    means = []
    for ltr in letters:
        fp = os.path.join(folder, f"{ltr}_col.png")
        if os.path.isfile(fp):
            im = cv2.imread(fp)
            if im is None:
                continue
            mu_bgr = np.array(cv2.mean(im)[:3], np.float64)
            means.append(mu_bgr[::-1] / 255.0)  # BGR -> RGB
    if not means:
        return np.array([0.5, 0.5, 0.5], np.float32)
    return np.mean(means, axis=0).astype(np.float32)


def face_pixels(faces, max_pix=150000, step=None):
    """把六面像素拼成 (N,3)，用于拟合由外表面估计的颜色映射。"""
    pix = [im.reshape(-1, 3) for im in faces.values()]
    allp = np.concatenate(pix, axis=0)
    if step is None:
        step = max(1, allp.shape[0] // max_pix)
    return np.ascontiguousarray(allp[::step], dtype=np.float32)


def fit_face_affine(faces, mode="full", max_pix=150000):
    """只用外表面拟合 "白平衡域 → 原始相机域" 的颜色映射。

    mode="full" : raw ≈ M @ wb + b   （3x3 跨通道，12 参数）
    mode="diag" : raw_c ≈ a_c*wb_c + b_c（逐通道，6 参数）
    返回 (apply_fn, mae, params)
    """
    raw = face_pixels(faces, max_pix=max_pix)
    # 六面合并后做一次全局白平衡，尺度与 proc_cube 的白平衡域一致
    mu_all = raw.mean(axis=0)
    wbv = np.clip(raw / (2.0 * (mu_all + 1e-6)), 0.0, 1.0)

    X = np.concatenate([wbv, np.ones((len(wbv), 1), np.float32)], axis=1)
    if mode == "full":
        W, *_ = np.linalg.lstsq(X, raw, rcond=None)   # (4,3)
        pred = X @ W
        M, b = W[:3].T, W[3]
    else:
        M = np.zeros((3, 3), np.float32)
        b = np.zeros(3, np.float32)
        for c in range(3):
            A = np.stack([wbv[:, c], np.ones(len(wbv), np.float32)], 1)
            sol, *_ = np.linalg.lstsq(A, raw[:, c], rcond=None)
            M[c, c], b[c] = sol[0], sol[1]
        pred = wbv @ M.T + b
    mae = float(np.abs(pred - raw).mean())
    # 逐通道均值/标准差匹配系数（用于诊断报告）
    return M.astype(np.float32), b.astype(np.float32), mae, mu_all.astype(np.float32)


def apply_affine(img, M, b):
    h, w, _ = img.shape
    x = img.reshape(-1, 3).astype(np.float32)
    y = x @ M.T + b
    return np.clip(y.reshape(h, w, 3), 0.0, 1.0)


# ---------------------------------------------------------------- 颜色统计
def lab(img):
    return cv2.cvtColor(np.clip(img, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2Lab)


def hsv(img):
    return cv2.cvtColor(np.clip(img, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2HSV)


def chroma_hue(img, cmin=3.0):
    l = lab(img)
    a, b = l[..., 1], l[..., 2]
    C = np.sqrt(a * a + b * b)
    m = C > cmin
    if m.sum() < 10:
        return float(C.mean()), float("nan")
    ang = np.arctan2(b[m], a[m])
    vx = float((np.cos(ang) * C[m]).sum())
    vy = float((np.sin(ang) * C[m]).sum())
    return float(C.mean()), float(np.degrees(np.arctan2(vy, vx)) % 360.0)


def mean_color_lab(img):
    m = img.reshape(-1, 3).mean(axis=0)[None, None, :]
    l = lab(m)[0, 0]
    return l  # (L,a,b)


def de76(l1, l2):
    d = np.asarray(l1, np.float32) - np.asarray(l2, np.float32)
    return float(np.sqrt((d * d).sum(-1)).mean())


def mean_sat(img):
    return float(hsv(img)[..., 1].mean())


def contrast_L(img):
    return float(lab(img)[..., 0].std())


def lowfreq(img, sigma=16.0):
    return cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)


def de_lowfreq(x, y, sigma=16.0, lab_space=True):
    """低频 ΔE：两侧先高斯模糊再加权，衡量"整块颜色观感"差异（对结构免疫）。"""
    xb, yb = lowfreq(x, sigma), lowfreq(y, sigma)
    if lab_space:
        d = lab(xb) - lab(yb)
        return float(np.sqrt((d * d).sum(axis=2)).mean())
    return float(np.abs(xb - yb).mean() * 255.0)


def hue_diff_deg(h1, h2):
    if not np.isfinite(h1) or not np.isfinite(h2):
        return float("nan")
    d = abs(h1 - h2) % 360.0
    return float(min(d, 360.0 - d))
