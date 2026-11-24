# -*- coding: utf-8 -*-
"""
================================================================================
V14.py -- 双输入网络 地表覆盖错分/漏分 检测-标注-自评价 (校准修正 + 双源候选)
          在 V8/V9/V11/V12/V13 基础上改写
================================================================================
研究目标: 双输入(遥感X + 地表覆盖P) -> 定位错误 -> 定性(错分/漏分a) ->
          供人工确认的核查清单 -> 基于双输入的自评价

V14 相对 V13 的两处补丁 + 连带修复:
  [Q1] 校准只跨模型, 不跨注入实现
       - V13 错误: 对 9 张(3模型×3注入)分数求均值, 却用 seed100 真值校准 -> 抹平
       - V14 正确: 对每个注入实现 es, 只对 3 个模型求均值 -> 逐实现校准 -> 取中位数
       - 连带: D00 诊断图 / 分数分布摘要 改用"单实现模型集合", 口径一致
  [Q2] 部署改用双源候选集, 不再只靠绝对阈值
       - 候选 = 分类头不一致集(disagree, 默认) / 误差头高分(score) / 并 / 交
       - 误差分数只用于排序与分级, 不再单独决定"是否候选"
       - 新增阈值敏感性表 + 候选模式对比表 + 两张图
       - 分级改用候选集内相对分位(默认 top25%), 避免绝对 0.7 在真实产品上失效
  [连带] L1 文案改为 "AUC≈0.5, 无判别力"(实测 0.47–0.52), 仅诊断不参与分级

保留不变: 网络拓扑 839,721 参数 / 注入 RNG 顺序 / 训练超参 /
          Stage II 分类目标 clean / 验证模式分位数决策 / 图件无 Easting/Northing
================================================================================
用法:
  # 完整流程
  python V14.py --rs 2010rssdlqstac.tif --glc 2010glc_sdlq.tif --out ./out

  # 仅部署(复用检查点+缓存验证分数), 默认 disagree 候选
  python V14.py --out ./out --deploy-only --rs ... --glc ...

  # 切换候选模式 / 调整分级分位
  python V14.py --out ./out --deploy-only --rs ... --glc ... \
      --candidate-mode union --score-hi-quantile 0.10
================================================================================
"""
import os
import re
import sys
import json
import time
import glob
import argparse
import platform
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage as ndi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, Rectangle, Polygon

from sklearn.metrics import (cohen_kappa_score, roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.features import shapes as rio_shapes
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
VERSION = "V14.0"

# =============================================================================
# 0. 配置
# =============================================================================
NODATA = 65535
C = 8
CLASS_MAP = {10: 0, 25: 1, 30: 2, 80: 3, 27: 4, 60: 5, 90: 6, 50: 7}
CLASS_NAME_EN = {0: "Cropland", 1: "Forest", 2: "Grassland", 3: "Artificial",
                 4: "Shrubland", 5: "Water", 6: "Bareland", 7: "Wetland"}
CLASS_NAME_CN = {0: "耕地", 1: "林地", 2: "草地", 3: "人工地表",
                 4: "灌木地", 5: "水体", 6: "裸地", 7: "湿地"}
CLASS_WEIGHT = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 2.0, 6: 1.0, 7: 2.0}
TRANS = {0: [2], 1: [4], 2: [0], 3: [0], 4: [1], 5: [7], 6: [3], 7: [5]}

PALETTE_LC = np.array([
    [0.89, 0.76, 0.58], [0.11, 0.47, 0.21], [0.50, 0.75, 0.48], [0.84, 0.19, 0.15],
    [0.69, 0.88, 0.54], [0.27, 0.46, 0.75], [1.00, 0.88, 0.55], [0.35, 0.66, 0.80]], np.float32)
NODATA_GREY = np.array([0.82, 0.82, 0.84], np.float32)
CMAP_TP = (0.0, 0.8, 0.3)
CMAP_FP = (1.0, 0.2, 0.2)
CMAP_FN = (0.0, 0.4, 1.0)
TIER_COLOR = {"A": (0.0, 0.7, 0.3), "B": (1.0, 0.75, 0.0), "C": (0.85, 0.2, 0.2)}
BAR_COLORS = ["#3A7CA5", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948", "#B07AA1", "#9C755F"]
C_BASE, C_FULL = "#3A7CA5", "#D7263D"
C_VAL, C_DEP = "#D7263D", "#3A7CA5"
MM = 1.0 / 25.4
COL1 = 90 * MM
COL2 = 183 * MM
FIG_DPI = 600
DEC_R, DEC_S = "raw", "5x5 smoothed"
CAND_MODES = ["disagree", "score", "union", "intersect"]


def set_pub_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.linewidth": 0.8, "lines.linewidth": 1.2, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": FIG_DPI, "figure.dpi": 150})


def save_fig(fig, stem, tight=False):
    kw = dict(bbox_inches="tight", pad_inches=0.05) if tight else {}
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    fig.savefig(stem + ".pdf", **kw)
    fig.savefig(stem + ".png", dpi=FIG_DPI, **kw)
    plt.close(fig)


def panel_label(ax, label, x=-0.02, y=1.0):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, fontweight="bold")


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _env_meta():
    return {"pipeline_version": VERSION, "python": platform.python_version(),
            "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _peak_mem():
    out = {"peak_gpu_mem_MB": None, "peak_ram_MB": None}
    if torch.cuda.is_available():
        out["peak_gpu_mem_MB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["peak_ram_MB"] = round(ru / 2 ** 20 if sys.platform == "darwin" else ru / 1024, 1)
    except Exception:
        pass
    return out


def _jd(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_jd)


def log(m):
    print(m, flush=True)


# =============================================================================
# 1. 地理配准 + 制图家具 (仅比例尺 / 指北针; 无坐标网)
# =============================================================================
GEO = {"px": 30.0, "crs": None, "transform": None}


def set_geo(transform=None, crs=None, px=None):
    if transform is not None:
        GEO["transform"] = transform
        GEO["px"] = float(abs(transform.a))
    if crs is not None:
        GEO["crs"] = crs
    if px is not None:
        GEO["px"] = float(px)


def _resolve_geo(px, transform):
    return (GEO["px"] if px is None else float(px),
            GEO["transform"] if transform is None else transform)


def _nice_length(target):
    if target <= 0:
        return 1.0
    e = np.floor(np.log10(target))
    best = 10.0 ** e
    for m in (1, 2, 5, 10):
        if m * 10.0 ** e <= target:
            best = m * 10.0 ** e
    return float(best)


def _north_vec(transform):
    if transform is None:
        return np.array([0.0, -1.0])
    A = np.array([[transform.a, transform.b], [transform.d, transform.e]], float)
    try:
        v = np.linalg.solve(A, np.array([0.0, 1.0]))
    except Exception:
        return np.array([0.0, -1.0])
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([0.0, -1.0])


def add_scale_bar(ax, shape, px=None, transform=None, frac=0.25,
                  x0=0.045, y0=0.935, fontsize=6.8):
    H, W = shape[:2]
    px, transform = _resolve_geo(px, transform)
    if px and px > 0:
        L_m = _nice_length(frac * W * px)
        L_px = L_m / px
        div, unit = (1000.0, "km") if L_m >= 1000 else (1.0, "m")
        texts = ["0", f"{L_m/2/div:g}", f"{L_m/div:g} {unit}"]
    else:
        L_px = _nice_length(frac * W)
        texts = ["0", f"{L_px/2:g}", f"{L_px:g} px"]
    x, y = x0 * W, y0 * H
    h = max(0.010 * H, 1.5)
    halo = [pe.withStroke(linewidth=2.0, foreground="white")]
    ax.add_patch(Rectangle((x - .4 * h, y - .4 * h), L_px + .8 * h, h + .8 * h,
                           fc="white", ec="none", alpha=.85, zorder=9))
    for i, fc in enumerate(("black", "white")):
        ax.add_patch(Rectangle((x + i * L_px / 2, y), L_px / 2, h,
                               fc=fc, ec="black", lw=.6, zorder=10))
    for xx, t in zip((x, x + L_px / 2, x + L_px), texts):
        ax.text(xx, y - .6 * h, t, ha="center", va="bottom",
                fontsize=fontsize, zorder=11, path_effects=halo)


def add_north_arrow(ax, shape, transform=None, x0=0.94, y0=0.115,
                    size=0.07, fontsize=7.0):
    H, W = shape[:2]
    _, transform = _resolve_geo(None, transform)
    v = _north_vec(transform)
    p = np.array([-v[1], v[0]])
    L, w = size * H, 0.30 * size * H
    c = np.array([x0 * W, y0 * H])
    head, tail = c + .5 * L * v, c - .5 * L * v
    ax.add_patch(Polygon(np.vstack([head, tail + w * p, tail - w * p]),
                         closed=True, fc="black", ec="white", lw=.8, zorder=11))
    pt = head + .55 * L * v
    ax.text(pt[0], pt[1], "N", ha="center", va="center", fontsize=fontsize,
            fontweight="bold", zorder=11,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])


def decorate(ax, shape):
    add_scale_bar(ax, shape)
    add_north_arrow(ax, shape)
    ax.set_axis_off()


def geo_caption(shape=None):
    px, tr = GEO["px"], GEO["transform"]
    crs = GEO["crs"]
    bits = []
    if crs is not None:
        bits.append(f"CRS: {crs.to_string() if hasattr(crs,'to_string') else crs}")
    if px:
        bits.append(f"pixel size {px:g} m")
    if shape is not None and px:
        bits.append(f"extent {shape[1]*px/1000:.2f} x {shape[0]*px/1000:.2f} km")
    return ("; ".join(bits) + ".") if bits else ""


def _lc_to_rgb(prod, valid=None):
    bad = (prod < 0) | (prod >= C)
    if valid is not None:
        bad = bad | (~valid)
    idx = np.clip(np.where(bad, 0, prod).astype(np.int64), 0, C - 1)
    rgb = PALETTE_LC[idx].copy()
    rgb[bad] = NODATA_GREY
    return np.clip(rgb, 0, 1)


def lc_legend_handles():
    hs = [Patch(facecolor=tuple(PALETTE_LC[c]), edgecolor="#333333", lw=.5,
                label=CLASS_NAME_EN[c]) for c in range(C)]
    hs.append(Patch(facecolor=tuple(NODATA_GREY), edgecolor="#333333", lw=.5, label="NoData"))
    return hs


def err_legend_handles():
    return [Patch(facecolor=(*CMAP_TP, 1.), edgecolor="#333333", lw=.5, label="TP (detected)"),
            Patch(facecolor=(*CMAP_FP, 1.), edgecolor="#333333", lw=.5, label="FP (false alarm)"),
            Patch(facecolor=(*CMAP_FN, 1.), edgecolor="#333333", lw=.5, label="FN (missed)")]


def tier_legend_handles():
    lab = {"A": "Tier A (high conf.)", "B": "Tier B (medium)", "C": "Tier C (low)"}
    return [Patch(facecolor=TIER_COLOR[t], edgecolor="#333333", lw=.5, label=lab[t])
            for t in ("A", "B", "C")]


def error_map_rgba(pred, gt, valid=None, solid=False):
    p = (pred > 0)
    g = (gt > 0)
    ok = np.ones(gt.shape, bool) if valid is None else valid
    rgba = np.zeros((*gt.shape, 4), np.float32)
    rgba[p & g] = (*CMAP_TP, .95)
    rgba[p & ~g] = (*CMAP_FP, .95)
    rgba[~p & g] = (*CMAP_FN, .95)
    if solid:
        rgba[~p & ~g & ok] = (.965, .965, .975, 1.0)
        rgba[~ok] = (*NODATA_GREY, 1.0)
    else:
        rgba[~p & ~g] = (0, 0, 0, 0)
        rgba[~ok] = (0, 0, 0, 0)
    return rgba


def tier_map_rgb(tier, valid, base=0.965):
    ev = np.ones((*tier.shape, 3), np.float32) * base
    for t, col in TIER_COLOR.items():
        ev[tier == t] = col
    ev[~valid] = NODATA_GREY
    return ev


# =============================================================================
# 2. 数据加载 + 注入器 (RNG 顺序与 V8 一致)
# =============================================================================
def load_scene(rs_path, glc_path):
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required")
    with rasterio.open(rs_path) as ds:
        raw = ds.read().astype(np.float32)
        H, W = ds.height, ds.width
        crs, tr = ds.crs, ds.transform
        set_geo(transform=tr, crs=crs)
        nd = (raw == NODATA).any(0) | (~np.isfinite(raw)).any(0)
        vals = raw[:, ~nd].ravel() if nd.any() else raw.ravel()
        vals = vals[np.isfinite(vals)]
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
        img = np.clip((raw - lo) / (hi - lo + 1e-6), 0, 1)
        img[:, nd] = 0.0
        img = np.moveaxis(img, 0, -1).astype(np.float32)
    with rasterio.open(glc_path) as ds:
        g = ds.read(1)
        src_tr, src_crs = ds.transform, ds.crs
    lut = np.full(65536, 255, np.uint8)
    for code, idx in CLASS_MAP.items():
        lut[int(code)] = int(idx)
    src = lut[np.clip(g.astype(np.int64), 0, 65535)]
    prod_u8 = np.full((H, W), 255, np.uint8)
    reproject(src, prod_u8, src_transform=src_tr, src_crs=src_crs,
              dst_transform=tr, dst_crs=crs, src_nodata=255, dst_nodata=255,
              resampling=Resampling.nearest)
    valid = (prod_u8 != 255) & (~nd)
    prod = np.full((H, W), -1, np.int64)
    prod[valid] = prod_u8[valid].astype(np.int64)
    return img, prod, valid


def make_synthetic_scene(H=512, W=512, seed=0):
    rng = np.random.default_rng(seed)
    field = ndi.gaussian_filter(rng.normal(size=(H, W)), sigma=12)
    cuts = np.quantile(field, np.linspace(0, 1, C + 1)[1:-1])
    prod = np.digitize(field, cuts).astype(np.int64)
    base = np.array([[0.55, 0.42, 0.30], [0.18, 0.45, 0.16], [0.40, 0.55, 0.32],
                     [0.62, 0.60, 0.58], [0.30, 0.52, 0.26], [0.12, 0.22, 0.45],
                     [0.72, 0.68, 0.55], [0.25, 0.40, 0.50]], np.float32)
    noise = ndi.gaussian_filter(rng.normal(size=(H, W, 3)).astype(np.float32),
                               sigma=(3, 3, 0)) * 0.06
    img = np.clip(base[prod] + noise, 0, 1).astype(np.float32)
    valid = np.ones((H, W), bool)
    valid[:8, :] = False; valid[-8:, :] = False
    valid[:, :8] = False; valid[:, -8:] = False
    prod[~valid] = -1
    set_geo(px=30.0)
    return img, prod, valid


def inject_pixel_mix_safe(prod, valid, rng, target=0.06):
    """CANONICAL injector. RNG 调用顺序与 V7/V8/V9 一致, 不可重排。"""
    prod2 = prod.copy()
    H, W = prod.shape
    valid_mask = valid & (prod2 >= 0) & (prod2 < C)
    y = np.zeros(prod.shape, bool)
    budget = int(target * valid_mask.sum())
    cap_patch = int(budget * 0.55)
    tries = 0
    while y.sum() < cap_patch and tries < 4000:
        tries += 1
        r, c = int(rng.integers(0, H)), int(rng.integers(0, W))
        if not valid_mask[r, c] or y[r, c]:
            continue
        cf = int(prod2[r, c])
        targets = [d for d in TRANS.get(cf, []) if d != cf]
        if not targets:
            continue
        ct = int(targets[int(rng.integers(0, len(targets)))])
        size = int(np.clip(rng.lognormal(3.4, 1.0), 3, 100))
        rr, cc = [r], [c]
        for _ in range(size):
            k = int(rng.integers(0, len(rr)))
            nr = rr[k] + int(rng.integers(-1, 2))
            nc = cc[k] + int(rng.integers(-1, 2))
            if 0 <= nr < H and 0 <= nc < W:
                rr.append(nr); cc.append(nc)
        rr, cc = np.asarray(rr), np.asarray(cc)
        m = valid_mask[rr, cc] & (~y[rr, cc]) & (prod2[rr, cc] == cf)
        rr, cc = rr[m], cc[m]
        if rr.size == 0 or y.sum() + rr.size > cap_patch + 200:
            continue
        prod2[rr, cc] = ct
        y[rr, cc] = True
    rest = budget - int(y.sum())
    cand = np.where(valid_mask & (~y))
    n = min(rest, cand[0].size)
    if n > 0:
        sel = rng.choice(cand[0].size, n, replace=False)
        rr, cc = cand[0][sel], cand[1][sel]
        for i in range(n):
            cf = int(prod2[rr[i], cc[i]])
            targets = [d for d in TRANS.get(cf, []) if d != cf]
            if targets:
                prod2[rr[i], cc[i]] = int(targets[int(rng.integers(0, len(targets)))])
                y[rr[i], cc[i]] = True
    return np.clip(prod2, 0, C - 1).astype(np.int64), y


def _boundary_mask(prod, valid):
    p = np.where(valid, prod, -1).astype(np.int32)
    return (ndi.maximum_filter(p, 3) != ndi.minimum_filter(p, 3)) & valid


def inject_alt(prod, valid, rng, target=0.06, mode="random_class"):
    prod2 = prod.copy()
    H, W = prod.shape
    vm = valid & (prod2 >= 0) & (prod2 < C)
    y = np.zeros(prod.shape, bool)
    budget = int(target * vm.sum())
    share = 0.0 if mode == "scattered_only" else 0.55
    cap = int(budget * share)

    def _succ(cf):
        if mode == "random_class":
            return int((cf + 1 + int(rng.integers(0, C - 1))) % C)
        t = [d for d in TRANS.get(cf, []) if d != cf]
        return int(t[int(rng.integers(0, len(t)))]) if t else None

    tries = 0
    while y.sum() < cap and tries < 4000:
        tries += 1
        r, c = int(rng.integers(0, H)), int(rng.integers(0, W))
        if not vm[r, c] or y[r, c]:
            continue
        cf = int(prod2[r, c]); ct = _succ(cf)
        if ct is None:
            continue
        size = int(np.clip(rng.lognormal(3.4, 1.0), 3, 100))
        rr, cc = [r], [c]
        for _ in range(size):
            k = int(rng.integers(0, len(rr)))
            nr = rr[k] + int(rng.integers(-1, 2)); nc = cc[k] + int(rng.integers(-1, 2))
            if 0 <= nr < H and 0 <= nc < W:
                rr.append(nr); cc.append(nc)
        rr, cc = np.asarray(rr), np.asarray(cc)
        m = vm[rr, cc] & (~y[rr, cc]) & (prod2[rr, cc] == cf)
        rr, cc = rr[m], cc[m]
        if rr.size == 0 or y.sum() + rr.size > cap + 200:
            continue
        prod2[rr, cc] = ct; y[rr, cc] = True
    rest = budget - int(y.sum())
    pool = vm & (~y)
    if mode == "edge_targeted":
        eb = _boundary_mask(prod2, vm) & (~y)
        if eb.sum() >= rest > 0:
            pool = eb
    cand = np.where(pool)
    n = min(rest, cand[0].size)
    if n > 0:
        sel = rng.choice(cand[0].size, n, replace=False)
        rr, cc = cand[0][sel], cand[1][sel]
        for i in range(n):
            cf = int(prod2[rr[i], cc[i]]); ct = _succ(cf)
            if ct is not None:
                prod2[rr[i], cc[i]] = ct; y[rr[i], cc[i]] = True
    return np.clip(prod2, 0, C - 1).astype(np.int64), y


# =============================================================================
# 3. 网络 (拓扑与 V8 一致, 839,721 参数)
# =============================================================================
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class ASPPErrHead(nn.Module):
    def __init__(self, in_ch, mid=32, dilations=(1, 3, 6)):
        super().__init__()
        self.glob = nn.Sequential(nn.Conv2d(in_ch, mid, 1, bias=False),
                                  nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, mid, 3, padding=d, dilation=d, bias=False),
                          nn.BatchNorm2d(mid), nn.ReLU(inplace=True)) for d in dilations])
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * (1 + len(dilations)), mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True), nn.Conv2d(mid, 1, 1))

    def forward(self, x):
        return self.fuse(torch.cat([self.glob(x)] + [b(x) for b in self.branches], 1))


class DualResUNetV6(nn.Module):
    def __init__(self, in_ch=3, n_classes=C, base=32, emb_dim=16, use_label_stream=True):
        super().__init__()
        self.n_classes = n_classes
        self.use_label_stream = use_label_stream
        self.emb = nn.Embedding(n_classes, emb_dim)
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.p1 = ConvBlock(emb_dim, base)
        self.p2 = ConvBlock(base, base * 2)
        self.p3 = ConvBlock(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = ConvBlock(base * 6 if use_label_stream else base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = ConvBlock(base * 3 if use_label_stream else base * 2, base)
        self.cls_head = nn.Conv2d(base, n_classes, 1)
        self.err_head = ASPPErrHead(base + 2, mid=max(base, 32))

    def forward(self, x, prod):
        prod_c = torch.clamp(prod.long(), 0, self.n_classes - 1)
        p = self.emb(prod_c).permute(0, 3, 1, 2)
        i1 = self.e1(x)
        i2 = self.e2(F.max_pool2d(i1, 2))
        i3 = self.e3(F.max_pool2d(i2, 2))
        if self.use_label_stream:
            p1 = self.p1(p)
            p2 = self.p2(F.max_pool2d(p1, 2))
            p3 = self.p3(F.max_pool2d(p2, 2))
            d2 = self.d2(torch.cat([i2, p2, self.up2(i3 + p3)], 1))
            d1 = self.d1(torch.cat([i1, p1, self.up1(d2)], 1))
        else:
            d2 = self.d2(torch.cat([i2, self.up2(i3)], 1))
            d1 = self.d1(torch.cat([i1, self.up1(d2)], 1))
        logits_cls = self.cls_head(d1)
        prob = F.softmax(logits_cls, dim=1)
        entropy = -(prob * (prob + 1e-8).log()).sum(1, keepdim=True) / float(np.log(self.n_classes))
        mismatch = (logits_cls.argmax(1) != prod_c).float().unsqueeze(1)
        return logits_cls, self.err_head(torch.cat([d1, entropy, mismatch], 1))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# =============================================================================
# 4. 数据集 + 训练 (Stage II 分类目标 = clean)
# =============================================================================
class _PatchBase(Dataset):
    def __init__(self, img, prod, valid, patch=128, n=1500, seed=0):
        self.img = img
        self.prod = np.clip(np.where(prod >= 0, prod, 0).astype(np.int64), 0, C - 1)
        self.valid = valid & (prod >= 0) & (prod < C)
        self.patch = patch
        rng = np.random.default_rng(seed)
        H, W = prod.shape
        self.coords = [(int(rng.integers(0, max(H - patch, 1))),
                        int(rng.integers(0, max(W - patch, 1)))) for _ in range(n)]

    def __len__(self):
        return len(self.coords)


class ConsistencyDataset(_PatchBase):
    def __init__(self, *args, flip_ratio=0.30, seed=0, **kw):
        super().__init__(*args, seed=seed, **kw)
        self.flip_ratio = flip_ratio
        self.rng = np.random.default_rng(seed + 1)

    def __getitem__(self, i):
        r, c = self.coords[i]; p = self.patch
        im = self.img[r:r + p, c:c + p].transpose(2, 0, 1)
        clean = self.prod[r:r + p, c:c + p]
        v = self.valid[r:r + p, c:c + p]
        noisy = clean.copy()
        hit = (self.rng.random(noisy.shape) < self.flip_ratio) & v
        alt = self.rng.integers(0, C, noisy.shape)
        same = alt == noisy
        alt[same] = (alt[same] + 1) % C
        noisy[hit] = alt[hit]
        return (torch.from_numpy(np.ascontiguousarray(im)).float(),
                torch.from_numpy(noisy).long(), torch.from_numpy(clean).long(),
                torch.from_numpy(v).float(), torch.from_numpy(hit.astype(np.float32)).float())


class AnomalyDataset(_PatchBase):
    def __init__(self, *args, target=0.06, seed=0, **kw):
        super().__init__(*args, seed=seed, **kw)
        self.target = target
        self.rng = np.random.default_rng(seed + 2)

    def __getitem__(self, i):
        r, c = self.coords[i]; p = self.patch
        im = self.img[r:r + p, c:c + p].transpose(2, 0, 1)
        pn = self.prod[r:r + p, c:c + p]
        v = self.valid[r:r + p, c:c + p]
        clean = pn.copy()
        pn, y_err = inject_pixel_mix_safe(pn, v, self.rng, self.target)
        return (torch.from_numpy(np.ascontiguousarray(im)).float(),
                torch.from_numpy(pn).long(), torch.from_numpy(clean).long(),
                torch.from_numpy(y_err).float(), torch.from_numpy(v).float())


def _weights(device, use_class_weight=True):
    w = torch.ones(C)
    if use_class_weight:
        for k, v in CLASS_WEIGHT.items():
            w[k] = v
    return w.to(device)


def _masked_ce(logits, target, v_mask, weight):
    return (F.cross_entropy(logits, target, weight=weight, reduction="none")
            * v_mask).sum() / (v_mask.sum() + 1e-8)


def _masked_bce_dice(logits, target, v_mask, dice_w=1.0):
    bce = (F.binary_cross_entropy_with_logits(logits.squeeze(1), target, reduction="none")
           * v_mask).sum() / (v_mask.sum() + 1e-8)
    p = torch.sigmoid(logits.squeeze(1)) * v_mask
    t = target * v_mask
    return bce + dice_w * (1.0 - (2.0 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0))


def train_stage1(model, img, prod, valid, epochs=12, lr=1e-3, bs=4, n_patch=1500,
                 patch=128, seed=0, tag="Stage I", err_warmup=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl = DataLoader(ConsistencyDataset(img, prod, valid, patch=patch, n=n_patch, seed=seed),
                    batch_size=bs, shuffle=True, drop_last=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    model.to(device); cw = _weights(device)
    log(f"[{tag}] pre-training ({epochs} ep)")
    for ep in range(epochs):
        use_err = ep >= epochs - err_warmup
        model.train()
        for im, noisy, clean, v, flip in dl:
            im, noisy, clean, v, flip = (im.to(device), noisy.to(device), clean.to(device),
                                         v.to(device), flip.to(device))
            lc, le = model(im, noisy)
            loss = _masked_ce(lc, clean, v, cw)
            if use_err:
                loss = loss + _masked_bce_dice(le, flip, v)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    return model


def train_stage2(model, img, prod, valid, epochs=25, lr=1e-3, bs=4, n_patch=1500,
                 patch=128, use_class_weight=True, lam=2.0, target=0.06, seed=0,
                 tag="Stage II", jitter=0.03, cls_target="clean", cls_w=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    model.to(device); cw = _weights(device, use_class_weight)
    rng_j = np.random.default_rng(seed + 500)
    log(f"[{tag}] fine-tuning ({epochs} ep, cls_target={cls_target}, cls_w={cls_w})")
    for ep in range(epochs):
        cur = float(np.clip(target + rng_j.uniform(-jitter, jitter), 0.02, min(0.12, target * 2)))
        dl = DataLoader(AnomalyDataset(img, prod, valid, patch=patch, n=n_patch,
                                       target=cur, seed=seed + ep),
                        batch_size=bs, shuffle=True, drop_last=True, num_workers=0)
        model.train()
        for im, pn, cl, y_err, v in dl:
            im, pn, cl, y_err, v = (im.to(device), pn.to(device), cl.to(device),
                                    y_err.to(device), v.to(device))
            lc, le = model(im, pn)
            tgt = cl if cls_target == "clean" else pn
            loss = cls_w * _masked_ce(lc, tgt, v, cw) + lam * _masked_bce_dice(le, y_err, v)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    return model


# =============================================================================
# 5. 推理 / 决策 / 集合 / 阈值校准
# =============================================================================
@torch.no_grad()
def infer_full(model, img, prod, patch=128):
    device = next(model.parameters()).device
    model.eval()
    H, W, _ = img.shape
    score = np.zeros((H, W), np.float32)
    pred = np.zeros((H, W), np.int64)
    conf = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    prod_i = np.where(prod >= 0, prod, 0)
    for r in range(0, H, patch):
        for c in range(0, W, patch):
            r1, c1 = min(r + patch, H), min(c + patch, W)
            im = img[r:r1, c:c1]; pr = prod_i[r:r1, c:c1]
            ph, pw = patch - im.shape[0], patch - im.shape[1]
            if ph or pw:
                im = np.pad(im, ((0, ph), (0, pw), (0, 0)), mode="reflect")
                pr = np.pad(pr, ((0, ph), (0, pw)), mode="reflect")
            t_im = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float()[None].to(device)
            t_pr = torch.from_numpy(np.ascontiguousarray(pr)).long()[None].to(device)
            lc, le = model(t_im, t_pr)
            prob = F.softmax(lc, dim=1)
            s = torch.sigmoid(le)[0, 0].cpu().numpy()[:r1 - r, :c1 - c]
            pc = prob.argmax(1)[0].cpu().numpy()[:r1 - r, :c1 - c]
            cf = prob.max(1)[0][0].cpu().numpy()[:r1 - r, :c1 - c]
            score[r:r1, c:c1] += s; cnt[r:r1, c:c1] += 1
            pred[r:r1, c:c1] = pc; conf[r:r1, c:c1] = cf
    return score / np.maximum(cnt, 1), pred, conf


@torch.no_grad()
def infer_score(model, img, prod, patch=128):
    return infer_full(model, img, prod, patch)[0]


def post_process_direct(score, valid, detect_ratio=0.10):
    """验证模式: 预算精确分位数 (仅在已知流行率的合成协议下有效)。"""
    thr = np.quantile(score[valid], 1 - detect_ratio)
    return (score >= thr) & valid


def post_process_global(score, valid, detect_ratio=0.10):
    s = ndi.uniform_filter(score, size=5)
    thr = np.quantile(s[valid], 1 - detect_ratio)
    return (s >= thr) & valid


def decide_absolute(score, valid, thr):
    return (score >= float(thr)) & valid


def decide(score, valid, detect_ratio=0.10, thr=None):
    return decide_absolute(score, valid, thr) if thr is not None \
        else post_process_direct(score, valid, detect_ratio)


def _decide(score, valid, ratio):
    return post_process_direct(score, valid, ratio)


def ensemble_predict(models, img, prod, patch=128):
    """集合均值分数 + 多数投票共识类别 + 类别一致率 (只跨模型)。"""
    scores, preds = [], []
    for m in models:
        sc, pc, cf = infer_full(m, img, prod, patch)
        scores.append(sc); preds.append(pc)
    scores = np.stack(scores, 0)
    preds = np.stack(preds, 0)
    sc_ens = scores.mean(0).astype(np.float32)
    pc_ens = np.apply_along_axis(lambda a: np.bincount(a, minlength=C).argmax(),
                                 0, preds).astype(np.int64)
    agree = np.mean(preds == pc_ens[None], axis=0).astype(np.float32)
    return sc_ens, pc_ens, agree, scores, preds


def stability_from_runs(flag_stack, pred_stack):
    freq = np.mean(flag_stack, axis=0).astype(np.float32)
    ps = np.stack(pred_stack, 0)
    mode_cls = np.apply_along_axis(lambda a: np.bincount(a, minlength=C).argmax(), 0, ps)
    agree = np.mean(ps == mode_cls[None], axis=0).astype(np.float32)
    return freq, agree


def calibrate_threshold(score_val, gt_val, valid, target_precision=0.90):
    """在验证集上求满足 precision>=target 的最大标记量所对应的分数阈值。"""
    s = np.asarray(score_val, np.float64)[valid]
    y = np.asarray(gt_val, bool)[valid].astype(np.int32)
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    ss = s[order]
    tp = np.cumsum(ys)
    n = np.arange(1, ys.size + 1, dtype=np.float64)
    prec = tp / n
    ok = np.where(prec >= target_precision)[0]
    if ok.size == 0:
        thr = float(np.quantile(s, 0.999))
        k = int(np.searchsorted(-ss, -thr))
        note = "target precision unreachable; fell back to 99.9th percentile"
    else:
        k = int(ok[-1])
        thr = float(ss[k])
        note = "largest flagged set meeting the target precision"
    n_flag = k + 1
    res = {"target_precision": float(target_precision),
           "threshold": thr,
           "val_n_flagged": int(n_flag),
           "val_flag_rate_pct": float(100.0 * n_flag / max(valid.sum(), 1)),
           "val_precision_pct": float(100.0 * prec[k]),
           "val_recall_pct": float(100.0 * tp[k] / max(int(y.sum()), 1)),
           "note": note}
    return thr, res


def threshold_curve(score_val, gt_val, valid, thrs=None):
    if thrs is None:
        thrs = np.unique(np.concatenate([np.linspace(0.001, 0.05, 25),
                                         np.linspace(0.06, 0.99, 60)]))
    s = np.asarray(score_val, np.float64)[valid]
    y = np.asarray(gt_val, bool)[valid]
    n_err = max(int(y.sum()), 1)
    rows = []
    for t in thrs:
        m = s >= t
        nf = int(m.sum())
        tp = int((m & y).sum())
        rows.append({"threshold": float(t), "n_flagged": nf,
                     "flag_rate_pct": 100.0 * nf / max(valid.sum(), 1),
                     "precision_pct": 100.0 * tp / max(nf, 1),
                     "recall_pct": 100.0 * tp / n_err})
    return pd.DataFrame(rows)


# =============================================================================
# 6. 指标
# =============================================================================
METRIC_KEYS = ["OA", "Kappa", "Precision", "Recall", "F1", "IoU"]


def _confusion(pred, gt, valid):
    ok = np.asarray(valid, bool).ravel()
    p = (np.asarray(pred) > 0).ravel()[ok]
    g = (np.asarray(gt) > 0).ravel()[ok]
    return (int(np.logical_and(p, g).sum()), int(np.logical_and(p, ~g).sum()),
            int(np.logical_and(~p, g).sum()), int(np.logical_and(~p, ~g).sum()))


def evaluate_change(pred, gt, valid=None):
    if valid is None:
        valid = np.ones(gt.shape, bool)
    tp, fp, fn, tn = _confusion(pred, gt, valid)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    oa = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    try:
        ok = np.asarray(valid, bool).ravel()
        kappa = cohen_kappa_score((np.asarray(gt) > 0).ravel()[ok],
                                 (np.asarray(pred) > 0).ravel()[ok])
    except Exception:
        kappa = 0.0
    prevalence = (tp + fn) / max(tp + fp + fn + tn, 1)
    return {"OA": float(oa), "Kappa": float(kappa), "Precision": float(precision),
            "Recall": float(recall), "F1": float(f1), "IoU": float(iou),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "prevalence": float(prevalence),
            "enrichment": float(precision / prevalence if prevalence > 0 else 0.0),
            "flag_rate": float((tp + fp) / max(tp + fp + fn + tn, 1)),
            "FPR": float(fp / max(fp + tn, 1))}


def aggregate_seeds(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}


def discrimination(score, gt, valid):
    ok = np.asarray(valid, bool).ravel()
    s = np.asarray(score, np.float64).ravel()[ok]
    y = (np.asarray(gt) > 0).ravel()[ok].astype(np.uint8)
    fin = np.isfinite(s); s, y = s[fin], y[fin]
    if y.size == 0 or y.min() == y.max():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


# =============================================================================
# 7. 有向标注 + 类别级错分/漏分
# =============================================================================
def annotate_from_candidates(prod_presented, pred_cls, candidate):
    """[Q2] 候选集已确定, 直接生成有向标注 (不再内部阈值)。"""
    P = np.clip(prod_presented, 0, C - 1)
    from_map = np.where(candidate, P, -1).astype(np.int64)
    to_map = np.where(candidate & (pred_cls != P), pred_cls, -1).astype(np.int64)
    code = np.zeros(prod_presented.shape, np.int32)
    has_dir = candidate & (to_map >= 0)
    code[has_dir] = (from_map[has_dir] * 10 + to_map[has_dir]).astype(np.int32)
    code[candidate & ~has_dir] = 99
    return from_map, to_map, candidate, code


def annotate_directed(prod_presented, pred_cls, score, valid,
                      detect_ratio=0.10, thr=None):
    flagged = decide(score, valid, detect_ratio, thr)
    return annotate_from_candidates(prod_presented, pred_cls, flagged)


def commission_omission(prod_presented, pred_cls, flagged, valid, out_dir,
                        px=None, tag=""):
    px = GEO["px"] if px is None else px
    P = np.clip(prod_presented, 0, C - 1)
    dir_mask = flagged & (pred_cls != P) & valid
    T = np.zeros((C, C), np.int64)
    np.add.at(T, (P[dir_mask], pred_cls[dir_mask]), 1)
    np.savetxt(os.path.join(out_dir, f"transition_8x8{tag}.csv"), T, delimiter=",",
               header=",".join(CLASS_NAME_EN[c] for c in range(C)), comments="", fmt="%d")
    n_prod = np.array([int((valid & (P == c)).sum()) for c in range(C)])
    n_pred = np.array([int((valid & (pred_cls == c)).sum()) for c in range(C)])
    rows = []
    for c in range(C):
        comm = int(T[c, :].sum()); omis = int(T[:, c].sum())
        rows.append({"Class": CLASS_NAME_EN[c], "Class_CN": CLASS_NAME_CN[c],
                     "product_area_px": n_prod[c], "image_evidence_area_px": n_pred[c],
                     "commission_px": comm, "omission_px": omis,
                     "commission_rate_pct": 100 * comm / max(n_prod[c], 1),
                     "omission_rate_pct": 100 * omis / max(n_pred[c], 1),
                     "net_area_bias_px": omis - comm,
                     "net_area_bias_km2": (omis - comm) * px * px / 1e6,
                     "interpretation": ("over-mapped" if comm > omis else
                                        "under-mapped" if omis > comm else "balanced")})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"commission_omission{tag}.csv"),
              index=False, encoding="utf-8-sig")
    log(f"\n[类别级 错分/漏分{tag}]\n" + df.round(3).to_string(index=False))
    return df, T


# =============================================================================
# 8. 自评价证据层 (L1 仅诊断; L4 独立)
# =============================================================================
def _class_stats(img, prod, valid, exclude=None):
    B = img.shape[2]
    mu = np.zeros((C, B)); cov = np.zeros((C, B, B)); n = np.zeros(C, int)
    base = valid & (prod >= 0) & (prod < C)
    if exclude is not None:
        base = base & (~exclude)
    for c in range(C):
        m = base & (prod == c)
        if m.sum() < 30:
            m = valid & (prod == c)
        x = img[m].astype(np.float64); n[c] = len(x)
        if len(x) > B + 1:
            mu[c] = x.mean(0); cov[c] = np.cov(x.T) + 1e-4 * np.eye(B)
        else:
            mu[c] = x.mean(0) if len(x) else np.zeros(B); cov[c] = np.eye(B)
    Sw = np.zeros((B, B)); tot = 0
    for c in range(C):
        if n[c] > B + 1:
            Sw += (n[c] - 1) * cov[c]; tot += n[c] - 1
    if tot > 0:
        Sw = Sw / tot + 1e-4 * np.eye(B)
        for c in range(C):
            cov[c] = Sw
    return mu, cov, n


def L1_spectral_likelihood(img, prod_presented, from_map, to_map, flagged, valid,
                           out_dir, mode="presented", tag=""):
    """[诊断] 实测判别 AUC≈0.47–0.52, 无信息, 不参与分级。"""
    excl = flagged if mode == "presented" else None
    mu, cov, n = _class_stats(img, prod_presented, valid, exclude=excl)
    inv = {c: np.linalg.inv(cov[c]) for c in range(C)}
    dir_mask = flagged & (to_map >= 0)
    ys, xs = np.where(dir_mask)
    lam = np.full(prod_presented.shape, np.nan, np.float32)
    vals = []
    for y, x in zip(ys, xs):
        f = int(from_map[y, x]); t = int(to_map[y, x])
        xv = img[y, x].astype(np.float64)
        df_ = xv - mu[f]; dt_ = xv - mu[t]
        v = 0.5 * (float(df_ @ inv[f] @ df_) - float(dt_ @ inv[t] @ dt_))
        lam[y, x] = v; vals.append(v)
    vals = np.asarray(vals)
    res = {"mode": mode,
           "role": "DIAGNOSTIC ONLY (AUC~0.5, no discriminative power; excluded from tiering)",
           "n_directed": int(len(vals)),
           "frac_spectral_support": float((vals > 0).mean()) if len(vals) else None,
           "mean_lambda": float(vals.mean()) if len(vals) else None,
           "median_lambda": float(np.median(vals)) if len(vals) else None,
           "frac_lambda_gt1": float((vals > 1).mean()) if len(vals) else None}
    jdump(res, os.path.join(out_dir, f"L1_spectral{tag}.json"))
    log(f"\n[L1 光谱似然比{tag} | 仅诊断] " + json.dumps(res, ensure_ascii=False, default=_jd))
    return lam, res


def L2_directed_accuracy(from_map, to_map, flagged, prod_clean, prod_presented,
                         y_true, valid, out_dir):
    inj = y_true & valid
    tp = flagged & inj
    true_from = np.clip(prod_presented, 0, C - 1)
    true_to = np.clip(prod_clean, 0, C - 1)
    dir_ok = tp & (to_map == true_to) & (from_map == true_from)
    n_tp = int(tp.sum()); n_dir = int(dir_ok.sum())
    m = tp & (to_map >= 0); n_covered = int(m.sum())
    per_class = []
    for c in range(C):
        mc_all = tp & (true_from == c); nc_all = int(mc_all.sum())
        mc = m & (true_from == c); nc = int(mc.sum())
        okc = int((mc & (to_map == true_to)).sum())
        per_class.append({"Class": CLASS_NAME_EN[c], "n_tp": nc_all, "n_covered": nc,
                          "coverage_pct": 100 * nc / max(nc_all, 1) if nc_all else None,
                          "conditional_acc_pct": 100 * okc / nc if nc else None,
                          "directed_acc_pct": 100 * okc / nc_all if nc_all else None})
    res = {"localization_recall_pct": 100 * n_tp / max(int(inj.sum()), 1),
           "directed_accuracy_pct": 100 * n_dir / max(n_tp, 1),
           "coverage_pct": 100 * n_covered / max(n_tp, 1),
           "conditional_accuracy_pct": 100 * n_dir / max(n_covered, 1),
           "n_injected": int(inj.sum()), "n_tp": n_tp,
           "n_covered": n_covered, "n_directed_correct": n_dir,
           "per_class_directed": per_class}
    jdump(res, os.path.join(out_dir, "L2_directed.json"))
    pd.DataFrame(per_class).to_csv(os.path.join(out_dir, "L2_directed_perclass.csv"),
                                   index=False, encoding="utf-8-sig")
    log("\n[L2 有向准确率] " + json.dumps(
        {k: v for k, v in res.items() if k != "per_class_directed"},
        ensure_ascii=False, default=_jd))
    return res


def L3_dual_readout(flagged, pred_cls, prod_presented, valid, out_dir, tag=""):
    P = np.clip(prod_presented, 0, C - 1)
    cls_disagree = (pred_cls != P) & valid
    try:
        kappa = cohen_kappa_score(flagged.ravel().astype(np.uint8),
                                 cls_disagree.ravel().astype(np.uint8))
    except Exception:
        kappa = 0.0
    inter = int((flagged & cls_disagree).sum())
    res = {"error_head_flagged": int(flagged.sum()),
           "cls_head_disagree": int(cls_disagree.sum()),
           "cls_head_disagree_rate_pct": 100 * int(cls_disagree.sum()) / max(int(valid.sum()), 1),
           "both": inter, "agreement_kappa": float(kappa),
           "frac_flagged_with_class_support": inter / max(int(flagged.sum()), 1)}
    jdump(res, os.path.join(out_dir, f"L3_dual{tag}.json"))
    log(f"\n[L3 双流互证{tag}] " + json.dumps(res, ensure_ascii=False, default=_jd))
    return res


def L4_context(prod_presented, to_map, flagged, valid, out_dir, win=5, tag=""):
    """独立于网络: 邻域支持来自"未标记像元的产品标签"。"""
    P = np.clip(prod_presented, 0, C - 1)
    base = valid & (~flagged)
    den = ndi.uniform_filter(base.astype(np.float32), size=win)
    support = np.zeros(prod_presented.shape, np.float32)
    for c in range(C):
        num = ndi.uniform_filter(((P == c) & base).astype(np.float32), size=win)
        support = np.where(to_map == c, num / np.maximum(den, 1e-6), support)
    dir_mask = flagged & (to_map >= 0)
    vals = support[dir_mask]
    res = {"window": win, "basis": "unflagged product labels (network-independent)",
           "mean_neighborhood_support": float(vals.mean()) if len(vals) else None,
           "frac_support_gt0.3": float((vals > 0.3).mean()) if len(vals) else None}
    jdump(res, os.path.join(out_dir, f"L4_context{tag}.json"))
    log(f"\n[L4 空间上下文{tag}] " + json.dumps(res, ensure_ascii=False, default=_jd))
    return support, res


# =============================================================================
# 9. 置信度分级 + 核查清单 (P2/P3/P4 保留)
# =============================================================================
def confidence_tiers(score, support, freq, agree, to_map, flagged,
                     score_hi=0.7, agree_hi=0.8, freq_hi=0.66, ctx_hi=0.3):
    """验证模式: 绝对高分阈值。L1 不参与。"""
    tier = np.full(score.shape, "", dtype="<U1")
    has_dir = to_map >= 0
    ctx_ok = support > ctx_hi
    stab_ok = (freq >= freq_hi) & (agree >= agree_hi)
    high = score >= score_hi
    A = flagged & has_dir & stab_ok & ctx_ok & high
    B = flagged & ~A & has_dir & ((agree >= agree_hi) | ctx_ok)
    Cc = flagged & ~A & ~B
    tier[A] = "A"; tier[B] = "B"; tier[Cc] = "C"
    return tier


def confidence_tiers_relative(score, support, freq, agree, to_map, candidate,
                              score_hi_quantile=0.25, agree_hi=0.8,
                              freq_hi=0.66, ctx_hi=0.3):
    """[Q2] 部署模式: 候选集内相对分位定高分, 避免绝对 0.7 在真实产品上失效。"""
    tier = np.full(score.shape, "", dtype="<U1")
    vals = score[candidate]
    thr_hi = float(np.quantile(vals, 1.0 - score_hi_quantile)) if vals.size else 1.0
    has_dir = to_map >= 0
    ctx_ok = support > ctx_hi
    stab_ok = (freq >= freq_hi) & (agree >= agree_hi)
    high = score >= thr_hi
    A = candidate & has_dir & stab_ok & ctx_ok & high
    B = candidate & ~A & has_dir & ((agree >= agree_hi) | ctx_ok)
    Cc = candidate & ~A & ~B
    tier[A] = "A"; tier[B] = "B"; tier[Cc] = "C"
    return tier, thr_hi


def _tier_by_fraction(tv, min_frac=0.5):
    n = max(len(tv), 1)
    fA = tv.count("A") / n
    fAB = (tv.count("A") + tv.count("B")) / n
    return "A" if fA >= min_frac else ("B" if fAB >= min_frac else "C")


def label_directed_objects(from_map, to_map, flagged):
    """对象只由有向像元构成, 按 (from,to) 分别连通 -> 不可能 from==to。"""
    dir_mask = flagged & (to_map >= 0)
    pair = np.where(dir_mask, from_map * C + to_map, -1).astype(np.int32)
    lab = np.zeros(flagged.shape, np.int32)
    pair_of_id = {}
    n_tot = 0
    for pv in np.unique(pair[dir_mask]):
        if pv < 0:
            continue
        l, n = ndi.label(pair == pv, structure=np.ones((3, 3), int))
        if n == 0:
            continue
        lab[l > 0] = l[l > 0] + n_tot
        for k in range(1, n + 1):
            pair_of_id[n_tot + k] = (int(pv) // C, int(pv) % C)
        n_tot += n
    return lab, pair_of_id, n_tot


def export_review_checklist(img, prod_presented, pred_cls, score, from_map, to_map,
                            flagged, tier, lam, support, freq, agree, valid, out_dir,
                            max_items=100000, min_obj_px=1, t_item_s=20.0, px=None,
                            crs=None, transform=None, tag="", tier_min_frac=0.5):
    px = GEO["px"] if px is None else px
    crs = GEO["crs"] if crs is None else crs
    transform = GEO["transform"] if transform is None else transform
    tier_rank = {"A": 0, "B": 1, "C": 2, "": 3}

    lab, pair_of_id, n_obj = label_directed_objects(from_map, to_map, flagged)
    items = []
    if n_obj > 0:
        objs = ndi.find_objects(lab)
        for i, sl in enumerate(objs, start=1):
            if sl is None or i not in pair_of_id:
                continue
            m = (lab[sl] == i)
            npix = int(m.sum())
            if npix < min_obj_px:
                continue
            ys, xs = np.where(m)
            ys = ys + sl[0].start; xs = xs + sl[1].start
            fr, to = pair_of_id[i]
            cy, cx = float(ys.mean()), float(xs.mean())
            sc = float(score[ys, xs].mean())
            tv = [tier[y, x] for y, x in zip(ys, xs)]
            tg = _tier_by_fraction(tv, tier_min_frac)
            lv = lam[ys, xs]; lv = lv[np.isfinite(lv)]
            items.append({
                "item_id": i, "tier": tg, "row": round(cy, 1), "col": round(cx, 1),
                "n_pixels": npix, "area_m2": round(npix * px * px, 1),
                "from_class": CLASS_NAME_EN[fr], "to_class": CLASS_NAME_EN[to],
                "from_CN": CLASS_NAME_CN[fr], "to_CN": CLASS_NAME_CN[to],
                "error_type": f"{CLASS_NAME_CN[fr]}错分 / {CLASS_NAME_CN[to]}漏分",
                "mean_error_score": round(sc, 4),
                "min_error_score": round(float(score[ys, xs].min()), 4),
                "spectral_lambda_diag": round(float(lv.mean()), 3) if len(lv) else None,
                "neighbor_support": round(float(support[ys, xs].mean()), 3),
                "stability_freq": round(float(freq[ys, xs].mean()), 3),
                "class_agree": round(float(agree[ys, xs].mean()), 3),
                "frac_A": round(tv.count("A") / npix, 3),
                "frac_B": round(tv.count("B") / npix, 3),
                "frac_C": round(tv.count("C") / npix, 3)})
    df = pd.DataFrame(items)
    if len(df):
        df["_tr"] = df["tier"].map(tier_rank)
        df = df.sort_values(["_tr", "mean_error_score", "n_pixels"],
                            ascending=[True, False, False]).drop(columns="_tr")
        df = df.head(max_items).reset_index(drop=True)
        df.insert(0, "review_priority", np.arange(1, len(df) + 1))
        if transform is not None:
            df["x_coord"] = (transform.c + transform.a * (df["col"] + 0.5)).round(2)
            df["y_coord"] = (transform.f + transform.e * (df["row"] + 0.5)).round(2)
            df["crs"] = str(crs)
        df["review_action"] = df["tier"].map({
            "A": "快速确认(证据充分)", "B": "仔细确认(部分支持)", "C": "重点核实(证据不足)"})
    df.to_csv(os.path.join(out_dir, f"review_checklist{tag}.csv"),
              index=False, encoding="utf-8-sig")

    n_flag = int(flagged.sum())
    n_dir = int((flagged & (to_map >= 0)).sum())
    n_nodir = n_flag - n_dir
    work = {"n_flagged_pixels": n_flag,
            "n_directed_pixels": n_dir,
            "n_nodir_pixels_lowpriority": n_nodir,
            "n_directed_objects": int(n_obj),
            "n_checklist_items": int(len(df)),
            "tier_counts_pixels": {t: int((tier == t).sum()) for t in ("A", "B", "C")},
            "tier_counts_objects": {t: int((df["tier"] == t).sum()) if len(df) else 0
                                    for t in ("A", "B", "C")},
            "est_hours_pixelwise_directed": round(n_dir * 2.0 / 3600, 2),
            "est_hours_objectwise_directed": round(n_obj * t_item_s / 3600, 2),
            "est_hours_pixelwise_all": round(n_flag * 2.0 / 3600, 2),
            "t_item_s": t_item_s,
            "note": "Checklist contains DIRECTED objects only. No-class flagged pixels "
                    "are reported separately as a low-priority pool."}
    jdump(work, os.path.join(out_dir, f"review_workload{tag}.json"))
    log(f"\n[核查清单{tag}] 前10项:\n" + (df.head(10).to_string(index=False) if len(df) else "(空)"))
    log(f"[工作量{tag}] " + json.dumps(work, ensure_ascii=False, default=_jd))
    return df, work, lab


def export_annotation_rasters(out_dir, code, score, tier, from_map, to_map,
                              transform=None, crs=None, tag=""):
    tier_num = np.zeros(tier.shape, np.uint8)
    tier_num[tier == "A"] = 1; tier_num[tier == "B"] = 2; tier_num[tier == "C"] = 3
    prod_map = {"error_directed_code": (code.astype(np.int32), "int32"),
                "error_confidence": (score.astype(np.float32), "float32"),
                "review_tier": (tier_num, "uint8"),
                "from_class": (from_map.astype(np.int16), "int16"),
                "to_class": (to_map.astype(np.int16), "int16")}
    tr = transform if transform is not None else GEO["transform"]
    cr = crs if crs is not None else GEO["crs"]
    if not HAS_RASTERIO or tr is None:
        for k, (a, _) in prod_map.items():
            np.save(os.path.join(out_dir, k + tag + ".npy"), a)
        return
    for k, (a, dt) in prod_map.items():
        with rasterio.open(os.path.join(out_dir, k + tag + ".tif"), "w", driver="GTiff",
                           height=a.shape[0], width=a.shape[1], count=1, dtype=dt,
                           crs=cr, transform=tr,
                           nodata=0 if k != "error_confidence" else None) as ds:
            ds.write(a, 1)


def export_vector_checklist(df, lab, out_dir, transform=None, crs=None,
                            max_features=5000, tag=""):
    if not HAS_RASTERIO:
        return
    tr = transform if transform is not None else GEO["transform"]
    cr = crs if crs is not None else GEO["crs"]
    if tr is None or not len(df):
        return
    keep_ids = set(df["item_id"].tolist())
    mask_lab = np.where(np.isin(lab, list(keep_ids)), lab, 0).astype(np.int32)
    feats = []
    for geom, val in rio_shapes(mask_lab, mask=mask_lab > 0, transform=tr):
        if val in keep_ids:
            row = df[df.item_id == val]
            props = {"item_id": int(val)}
            if len(row):
                r = row.iloc[0]
                for k in ("tier", "from_class", "to_class", "error_type",
                          "mean_error_score", "review_action", "n_pixels"):
                    if k in r:
                        props[k] = r[k]
            feats.append({"type": "Feature", "geometry": geom, "properties": props})
        if len(feats) >= max_features:
            break
    gj = {"type": "FeatureCollection", "features": feats}
    if cr is not None:
        gj["crs"] = {"type": "name", "properties": {"name": str(cr)}}
    jdump(gj, os.path.join(out_dir, f"review_checklist{tag}.geojson"))
    log(f"[矢量{tag}] 导出 {len(feats)} 个核查图斑")


def make_review_chips(img, prod_presented, pred_cls, tier, df, out_dir,
                      n_chips=12, chip=96, tag=""):
    os.makedirs(os.path.join(out_dir, "review_chips"), exist_ok=True)
    P = np.clip(prod_presented, 0, C - 1)
    H, W = tier.shape
    for _, r in df.head(n_chips).iterrows():
        cy, cx = int(r["row"]), int(r["col"])
        r0 = max(0, min(cy - chip // 2, H - chip))
        c0 = max(0, min(cx - chip // 2, W - chip))
        r1, c1 = r0 + chip, c0 + chip
        im = img[r0:r1, c0:c1]
        lc = PALETTE_LC[P[r0:r1, c0:c1]]
        prop = PALETTE_LC[np.clip(pred_cls[r0:r1, c0:c1], 0, C - 1)]
        ev = tier_map_rgb(tier[r0:r1, c0:c1], np.ones((chip, chip), bool))
        fig, axes = plt.subplots(1, 4, figsize=(COL2, 42 * MM))
        for ax, data, tt in zip(axes, [im, lc, prop, ev],
                                ["Image", f"Product: {r.get('from_class','')}",
                                 f"Proposed: {r.get('to_class','')}",
                                 f"Tier {r.get('tier','')}"]):
            ax.imshow(np.clip(data, 0, 1)); ax.set_title(tt, fontsize=7); ax.set_axis_off()
        fig.suptitle(f"#{int(r['review_priority'])}  {r.get('error_type','')}  "
                     f"score={r.get('mean_error_score','')}", fontsize=8)
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, "review_chips",
                                   f"chip{tag}_{int(r['review_priority']):04d}"), tight=True)


# =============================================================================
# 10. [Q2] 双源候选集 + 阈值敏感性 + 模式对比
# =============================================================================
def deployment_candidates(sc_ens, pc_ens, prod, valid, thr, mode="disagree"):
    """
    [Q2] 双源候选:
      disagree  - 分类头不一致集 (推荐主口径; "应该是什么类"的直接证据)
      score     - 误差头高分集 (绝对阈值)
      union     - 并集
      intersect - 交集
    误差分数仅用于排序/分级, 不单独决定候选(disagree 模式下)。
    """
    P = np.clip(prod, 0, C - 1)
    disagree = (pc_ens != P) & valid
    scored = (sc_ens >= float(thr)) & valid
    return {"disagree": disagree, "score": scored,
            "union": (scored | disagree) & valid,
            "intersect": (scored & disagree) & valid}[mode]


def threshold_sensitivity(sc_ens, pc_ens, prod, valid, thr_cal, out_dir,
                          thrs=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)):
    """[Q2] 阈值敏感性: 各档候选量 / 有向量 / 图斑数 / 工时, 供核查方按人力选档。"""
    P = np.clip(prod, 0, C - 1)
    disagree = (pc_ens != P) & valid
    rows = []
    all_t = list(thrs) + [float(thr_cal)]
    for t in all_t:
        scored = (sc_ens >= t) & valid
        union = (scored | disagree) & valid
        inter = (scored & disagree) & valid
        n_dir_u = int((union & (pc_ens != P)).sum())
        _, _, n_obj_u = label_directed_objects(
            *annotate_from_candidates(prod, pc_ens, union)[:2], union)
        rows.append({"threshold": float(t),
                     "is_calibrated": bool(abs(t - thr_cal) < 1e-9),
                     "n_score": int(scored.sum()),
                     "n_disagree": int(disagree.sum()),
                     "n_union": int(union.sum()),
                     "n_intersect": int(inter.sum()),
                     "n_union_directed": n_dir_u,
                     "n_union_objects": int(n_obj_u),
                     "union_review_hours_2s": round(n_dir_u * 2.0 / 3600, 2)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "threshold_sensitivity.csv"),
              index=False, encoding="utf-8-sig")
    log("\n[阈值敏感性]\n" + df.round(4).to_string(index=False))
    return df


def candidate_mode_comparison(sc_ens, pc_ens, prod, valid, thr, out_dir):
    """[Q2] 四种候选模式对比: 候选量 / 有向量 / 图斑数 / 工时。"""
    P = np.clip(prod, 0, C - 1)
    rows = []
    for mode in CAND_MODES:
        cand = deployment_candidates(sc_ens, pc_ens, prod, valid, thr, mode)
        fm, tm, _, _ = annotate_from_candidates(prod, pc_ens, cand)
        _, _, n_obj = label_directed_objects(fm, tm, cand)
        n_dir = int((cand & (tm >= 0)).sum())
        rows.append({"mode": mode, "n_candidates": int(cand.sum()),
                     "candidate_rate_pct": 100 * int(cand.sum()) / max(int(valid.sum()), 1),
                     "n_directed": n_dir, "n_objects": int(n_obj),
                     "review_hours_pixel_2s": round(n_dir * 2.0 / 3600, 2),
                     "review_hours_object_20s": round(n_obj * 20.0 / 3600, 2)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "candidate_mode_comparison.csv"),
              index=False, encoding="utf-8-sig")
    log("\n[候选模式对比]\n" + df.round(4).to_string(index=False))
    return df


def plot_threshold_sensitivity(df, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 56 * MM))
    ax = axes[0]
    ax.plot(df.threshold, df.n_score, "o-", color="#E15759", ms=3.5, lw=1.3, label="score (error head)")
    ax.plot(df.threshold, df.n_union, "s-", color="#59A14F", ms=3.5, lw=1.3, label="union")
    ax.plot(df.threshold, df.n_intersect, "^-", color="#F28E2B", ms=3.5, lw=1.3, label="intersect")
    ax.axhline(df.n_disagree.iloc[0], color="#3A7CA5", ls="--", lw=1.2, label="disagree (constant)")
    cal = df[df.is_calibrated]
    if len(cal):
        ax.axvline(cal.threshold.iloc[0], color="#2E86C1", ls=":", lw=1.1)
    ax.set_yscale("log")
    ax.set_xlabel("Absolute score threshold"); ax.set_ylabel("Candidate pixels (log)")
    ax.legend(fontsize=6.0, frameon=False)
    ax.grid(alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(a)")
    ax = axes[1]
    ax.plot(df.threshold, df.union_review_hours_2s, "s-", color="#59A14F", ms=3.5, lw=1.3)
    ax.set_xlabel("Absolute score threshold"); ax.set_ylabel("Union review hours (2 s/px)")
    ax.grid(alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(b)")
    fig.tight_layout(); save_fig(fig, out_stem)


def plot_candidate_modes(df, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 56 * MM))
    x = np.arange(len(df))
    ax = axes[0]
    ax.bar(x, df.n_candidates, 0.6, color=BAR_COLORS[:len(df)], edgecolor="black", lw=.5)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(df["mode"], fontsize=7)
    ax.set_ylabel("Candidate pixels (log)")
    for i, v in enumerate(df.n_candidates):
        ax.text(i, v * 1.1, f"{int(v):,}", ha="center", fontsize=6.2)
    ax.grid(axis="y", alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(a)")
    ax = axes[1]
    ax.bar(x - 0.2, df.review_hours_pixel_2s, 0.4, label="pixel-wise (2s)", color="#59A14F", edgecolor="black", lw=.5)
    ax.bar(x + 0.2, df.review_hours_object_20s, 0.4, label="object-wise (20s)", color="#F28E2B", edgecolor="black", lw=.5)
    ax.set_xticks(x); ax.set_xticklabels(df["mode"], fontsize=7)
    ax.set_ylabel("Review hours"); ax.legend(fontsize=6.2, frameon=False)
    ax.grid(axis="y", alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(b)")
    fig.tight_layout(); save_fig(fig, out_stem)


# =============================================================================
# 11. 关键诊断图: 验证 vs 部署 分数分布 (口径一致)
# =============================================================================
def score_distribution_summary(s_val, s_dep, thr, out_dir):
    qs = [0.50, 0.90, 0.99, 0.999, 0.9999, 1.0]
    rows = []
    for name, s in (("validation (injected, model-ensemble)", s_val),
                    ("deployment (real product, model-ensemble)", s_dep)):
        row = {"Domain": name, "n_valid": int(s.size)}
        for q in qs:
            row[f"q{q*100:g}"] = float(np.quantile(s, q))
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            row[f"frac_ge_{t}"] = float((s >= t).mean())
        row["frac_ge_calibrated_thr"] = float((s >= thr).mean())
        row["n_ge_calibrated_thr"] = int((s >= thr).sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "score_distribution_summary.csv"),
              index=False, encoding="utf-8-sig")
    log("\n[分数分布摘要]\n" + df.round(6).to_string(index=False))
    return df


def plot_score_distribution(s_val, s_dep, thr, curve_df, calib, out_stem):
    """四联诊断图: 全域分布 / 头部放大 / 精度召回-阈值 / 标记量-阈值。"""
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 110 * MM))
    rng = np.random.default_rng(0)
    sv = s_val if s_val.size <= 400000 else rng.choice(s_val, 400000, replace=False)
    sd = s_dep if s_dep.size <= 400000 else rng.choice(s_dep, 400000, replace=False)

    ax = axes[0, 0]
    bins = np.linspace(0, 1, 120)
    ax.hist(sv, bins=bins, density=True, histtype="step", lw=1.4,
            color=C_VAL, label="Validation (injected, π=6%)")
    ax.hist(sd, bins=bins, density=True, histtype="step", lw=1.4,
            color=C_DEP, label="Deployment (real product)")
    ax.axvline(thr, color="#2E86C1", ls="--", lw=1.2, label=f"Calibrated thr = {thr:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("Error score"); ax.set_ylabel("Density (log)")
    ax.set_xlim(0, 1); ax.legend(fontsize=6.2, frameon=False)
    ax.grid(alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(a)")

    ax = axes[0, 1]
    hb = np.linspace(0, 1, 60)
    ax.hist(sv, bins=hb, density=True, alpha=.55, color=C_VAL, label="Validation", edgecolor="black", lw=.2)
    ax.hist(sd, bins=hb, density=True, alpha=.55, color=C_DEP, label="Deployment", edgecolor="black", lw=.2)
    ax.axvline(thr, color="#2E86C1", ls="--", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Error score"); ax.set_ylabel("Density (log)")
    ax.set_xlim(0, 1); ax.legend(fontsize=6.2, frameon=False)
    ax.set_title("Head of the ranking", fontsize=7)
    ax.grid(alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(b)")

    ax = axes[1, 0]
    if curve_df is not None and len(curve_df):
        ax.plot(curve_df.threshold, curve_df.precision_pct, color="#59A14F", lw=1.5, label="Precision (val)")
        ax.plot(curve_df.threshold, curve_df.recall_pct, color="#F28E2B", lw=1.5, label="Recall (val)")
        ax.axhline(calib.get("target_precision", 0.9) * 100, color="#555555", ls=":", lw=1.0,
                   label=f"target = {calib.get('target_precision',0.9)*100:.0f}%")
    ax.axvline(thr, color="#2E86C1", ls="--", lw=1.2, label=f"thr = {thr:.3f}")
    ax.set_xlabel("Absolute score threshold"); ax.set_ylabel("%")
    ax.set_xlim(0, 1); ax.set_ylim(0, 105)
    ax.legend(fontsize=6.0, frameon=False, loc="center right")
    ax.grid(alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(c)")

    ax = axes[1, 1]
    tg = np.linspace(0.001, 0.999, 200)
    nv = np.array([(s_val >= t).sum() for t in tg], float)
    nd = np.array([(s_dep >= t).sum() for t in tg], float)
    ax.plot(tg, np.maximum(nv, 1), color=C_VAL, lw=1.5, label="Validation")
    ax.plot(tg, np.maximum(nd, 1), color=C_DEP, lw=1.5, label="Deployment")
    ax.axvline(thr, color="#2E86C1", ls="--", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Absolute score threshold"); ax.set_ylabel("Flagged pixels (log)")
    ax.legend(fontsize=6.2, frameon=False)
    ax.grid(alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(d)")

    fig.text(.005, -.02,
             "Both domains use a model-ensemble on a SINGLE injection realisation (validation) "
             "or the real product (deployment), so the comparison is coherent. The real-product "
             "score distribution is compressed near zero; deployment therefore uses a "
             "class-head-disagreement candidate set ranked by the error score, not a fixed budget.",
             fontsize=6.8, ha="left", va="top")
    fig.tight_layout(); save_fig(fig, out_stem, tight=True)


# =============================================================================
# 12. 部署模式 (双源候选)
# =============================================================================
def audit_real_product(models, img, prod, valid, out_dir, thr,
                       candidate_mode="disagree", score_hi_quantile=0.25,
                       max_items=100000, n_chips=12, spectral_mode="presented"):
    set_pub_style()
    ann_dir = os.path.join(out_dir, "deploy")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(ann_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)
    log("\n" + "=" * 72)
    log(f"[部署模式] 原始产品推理; 候选模式={candidate_mode}; 校准阈值={thr:.6f}")
    log("=" * 72)

    sc_ens, pc_ens, agree, scores, preds = ensemble_predict(models, img, prod)
    flag_stack = [decide_absolute(s, valid, thr) for s in scores]
    freq = np.mean(flag_stack, axis=0).astype(np.float32)

    # [Q2] 双源候选
    candidate = deployment_candidates(sc_ens, pc_ens, prod, valid, thr, candidate_mode)
    from_map, to_map, flagged, code = annotate_from_candidates(prod, pc_ens, candidate)

    co_df, T = commission_omission(prod, pc_ens, flagged, valid, ann_dir, tag="_deploy")
    lam, L1 = L1_spectral_likelihood(img, prod, from_map, to_map, flagged, valid,
                                     ann_dir, spectral_mode, tag="_deploy")
    L3 = L3_dual_readout(flagged, pc_ens, prod, valid, ann_dir, tag="_deploy")
    support, L4 = L4_context(prod, to_map, flagged, valid, ann_dir, tag="_deploy")
    tier, thr_hi = confidence_tiers_relative(sc_ens, support, freq, agree, to_map,
                                             candidate, score_hi_quantile)
    log(f"[Q2] 候选集内高分分位阈值 (top {score_hi_quantile*100:.0f}%) = {thr_hi:.4f}")

    df, work, lab = export_review_checklist(
        img, prod, pc_ens, sc_ens, from_map, to_map, flagged, tier, lam, support,
        freq, agree, valid, ann_dir, max_items=max_items, tag="_deploy")
    export_annotation_rasters(ann_dir, code, sc_ens, tier, from_map, to_map, tag="_deploy")
    export_vector_checklist(df, lab, ann_dir, tag="_deploy")
    make_review_chips(img, prod, pc_ens, tier, df, ann_dir, n_chips=n_chips, tag="_deploy")

    # [Q2] 敏感性 + 模式对比
    sens = threshold_sensitivity(sc_ens, pc_ens, prod, valid, thr, ann_dir)
    comp = candidate_mode_comparison(sc_ens, pc_ens, prod, valid, thr, ann_dir)
    plot_threshold_sensitivity(sens, os.path.join(fig_dir, "D10_threshold_sensitivity"))
    plot_candidate_modes(comp, os.path.join(fig_dir, "D11_candidate_modes"))

    rows = [{"Layer": "L1 光谱似然比", "Role": "仅诊断(不参与分级)",
             "Metric": "Λ>0 比例", "Value": L1.get("frac_spectral_support")},
            {"Layer": "L3 双流互证", "Role": "半独立",
             "Metric": "标记中有类别支持比例", "Value": L3.get("frac_flagged_with_class_support")},
            {"Layer": "L4 空间上下文", "Role": "独立",
             "Metric": "邻域支持>0.3比例", "Value": L4.get("frac_support_gt0.3")},
            {"Layer": "L5 稳定性", "Role": "独立",
             "Metric": "A级像元数", "Value": work["tier_counts_pixels"]["A"]}]
    pd.DataFrame(rows).to_csv(os.path.join(ann_dir, "self_evaluation_deploy.csv"),
                              index=False, encoding="utf-8-sig")

    plot_deploy_overview(img, prod, sc_ens, flagged, tier, valid,
                         os.path.join(fig_dir, "D01_deploy_overview"))
    win = pick_zoom_window_deploy(flagged, to_map, valid, size=160)
    plot_deploy_zoom(img, prod, flagged, tier, valid, win,
                     os.path.join(fig_dir, "D02_deploy_zoom"))
    plot_directed_error_map(prod, from_map, to_map, flagged, valid,
                            os.path.join(fig_dir, "D03_directed_errors"))
    plot_commission_omission(co_df, os.path.join(fig_dir, "D04_commission_omission"))
    plot_net_area_bias(co_df, os.path.join(fig_dir, "D05_net_area_bias"))
    plot_transition_matrix(T, os.path.join(fig_dir, "D06_transition_matrix"))
    plot_lambda_hist(lam, flagged, to_map, os.path.join(fig_dir, "D07_lambda_hist"))
    plot_tier_distribution(tier, flagged, os.path.join(fig_dir, "D08_tier_distribution"))
    plot_workload(work, os.path.join(fig_dir, "D09_workload"))

    jdump({"mode": "deployment", "candidate_mode": candidate_mode,
           "calibrated_threshold": float(thr),
           "relative_high_score_threshold": float(thr_hi),
           "n_candidates": int(candidate.sum()),
           "candidate_rate_pct": 100 * int(candidate.sum()) / max(int(valid.sum()), 1),
           "n_directed": int((flagged & (to_map >= 0)).sum()),
           "tier_counts_pixels": work["tier_counts_pixels"],
           "tier_counts_objects": work["tier_counts_objects"],
           "L1": L1, "L3": L3, "L4": L4,
           "note": "Real-product audit. Candidates = class-head disagreement (default), "
                   "ranked by ensemble error score. No ground truth; precision unknown "
                   "until human confirmation. Rates are strong-contradiction rates."},
          os.path.join(ann_dir, "deploy_summary.json"))
    log(f"\n[部署完成] 真实核查清单 -> {ann_dir}")
    return {"co": co_df, "T": T, "work": work, "L1": L1, "L3": L3, "L4": L4,
            "score": sc_ens, "flagged": flagged}


# =============================================================================
# 13. 验证模式 (合成注入) + [Q1] 逐实现校准
# =============================================================================
def run_validation(models, img, prod, valid, out_dir, train_seeds, eval_seeds,
                   detect_ratio=0.10, injection=0.06, target_precision=0.90,
                   cache_scores=True):
    set_pub_style()
    tab_dir = os.path.join(out_dir, "tables")
    fig_dir = os.path.join(out_dir, "figures")
    ann_dir = os.path.join(out_dir, "validation")
    for d in (tab_dir, fig_dir, ann_dir):
        os.makedirs(d, exist_ok=True)
    log("\n" + "=" * 72 + "\n[验证模式] 合成注入 -> 方法有效性佐证\n" + "=" * 72)

    # [Q1] 按注入实现分组: 同一 es 下只累积模型
    score_by_eval = {es: [] for es in eval_seeds}
    pred_by_eval = {es: [] for es in eval_seeds}
    gt_by_eval = {}
    pa_by_eval = {}
    all_rows = []
    for ti, model in enumerate(models):
        ts = train_seeds[ti]
        for es in eval_seeds:
            rng = np.random.default_rng(es)
            p_a, y_a = inject_pixel_mix_safe(prod.copy(), valid, rng, injection)
            sc, pc, cf = infer_full(model, img, p_a)
            pred = post_process_direct(sc, valid, detect_ratio)
            m = evaluate_change(pred, y_a, valid)
            m["train_seed"] = ts; m["eval_seed"] = es
            m["auc_raw"], m["ap_raw"] = discrimination(sc, y_a, valid)
            all_rows.append(m)
            score_by_eval[es].append(sc)
            pred_by_eval[es].append(pc)
            gt_by_eval[es] = y_a
            pa_by_eval[es] = p_a
            if cache_scores:
                np.savez_compressed(os.path.join(tab_dir, f"score_{ts}_{es}.npz"),
                                    score=sc.astype(np.float32), gt=y_a,
                                    valid=valid, pred_cls=pc.astype(np.int64))
    pd.DataFrame(all_rows).to_csv(os.path.join(tab_dir, "runs_raw.csv"),
                                  index=False, encoding="utf-8-sig")
    ours = aggregate_seeds(all_rows)
    log(f"\n[检测性能] Recall={ours['Recall']*100:.2f}%  "
        f"Precision={ours['Precision']*100:.2f}%  "
        f"AUC={np.mean([r['auc_raw'] for r in all_rows]):.4f}")

    # 首个实现的逐 run 结果 (用于 L2/L3/标注/图)
    es0 = eval_seeds[0]
    sc0 = score_by_eval[es0][0]
    pc0 = pred_by_eval[es0][0]
    p_a0 = pa_by_eval[es0]
    y_a0 = gt_by_eval[es0]
    pred0 = post_process_direct(sc0, valid, detect_ratio)

    flag_stack = [post_process_direct(s, valid, detect_ratio)
                  for es in eval_seeds for s in score_by_eval[es]]
    pred_stack = [p for es in eval_seeds for p in pred_by_eval[es]]
    freq, agree = stability_from_runs(flag_stack, pred_stack)

    from_map, to_map, flagged, code = annotate_directed(p_a0, pc0, sc0, valid, detect_ratio)
    co_df, T = commission_omission(p_a0, pc0, flagged, valid, ann_dir, tag="_val")
    lam, L1 = L1_spectral_likelihood(img, p_a0, from_map, to_map, flagged, valid,
                                     ann_dir, "presented", tag="_val")
    L2 = L2_directed_accuracy(from_map, to_map, flagged, prod, p_a0, y_a0, valid, ann_dir)
    L3 = L3_dual_readout(flagged, pc0, p_a0, valid, ann_dir, tag="_val")
    support, L4 = L4_context(p_a0, to_map, flagged, valid, ann_dir, tag="_val")
    tier = confidence_tiers(sc0, support, freq, agree, to_map, flagged)

    dir_mask = flagged & (to_map >= 0)
    true_to = np.clip(prod, 0, C - 1)
    correct = (to_map == true_to)[dir_mask]
    lam_v = lam[dir_mask]
    fin = np.isfinite(lam_v)
    lam_auc = None
    if fin.sum() > 10 and correct[fin].any() and (~correct[fin]).any():
        lam_auc = float(roc_auc_score(correct[fin].astype(int), lam_v[fin]))
    jdump({"lambda_discrimination_auc": lam_auc,
           "role": "L1 excluded from tiering: AUC~0.5, no discriminative power"},
          os.path.join(ann_dir, "L1_lambda_auc.json"))
    log(f"\n[L1 判别力 AUC] {lam_auc}")

    pw = precision_workload_tradeoff(sc0, pc0, p_a0, y_a0, valid, detect_ratio, ann_dir)

    # ---- [Q1] 逐实现模型集合 -> 逐实现校准 -> 中位数 ----
    thrs, calibs = [], []
    val_ens_by_eval = {}
    for es in eval_seeds:
        sc_ens_es = np.mean(np.stack(score_by_eval[es], 0), axis=0).astype(np.float32)
        val_ens_by_eval[es] = sc_ens_es
        t_e, c_e = calibrate_threshold(sc_ens_es, gt_by_eval[es], valid, target_precision)
        c_e["eval_seed"] = int(es)
        thrs.append(t_e); calibs.append(c_e)
        np.savez_compressed(os.path.join(ann_dir, f"val_ensemble_es{es}.npz"),
                            score=sc_ens_es, gt=gt_by_eval[es], valid=valid)
    thr = float(np.median(thrs))
    calib = {"target_precision": float(target_precision),
             "per_realisation_thresholds": [float(t) for t in thrs],
             "per_realisation": calibs,
             "threshold": thr,
             "aggregation": "median over per-realisation model-ensemble calibrations",
             "note": "Ensemble across MODELS only within each injection realisation; "
                     "never across realisations (V13 bug fixed)."}
    jdump(calib, os.path.join(out_dir, "calibration.json"))
    log("\n[Q1 逐实现校准] " + json.dumps(
        {"per_realisation_thresholds": thrs, "median_threshold": thr},
        ensure_ascii=False, default=_jd))
    for c_e in calibs:
        log("  es=%d thr=%.4f valP=%.2f%% valR=%.2f%% n=%d" % (
            c_e["eval_seed"], c_e["threshold"], c_e["val_precision_pct"],
            c_e["val_recall_pct"], c_e["val_n_flagged"]))

    # 诊断图用首个实现的模型集合 (口径一致)
    sc_val_plot = val_ens_by_eval[es0]
    curve = threshold_curve(sc_val_plot, y_a0, valid)
    curve.to_csv(os.path.join(ann_dir, "threshold_curve.csv"), index=False, encoding="utf-8-sig")
    np.savez_compressed(os.path.join(ann_dir, "val_ensemble.npz"),
                        score=sc_val_plot, gt=y_a0, valid=valid)

    # ---- 验证图组 ----
    plot_val_overlay(prod, pred0, y_a0, valid, os.path.join(fig_dir, "V01_detection_overlay"))
    win = pick_zoom_window(pred0, y_a0, valid, size=160)
    plot_val_zoom(prod, pred0, y_a0, valid, win, os.path.join(fig_dir, "V02_detection_zoom"))
    scores_curves = [s for es in eval_seeds for s in score_by_eval[es]]
    gts_curves = [gt_by_eval[es] for es in eval_seeds for _ in score_by_eval[es]]
    plot_roc_pr({"DS-ResUNet (raw score)": scores_curves}, gts_curves,
                [valid] * len(scores_curves), os.path.join(fig_dir, "V03_roc_pr"))
    df_tau = tau_sweep(tab_dir, tab_dir, prevalence=injection)
    if df_tau is not None:
        plot_tau_sweep(df_tau, os.path.join(fig_dir, "V04_tau_sweep"), prevalence=injection)
    plot_precision_workload(pw, os.path.join(fig_dir, "V05_precision_workload"))
    plot_directed_perclass(L2, os.path.join(fig_dir, "V06_directed_perclass"))
    plot_lambda_discrimination(lam, flagged, to_map, true_to, lam_auc,
                               os.path.join(fig_dir, "V07_lambda_discrimination"))
    plot_stability(freq, agree, flagged, os.path.join(fig_dir, "V08_stability"))
    plot_commission_omission(co_df, os.path.join(fig_dir, "V09_commission_omission_val"))
    cg = cross_generator_eval(out_dir, img, prod, valid, train_seeds, eval_seeds,
                              detect_ratio, injection)
    if cg is not None:
        plot_cross_generator(cg, os.path.join(fig_dir, "V10_cross_generator"))

    jdump({"mode": "validation", "metrics": ours, "L1": L1, "L2": L2, "L3": L3,
           "L4": L4, "lambda_auc": lam_auc, "precision_workload": pw.to_dict("records"),
           "calibration": calib},
          os.path.join(ann_dir, "validation_summary.json"))
    log(f"\n[验证完成] -> {ann_dir}")
    return {"metrics": ours, "thr": thr, "calib": calib, "curve": curve,
            "score_ens": sc_val_plot, "gt": y_a0}


def precision_workload_tradeoff(score, pred_cls, prod_presented, y_true, valid,
                                tau, out_dir):
    P = np.clip(prod_presented, 0, C - 1)
    flagged = post_process_direct(score, valid, tau)
    directed = flagged & (pred_cls != P)
    rows = []
    for name, mask in (("All flagged", flagged), ("Directed (A+B)", directed),
                       ("No class (C)", flagged & ~directed)):
        m = evaluate_change(mask, y_true, valid)
        rows.append({"Subset": name, "n_pixels": int(mask.sum()),
                     "Precision (%)": 100 * m["Precision"],
                     "Recall (%)": 100 * m["Recall"],
                     "Review_hours_2s": int(mask.sum()) * 2.0 / 3600})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "precision_workload.csv"), index=False, encoding="utf-8-sig")
    log("\n[精度-工作量权衡]\n" + df.round(2).to_string(index=False))
    return df


def tau_sweep(tab_dir, out_dir, taus=(0.02, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20),
              prevalence=0.06):
    rows = []
    for f in sorted(glob.glob(os.path.join(tab_dir, "score_4*_*.npz"))):
        b = os.path.basename(f)[6:-4].split("_")
        d = np.load(f)
        s = d["score"].astype(np.float64); gt = d["gt"].astype(bool); ok = d["valid"].astype(bool)
        sm = ndi.uniform_filter(s, 5)
        for tau in taus:
            for dec, sc in ((DEC_R, s), (DEC_S, sm)):
                m = evaluate_change(_decide(sc, ok, tau), gt, ok)
                rows.append({"train_seed": int(b[0]), "eval_seed": int(b[1]),
                             "tau": tau, "Decision statistic": dec,
                             "Recall (%)": 100 * m["Recall"],
                             "Precision (%)": 100 * m["Precision"],
                             "F1 (%)": 100 * m["F1"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "tau_sweep.csv"), index=False, encoding="utf-8-sig")
    return df


def cross_generator_eval(out_dir, img, prod, valid, train_seeds=(42, 43, 44),
                         eval_seeds=(100, 101, 102), tau=0.10, injection=0.06,
                         modes=("random_class", "scattered_only", "edge_targeted")):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = [os.path.join(out_dir, f"model_full_{ts}.pt") for ts in train_seeds]
    if not all(os.path.exists(p) for p in ck):
        return None
    rows = []
    for ts, path in zip(train_seeds, ck):
        model = DualResUNetV6(in_ch=img.shape[2]).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        for mode in modes:
            for es in eval_seeds:
                rng = np.random.default_rng(es + 9000)
                p_a, y_a = inject_alt(prod.copy(), valid, rng, injection, mode)
                sc = infer_score(model, img, p_a)
                m = evaluate_change(_decide(sc, valid, tau), y_a, valid)
                auc, ap = discrimination(sc, y_a, valid)
                rows.append({"Generator": mode, "train_seed": ts, "eval_seed": es,
                             "Recall (%)": 100 * m["Recall"], "AUC": auc, "AP": ap})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "tables", "cross_generator.csv"),
              index=False, encoding="utf-8-sig")
    log("\n[跨生成器]\n" + df.groupby("Generator")[["Recall (%)", "AUC", "AP"]].mean().round(4).to_string())
    return df


# =============================================================================
# 14. 图: 制图类 (比例尺 + 指北针 + 图例; 无 Easting/Northing)
# =============================================================================
def pick_zoom_window(pred, gt, valid, size=160):
    H, W = gt.shape
    em = ((pred > 0) ^ (gt > 0)) & valid
    dens = ndi.uniform_filter(em.astype(float), size=size); dens[~valid] = 0
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape) if dens.max() > 0 else (H // 2, W // 2)
    half = size // 2
    r0 = int(np.clip(cy - half, 0, max(H - size, 0)))
    c0 = int(np.clip(cx - half, 0, max(W - size, 0)))
    return (r0, min(r0 + size, H), c0, min(c0 + size, W))


def pick_zoom_window_deploy(flagged, to_map, valid, size=160):
    H, W = flagged.shape
    dm = (flagged & (to_map >= 0)).astype(float)
    dens = ndi.uniform_filter(dm, size=size); dens[~valid] = 0
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape) if dens.max() > 0 else (H // 2, W // 2)
    half = size // 2
    r0 = int(np.clip(cy - half, 0, max(H - size, 0)))
    c0 = int(np.clip(cx - half, 0, max(W - size, 0)))
    return (r0, min(r0 + size, H), c0, min(c0 + size, W))


def plot_val_overlay(landcover, pred, gt, valid, out_stem):
    lc = _lc_to_rgb(landcover, valid)
    det = error_map_rgba(pred, gt, valid, solid=True)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 86 * MM))
    for i, (im, handles, ncol) in enumerate(((lc, lc_legend_handles(), 5),
                                             (det, err_legend_handles(), 3))):
        axes[i].imshow(im, interpolation="nearest")
        decorate(axes[i], lc.shape)
        axes[i].legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.02),
                       ncol=ncol, frameon=False, fontsize=6.5, handlelength=1.1,
                       columnspacing=.9, handletextpad=.4)
        panel_label(axes[i], "(a)" if i == 0 else "(b)")
    fig.tight_layout(); save_fig(fig, out_stem, tight=True)


def plot_val_zoom(landcover, pred, gt, valid, window, out_stem):
    r0, r1, c0, c1 = window
    full = _lc_to_rgb(landcover, valid)
    zoom = error_map_rgba(pred[r0:r1, c0:c1], gt[r0:r1, c0:c1],
                          valid[r0:r1, c0:c1], solid=True)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 80 * MM))
    axes[0].imshow(full, interpolation="nearest")
    axes[0].imshow(error_map_rgba(pred, gt, valid), interpolation="nearest")
    axes[0].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                                edgecolor="#333333", lw=1.2, ls="--", zorder=8))
    axes[1].imshow(zoom, interpolation="nearest")
    decorate(axes[0], full.shape); decorate(axes[1], zoom.shape)
    axes[1].legend(handles=err_legend_handles(), loc="upper center",
                   bbox_to_anchor=(.5, -.02), ncol=3, frameon=False, fontsize=6.5)
    panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)")
    fig.tight_layout(); save_fig(fig, out_stem, tight=True)


def plot_deploy_overview(img, prod, score, flagged, tier, valid, out_stem):
    lc = _lc_to_rgb(prod, valid)
    ev = tier_map_rgb(tier, valid)
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 72 * MM))
    axes[0].imshow(np.clip(img, 0, 1)); axes[0].set_title("Image", fontsize=8)
    axes[1].imshow(lc); axes[1].set_title("Product (audited)", fontsize=8)
    axes[2].imshow(ev); axes[2].set_title("Review tiers", fontsize=8)
    for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
        decorate(ax, lc.shape); panel_label(ax, lab)
    axes[1].legend(handles=lc_legend_handles(), loc="upper center",
                   bbox_to_anchor=(.5, -.02), ncol=5, frameon=False, fontsize=5.8,
                   handlelength=1.0, columnspacing=.7, handletextpad=.3)
    axes[2].legend(handles=tier_legend_handles(), loc="upper center",
                   bbox_to_anchor=(.5, -.02), ncol=3, frameon=False, fontsize=6.2)
    fig.tight_layout(); save_fig(fig, out_stem, tight=True)


def plot_deploy_zoom(img, prod, flagged, tier, valid, window, out_stem):
    r0, r1, c0, c1 = window
    lc = _lc_to_rgb(prod, valid)
    ev = tier_map_rgb(tier, valid)
    zsh = (r1 - r0, c1 - c0)
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 66 * MM))
    axes[0].imshow(lc, interpolation="nearest")
    axes[0].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                                edgecolor="#333333", lw=1.2, ls="--", zorder=8))
    axes[0].set_title("Product + zoom extent", fontsize=7.5)
    axes[1].imshow(np.clip(img[r0:r1, c0:c1], 0, 1), interpolation="nearest")
    axes[1].set_title("Image (zoom)", fontsize=7.5)
    axes[2].imshow(ev[r0:r1, c0:c1], interpolation="nearest")
    axes[2].set_title("Review tiers (zoom)", fontsize=7.5)
    decorate(axes[0], lc.shape); decorate(axes[1], zsh); decorate(axes[2], zsh)
    for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
        panel_label(ax, lab)
    axes[2].legend(handles=tier_legend_handles(), loc="upper center",
                   bbox_to_anchor=(.5, -.02), ncol=3, frameon=False, fontsize=6.0)
    fig.tight_layout(); save_fig(fig, out_stem, tight=True)


def plot_directed_error_map(prod, from_map, to_map, flagged, valid, out_stem):
    lc = _lc_to_rgb(prod, valid)
    overlay = np.zeros((*flagged.shape, 4), np.float32)
    pairs = [(a, b) for a in range(C) for b in TRANS.get(a, [])]
    extra = [(a, b) for a in range(C) for b in range(C)
             if a != b and (a, b) not in pairs]
    cmap = plt.get_cmap("tab20")
    handles = []
    k = 0
    for (a, b) in pairs + extra:
        m = flagged & (from_map == a) & (to_map == b)
        if m.sum() == 0:
            continue
        col = cmap(k % 20)[:3]; k += 1
        overlay[m] = (*col, 0.95)
        handles.append(Patch(facecolor=col, edgecolor="#333333", lw=.5,
                             label=f"{CLASS_NAME_EN[a]}→{CLASS_NAME_EN[b]}"))
    nodir = flagged & (to_map < 0)
    if nodir.sum() > 0:
        overlay[nodir] = (0.5, 0.5, 0.5, 0.9)
        handles.append(Patch(facecolor=(0.5, 0.5, 0.5), edgecolor="#333333",
                             lw=.5, label="No class suggestion"))
    fig, ax = plt.subplots(figsize=(COL1 * 1.5, 104 * MM))
    ax.imshow(lc, interpolation="nearest")
    ax.imshow(overlay, interpolation="nearest")
    decorate(ax, lc.shape)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.02),
              ncol=3, frameon=False, fontsize=6.2, handlelength=1.1)
    ax.set_title("Directed errors (commission → omission)", fontsize=8)
    fig.tight_layout(); save_fig(fig, out_stem, tight=True)


# =============================================================================
# 15. 图: 佐证类
# =============================================================================
def plot_roc_pr(score_runs, gt_runs, valid_runs, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 60 * MM))
    grid = np.linspace(0, 1, 201)
    for i, (name, scores) in enumerate(score_runs.items()):
        color = C_FULL if i == 0 else "#1B998B"
        tprs, precs, aucs, aps = [], [], [], []
        for s, gt, vm in zip(scores, gt_runs, valid_runs):
            sel = vm.ravel()
            y = (gt.ravel()[sel] > 0).astype(np.uint8); sc = s.ravel()[sel]
            if y.min() == y.max():
                continue
            fpr, tpr, _ = roc_curve(y, sc)
            pr, rc, _ = precision_recall_curve(y, sc)
            tprs.append(np.interp(grid, fpr, tpr))
            o = np.argsort(rc); precs.append(np.interp(grid, rc[o], pr[o]))
            aucs.append(roc_auc_score(y, sc)); aps.append(average_precision_score(y, sc))
        if not tprs:
            continue
        ta, pa = np.vstack(tprs), np.vstack(precs)
        for ax, curve, lab, extra in ((axes[0], ta, "AUC", aucs), (axes[1], pa, "AP", aps)):
            ax.plot(grid, curve.mean(0), color=color, lw=1.5,
                    label=f"{name} ({lab}={np.mean(extra):.4f}±{np.std(extra):.4f})")
            if len(curve) > 1:
                ax.fill_between(grid, curve.mean(0) - curve.std(0),
                                curve.mean(0) + curve.std(0), color=color, alpha=.2, lw=0)
    axes[0].plot([0, 1], [0, 1], color="#555555", ls="--", lw=1.0, label="Random")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    for ax, lab in zip(axes, ["(a)", "(b)"]):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(alpha=.3, ls="--", lw=.5); despine(ax)
        ax.legend(loc="lower right" if lab == "(a)" else "lower left",
                  fontsize=6.5, frameon=True, framealpha=.85, edgecolor="none")
        panel_label(ax, lab)
    fig.tight_layout(); save_fig(fig, out_stem)


def plot_tau_sweep(df, out_stem, prevalence=0.06):
    g = df.groupby(["tau", "Decision statistic"]).agg(
        R=("Recall (%)", "mean"), Rs=("Recall (%)", "std"),
        P=("Precision (%)", "mean"), F=("F1 (%)", "mean")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 56 * MM))
    for dec, col, mk in ((DEC_R, C_FULL, "s"), (DEC_S, "#F28E2B", "o")):
        s = g[g["Decision statistic"] == dec].sort_values("tau")
        lab = "Raw-score quantile" if dec == DEC_R else "5×5 smoothed + quantile"
        for ax, y, e in ((axes[0], s.R, s.Rs), (axes[1], s.P, None), (axes[2], s.F, None)):
            ax.errorbar(s.tau * 100, y, yerr=(e.fillna(0) if e is not None else None),
                        color=col, marker=mk, ms=3.5, capsize=2, lw=1.3, label=lab)
    tt = np.linspace(2, 20, 200)
    axes[0].plot(tt, np.minimum(1, prevalence / (tt / 100)) * 100, color="#555555",
                 ls="--", lw=1.0, label=r"$R_{max}$")
    axes[1].plot(tt, np.minimum(1, prevalence / (tt / 100)) * 100, color="#555555",
                 ls="--", lw=1.0, label=r"$P_{max}$")
    axes[2].plot(tt, 2 * np.minimum(tt / 100, prevalence) / (tt / 100 + prevalence) * 100,
                 color="#555555", ls="--", lw=1.0, label=r"$F_{1,max}$")
    for ax in axes:
        ax.axvline(prevalence * 100, color="#2E86C1", ls=":", lw=1.1)
        ax.set_xlabel(r"Review budget $\tau$ (%)")
        ax.grid(alpha=.35, ls="--", lw=.5); despine(ax)
        ax.legend(fontsize=5.8, frameon=False, loc="lower right")
    axes[0].set_ylabel("Recall (%)"); axes[1].set_ylabel("Precision (%)")
    axes[2].set_ylabel("F1 (%)"); axes[2].set_title(r"dotted: $\tau=\pi$", fontsize=7)
    for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
        panel_label(ax, lab)
    fig.tight_layout(); save_fig(fig, out_stem)


def plot_precision_workload(df, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 56 * MM))
    x = np.arange(len(df)); cols = ["#76B7B2", "#59A14F", "#E15759"]
    ax = axes[0]
    ax.bar(x - 0.2, df["Precision (%)"], 0.4, label="Precision", color="#3A7CA5",
           edgecolor="black", lw=.5)
    ax.bar(x + 0.2, df["Recall (%)"], 0.4, label="Recall", color="#F28E2B",
           edgecolor="black", lw=.5)
    ax.set_xticks(x); ax.set_xticklabels(df["Subset"], fontsize=6.5, rotation=12)
    ax.set_ylabel("%"); ax.set_ylim(0, 108)
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(axis="y", alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(a)")
    ax = axes[1]
    ax.bar(x, df["Review_hours_2s"], 0.5, color=cols, edgecolor="black", lw=.5)
    for i, v in enumerate(df["Review_hours_2s"]):
        ax.text(i, v + 1, f"{v:.0f} h", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(df["Subset"], fontsize=6.5, rotation=12)
    ax.set_ylabel("Review hours (2 s/pixel)")
    ax.grid(axis="y", alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(b)")
    fig.tight_layout(); save_fig(fig, out_stem)


def plot_directed_perclass(L2, out_stem):
    pc = pd.DataFrame(L2["per_class_directed"])
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 56 * MM))
    x = np.arange(len(pc))
    ax = axes[0]
    ax.bar(x, pc["coverage_pct"].fillna(0), 0.6, color="#3A7CA5", edgecolor="black", lw=.5)
    ax.set_xticks(x); ax.set_xticklabels(pc["Class"], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("Coverage (%)"); ax.set_ylim(0, 108)
    ax.set_title("Fraction of TP given a class suggestion", fontsize=7)
    ax.grid(axis="y", alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(a)")
    ax = axes[1]
    ax.bar(x, pc["conditional_acc_pct"].fillna(0), 0.6, color="#59A14F",
           edgecolor="black", lw=.5)
    ax.set_xticks(x); ax.set_xticklabels(pc["Class"], rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("Conditional accuracy (%)"); ax.set_ylim(0, 108)
    ax.set_title("Accuracy when a suggestion is made", fontsize=7)
    ax.grid(axis="y", alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(b)")
    fig.tight_layout(); save_fig(fig, out_stem)


def plot_lambda_hist(lam, flagged, to_map, out_stem):
    vals = lam[flagged & (to_map >= 0)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(COL1 * 1.3, 56 * MM))
    ax.hist(vals, bins=60, color="#3A7CA5", edgecolor="black", lw=.3, alpha=.85)
    ax.axvline(0, color="#E15759", ls="--", lw=1.2, label="Λ=0 (neutral)")
    ax.axvline(np.median(vals), color="#59A14F", ls=":", lw=1.2,
               label=f"median={np.median(vals):.3f}")
    ax.set_xlabel("Spectral log-likelihood ratio Λ (diagnostic only)")
    ax.set_ylabel("Count")
    ax.set_title(f"Λ>0 for {100*(vals>0).mean():.1f}% — NOT used for tiering", fontsize=7.5)
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(axis="y", alpha=.35, ls="--", lw=.5); despine(ax)
    fig.tight_layout(); save_fig(fig, out_stem)


def plot_lambda_discrimination(lam, flagged, to_map, true_to, lam_auc, out_stem):
    dir_mask = flagged & (to_map >= 0)
    correct = (to_map == true_to)[dir_mask]
    lv = lam[dir_mask]; fin = np.isfinite(lv)
    correct, lv = correct[fin], lv[fin]
    if lv.size == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 56 * MM))
    ax = axes[0]
    if correct.any():
        ax.hist(lv[correct], bins=50, density=True, alpha=.6, color="#59A14F",
                label="Correct suggestion", edgecolor="black", lw=.2)
    if (~correct).any():
        ax.hist(lv[~correct], bins=50, density=True, alpha=.6, color="#E15759",
                label="Wrong suggestion", edgecolor="black", lw=.2)
    ax.axvline(0, color="#333333", ls="--", lw=1.0)
    ax.set_xlabel("Λ"); ax.set_ylabel("Density")
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(a)")
    ax = axes[1]
    if lam_auc is not None:
        fpr, tpr, _ = roc_curve(correct.astype(int), lv)
        ax.plot(fpr, tpr, color=C_FULL, lw=1.5, label=f"Λ discriminates (AUC={lam_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#555555", ls="--", lw=1.0, label="Chance")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax.grid(alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(b)")
    fig.text(.005, -.03, "AUC ≈ 0.5 means Λ carries no information about correctness; "
                         "L1 is reported as a diagnostic and excluded from tiering.",
             fontsize=6.8, ha="left", va="top")
    fig.tight_layout(); save_fig(fig, out_stem, tight=True)


def plot_stability(freq, agree, flagged, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 56 * MM))
    for ax, data, title, cmap in ((axes[0], freq, "Flag frequency across runs", "viridis"),
                                  (axes[1], agree, "Class consensus across runs", "magma")):
        d = np.where(flagged, data, np.nan)
        im = ax.imshow(d, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=7.5); ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04).ax.tick_params(labelsize=6)
    for ax, lab in zip(axes, ["(a)", "(b)"]):
        panel_label(ax, lab, x=0, y=1.0)
    fig.tight_layout(); save_fig(fig, out_stem)


def plot_commission_omission(df, out_stem):
    fig, ax = plt.subplots(figsize=(COL2, 58 * MM))
    x = np.arange(len(df)); w = 0.38
    ax.bar(x - w / 2, df["commission_rate_pct"], w, label="Commission (错分)",
           color="#E15759", edgecolor="black", lw=.5)
    ax.bar(x + w / 2, df["omission_rate_pct"], w, label="Omission (漏分a)",
           color="#3A7CA5", edgecolor="black", lw=.5)
    ax.set_xticks(x); ax.set_xticklabels(df["Class"], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Rate (%)"); ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=.35, ls="--", lw=.5); despine(ax)
    fig.tight_layout(); save_fig(fig, out_stem)


def plot_net_area_bias(df, out_stem):
    fig, ax = plt.subplots(figsize=(COL2, 56 * MM))
    x = np.arange(len(df)); vals = df["net_area_bias_km2"].values
    cols = ["#3A7CA5" if v >= 0 else "#E15759" for v in vals]
    ax.bar(x, vals, 0.6, color=cols, edgecolor="black", lw=.5)
    ax.axhline(0, color="black", lw=.8)
    span = max(np.abs(vals).max(), 1e-6)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.02 * span if v >= 0 else -0.05 * span), f"{v:+.2f}",
                ha="center", fontsize=6.2)
    ax.set_xticks(x); ax.set_xticklabels(df["Class"], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Net area bias (km²)  [+ = under-mapped]")
    ax.grid(axis="y", alpha=.35, ls="--", lw=.5); despine(ax)
    fig.tight_layout(); save_fig(fig, out_stem)


def plot_transition_matrix(T, out_stem):
    fig, ax = plt.subplots(figsize=(COL1 * 1.5, 62 * MM))
    Tn = T.astype(float)
    vmax = max(Tn.max(), 1)
    im = ax.imshow(Tn, cmap="YlOrRd", aspect="auto", vmax=vmax)
    ax.set_xticks(range(C)); ax.set_xticklabels([CLASS_NAME_EN[c] for c in range(C)],
                                                rotation=90, fontsize=6)
    ax.set_yticks(range(C)); ax.set_yticklabels([CLASS_NAME_EN[c] for c in range(C)], fontsize=6)
    ax.set_xlabel("To (omission class)"); ax.set_ylabel("From (commission class)")
    for i in range(C):
        for j in range(C):
            if T[i, j] > 0:
                ax.text(j, i, f"{T[i,j]}", ha="center", va="center", fontsize=5,
                        color="white" if T[i, j] > vmax * 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04).ax.tick_params(labelsize=6)
    ax.set_title("Directed transition matrix (pixel counts)", fontsize=7.5)
    fig.tight_layout(); save_fig(fig, out_stem)


def plot_tier_distribution(tier, flagged, out_stem):
    counts = [int((tier == t).sum()) for t in ("A", "B", "C")]
    if sum(counts) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 56 * MM))
    ax = axes[0]
    ax.bar(["A", "B", "C"], counts, 0.6, color=[TIER_COLOR[t] for t in ("A", "B", "C")],
           edgecolor="black", lw=.5)
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.01, f"{v:,}", ha="center", fontsize=6.5)
    ax.set_ylabel("Pixels"); ax.set_title("Confidence tier counts", fontsize=7.5)
    ax.grid(axis="y", alpha=.35, ls="--", lw=.5); despine(ax); panel_label(ax, "(a)")
    ax = axes[1]
    ax.pie(counts, labels=["A", "B", "C"], colors=[TIER_COLOR[t] for t in ("A", "B", "C")],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 7})
    ax.set_title("Tier share", fontsize=7.5); panel_label(ax, "(b)")
    fig.tight_layout(); save_fig(fig, out_stem)


def plot_workload(work, out_stem):
    fig, ax = plt.subplots(figsize=(COL1 * 1.3, 56 * MM))
    labs = ["Pixel-wise\n(directed)", "Object-wise\n(directed)", "Pixel-wise\n(all flagged)"]
    vals = [work["est_hours_pixelwise_directed"], work["est_hours_objectwise_directed"],
            work["est_hours_pixelwise_all"]]
    cols = ["#59A14F", "#F28E2B", "#E15759"]
    ax.bar(np.arange(3), vals, 0.55, color=cols, edgecolor="black", lw=.5)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.015, f"{v:.1f} h", ha="center", fontsize=6.5)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(labs, fontsize=6.5)
    ax.set_ylabel("Expert review time (hours)")
    ax.set_title("Review workload by strategy", fontsize=7.5)
    ax.grid(axis="y", alpha=.35, ls="--", lw=.5); despine(ax)
    fig.tight_layout(); save_fig(fig, out_stem)


def plot_cross_generator(df, out_stem):
    g = df.groupby("Generator")[["Recall (%)", "AUC", "AP"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(COL2, 56 * MM))
    x = np.arange(len(g)); w = 0.25
    ax.bar(x - w, g["Recall (%)"], w, label="Recall (%)", color="#3A7CA5",
           edgecolor="black", lw=.5)
    ax.bar(x, g["AUC"] * 100, w, label="AUC ×100", color="#59A14F",
           edgecolor="black", lw=.5)
    ax.bar(x + w, g["AP"] * 100, w, label="AP ×100", color="#F28E2B",
           edgecolor="black", lw=.5)
    ax.set_xticks(x); ax.set_xticklabels(g["Generator"], rotation=15, ha="right", fontsize=6.5)
    ax.set_ylabel("%"); ax.set_ylim(0, 108)
    ax.legend(fontsize=6.5, frameon=False, ncol=3)
    ax.grid(axis="y", alpha=.35, ls="--", lw=.5); despine(ax)
    fig.tight_layout(); save_fig(fig, out_stem)


# =============================================================================
# 16. 主流程 + CLI
# =============================================================================
def _load_models(out_dir, img, train_seeds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for ts in train_seeds:
        p = os.path.join(out_dir, f"model_full_{ts}.pt")
        if os.path.exists(p):
            m = DualResUNetV6(in_ch=img.shape[2]).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            models.append(m)
    return models


def _load_val_ensemble_per_eval(out_dir, eval_seeds):
    """[Q1] 读取逐实现模型集合; 返回 {es: (score, gt, valid)}。"""
    out = {}
    for es in eval_seeds:
        p = os.path.join(out_dir, "validation", f"val_ensemble_es{es}.npz")
        if os.path.exists(p):
            d = np.load(p)
            out[es] = (d["score"].astype(np.float32), d["gt"].astype(bool), d["valid"].astype(bool))
    return out


def _load_val_ensemble_legacy(out_dir):
    p = os.path.join(out_dir, "validation", "val_ensemble.npz")
    if os.path.exists(p):
        d = np.load(p)
        return d["score"].astype(np.float32), d["gt"].astype(bool), d["valid"].astype(bool)
    return None, None, None


def _load_calibration(out_dir):
    p = os.path.join(out_dir, "calibration.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return None


def run_pipeline(img, prod, valid, out_dir, train_seeds=(42, 43, 44),
                 eval_seeds=(100, 101, 102), stage1_epochs=12, stage2_epochs=25,
                 patch=128, n_patch=1500, bs=4, detect_ratio=0.10, injection=0.06,
                 cls_target="clean", cls_w=0.1, target_precision=0.90,
                 candidate_mode="disagree", score_hi_quantile=0.25,
                 max_items=100000, n_chips=12, spectral_mode="presented",
                 do_validation=True, do_deploy=True):
    set_pub_style()
    for sub in ("figures", "tables", "validation", "deploy"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    meta = _env_meta()
    meta.update({"n_valid": int(valid.sum()), "scene_shape": list(valid.shape),
                 "train_seeds": list(train_seeds), "eval_seeds": list(eval_seeds),
                 "detect_ratio_validation": detect_ratio, "injection": injection,
                 "cls_target": cls_target, "target_precision": target_precision,
                 "candidate_mode": candidate_mode, "score_hi_quantile": score_hi_quantile,
                 "stage1_epochs": stage1_epochs, "stage2_epochs": stage2_epochs,
                 "patch": patch, "n_patch": n_patch, "batch_size": bs,
                 "n_params_full": count_params(DualResUNetV6(in_ch=img.shape[2])),
                 "geo": {"px": GEO["px"], "crs": str(GEO["crs"]) if GEO["crs"] else None},
                 "patches_applied": ["Q1 per-realisation model-ensemble calibration",
                                     "Q2 two-source candidate set + sensitivity"],
                 "figure_note": "No graticule / Easting-Northing; scale bar + north arrow only"})
    log(f"[Model] parameters = {meta['n_params_full']:,}; cls_target={cls_target}")

    models = []
    t0 = time.time()
    for ts in train_seeds:
        set_seed(ts)
        model = DualResUNetV6(in_ch=img.shape[2])
        model = train_stage1(model, img, prod, valid, epochs=stage1_epochs, bs=bs,
                             n_patch=n_patch, patch=patch, seed=ts, tag=f"I[{ts}]")
        model = train_stage2(model, img, prod, valid, epochs=stage2_epochs, bs=bs,
                             n_patch=n_patch, patch=patch, target=injection, seed=ts,
                             tag=f"II[{ts}]", cls_target=cls_target, cls_w=cls_w)
        torch.save(model.state_dict(), os.path.join(out_dir, f"model_full_{ts}.pt"))
        models.append(model)
    meta["train_full_seconds"] = round(time.time() - t0, 1)

    val_res = None
    if do_validation:
        val_res = run_validation(models, img, prod, valid, out_dir, train_seeds,
                                 eval_seeds, detect_ratio, injection, target_precision)
    thr = val_res["thr"] if val_res else None
    if thr is None:
        calib = _load_calibration(out_dir)
        thr = calib["threshold"] if calib else 0.5
    meta["deploy_threshold"] = float(thr)
    meta.update(_peak_mem())
    jdump(meta, os.path.join(out_dir, "run_meta.json"))

    dep_res = None
    if do_deploy:
        dep_res = audit_real_product(models, img, prod, valid, out_dir, thr,
                                     candidate_mode, score_hi_quantile,
                                     max_items, n_chips, spectral_mode)

    if val_res is not None and dep_res is not None:
        s_val = val_res["score_ens"][valid]
        s_dep = dep_res["score"][valid]
        score_distribution_summary(s_val, s_dep, thr, os.path.join(out_dir, "tables"))
        plot_score_distribution(s_val, s_dep, thr, val_res["curve"], val_res["calib"],
                                os.path.join(out_dir, "figures", "D00_score_distribution"))
        log("\n[关键诊断图] figures/D00_score_distribution")
    log(f"\n[Done] 全部输出 -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=f"DS-ResUNet {VERSION} 校准修正+双源候选")
    ap.add_argument("--rs", default=None)
    ap.add_argument("--glc", default=None)
    ap.add_argument("--out", default="./V14_output")
    ap.add_argument("--train-seeds", default="42,43,44")
    ap.add_argument("--eval-seeds", default="100,101,102")
    ap.add_argument("--stage1-epochs", type=int, default=12)
    ap.add_argument("--stage2-epochs", type=int, default=25)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--patches", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--detect-ratio", type=float, default=0.10,
                    help="仅用于验证模式的预算精确分位数")
    ap.add_argument("--injection", type=float, default=0.06)
    ap.add_argument("--cls-target", default="clean", choices=["clean", "presented"])
    ap.add_argument("--cls-w", type=float, default=0.1)
    ap.add_argument("--target-precision", type=float, default=0.90,
                    help="[Q1] 逐实现校准的目标精度")
    ap.add_argument("--deploy-thr", type=float, default=None,
                    help="手动指定部署绝对阈值 (覆盖校准)")
    ap.add_argument("--candidate-mode", default="disagree", choices=CAND_MODES,
                    help="[Q2] 部署候选模式: disagree(默认)/score/union/intersect")
    ap.add_argument("--score-hi-quantile", type=float, default=0.25,
                    help="[Q2] 候选集内 top 分位定为高分(分级用)")
    ap.add_argument("--max-items", type=int, default=100000)
    ap.add_argument("--n-chips", type=int, default=12)
    ap.add_argument("--spectral-mode", default="presented", choices=["presented", "clean"])
    ap.add_argument("--px", type=float, default=None)
    ap.add_argument("--deploy-only", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    train_seeds = tuple(int(s) for s in args.train_seeds.split(",") if s.strip())
    eval_seeds = tuple(int(s) for s in args.eval_seeds.split(",") if s.strip())
    os.makedirs(args.out, exist_ok=True)
    set_pub_style()

    img = prod = valid = None
    if args.rs and args.glc:
        img, prod, valid = load_scene(args.rs, args.glc)
    elif not (args.deploy_only or args.validate_only):
        log("[Data] 无 GeoTIFF -> 合成烟雾测试 (非发表结果)")
        img, prod, valid = make_synthetic_scene(512, 512)
    if args.px is not None:
        set_geo(px=args.px)

    if args.deploy_only or args.validate_only:
        if img is None:
            log("[Error] 需要 --rs / --glc"); return
        models = _load_models(args.out, img, train_seeds)
        if not models:
            log("[Error] 未找到检查点, 请先完整训练"); return
        val_res = None
        if args.validate_only:
            val_res = run_validation(models, img, prod, valid, args.out, train_seeds,
                                     eval_seeds, args.detect_ratio, args.injection,
                                     args.target_precision)
        thr = args.deploy_thr
        calib = _load_calibration(args.out)
        if thr is None and val_res is not None:
            thr = val_res["thr"]
        if thr is None and calib is not None:
            thr = calib["threshold"]
        if thr is None:
            per = _load_val_ensemble_per_eval(args.out, eval_seeds)
            if per:
                thrs = []
                for es, (sv, gv, vv) in per.items():
                    t_e, _ = calibrate_threshold(sv, gv, vv, args.target_precision)
                    thrs.append(t_e)
                thr = float(np.median(thrs))
            else:
                sv, gv, vv = _load_val_ensemble_legacy(args.out)
                if sv is not None:
                    thr, _ = calibrate_threshold(sv, gv, vv, args.target_precision)
        if thr is None:
            thr = 0.5
            log("[Warn] 无校准信息, 使用默认阈值 0.5")
        dep = audit_real_product(models, img, prod, valid, args.out, thr,
                                 args.candidate_mode, args.score_hi_quantile,
                                 args.max_items, args.n_chips, args.spectral_mode)
        # 诊断图: 用逐实现集合的首个实现
        per = _load_val_ensemble_per_eval(args.out, eval_seeds)
        if per:
            es0 = eval_seeds[0] if eval_seeds[0] in per else list(per.keys())[0]
            sv, gv, vv = per[es0]
            s_val = sv[vv]; s_dep = dep["score"][valid]
            score_distribution_summary(s_val, s_dep, thr, os.path.join(args.out, "tables"))
            curve = threshold_curve(sv, gv, vv)
            plot_score_distribution(s_val, s_dep, thr, curve,
                                    calib or {"target_precision": args.target_precision},
                                    os.path.join(args.out, "figures", "D00_score_distribution"))
            log("\n[关键诊断图] figures/D00_score_distribution")
        return

    run_pipeline(img, prod, valid, args.out, train_seeds, eval_seeds,
                 args.stage1_epochs, args.stage2_epochs, args.patch, args.patches,
                 args.batch_size, args.detect_ratio, args.injection, args.cls_target,
                 args.cls_w, args.target_precision, args.candidate_mode,
                 args.score_hi_quantile, args.max_items, args.n_chips, args.spectral_mode)


if __name__ == "__main__":
    main()