# -*- coding: utf-8 -*-
"""
utils.py — 训练/评测常用工具函数
"""
from __future__ import annotations
import math, random
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

try:
    from config import CFG
except Exception:
    class _Dummy: pass
    CFG = _Dummy()

def set_seed(seed: int = 3407, deterministic: bool = False) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = bool(deterministic)
        cudnn.benchmark = not deterministic
    except Exception:
        pass

def psnr_batch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if pred.dim() == 3:   pred = pred.unsqueeze(0)
    if target.dim() == 3: target = target.unsqueeze(0)
    x = pred.float().clamp(0, 1); y = target.float().clamp(0, 1)
    mse = F.mse_loss(x, y, reduction="none").flatten(1).mean(dim=1).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)

def _gaussian_window(win: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(win, device=device, dtype=dtype) - (win - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma)); g = g / g.sum()
    w2d = (g[:, None] @ g[None, :]).contiguous(); w2d = w2d / w2d.sum()
    return w2d.view(1, 1, win, win)

def ssim_batch(pred: torch.Tensor, target: torch.Tensor,
               win_size: int = 11, sigma: float = 1.5,
               K1: float = 0.01, K2: float = 0.03,
               data_range: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    if pred.dim() == 3:   pred = pred.unsqueeze(0)
    if target.dim() == 3: target = target.unsqueeze(0)
    x = pred.float().clamp(0, 1); y = target.float().clamp(0, 1)
    B, C, H, W = x.shape
    win = min(H, W, win_size); win = win - 1 if (win % 2 == 0) else win
    if win < 3:
        mse = F.mse_loss(x, y, reduction="none").flatten(1).mean(1)
        return (1.0 / (1.0 + mse)).clamp(0.0, 1.0)
    w = _gaussian_window(win, sigma, device=x.device, dtype=x.dtype).expand(C, 1, win, win)
    pad = win // 2
    C1 = (K1 * data_range) ** 2; C2 = (K2 * data_range) ** 2
    mu_x = F.conv2d(x, w, padding=pad, groups=C); mu_y = F.conv2d(y, w, padding=pad, groups=C)
    mu_x2 = mu_x * mu_x; mu_y2 = mu_y * mu_y; mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, w, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=pad, groups=C) - mu_xy
    ssim_map = ((2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + eps)
    return ssim_map.flatten(1).mean(1).clamp(0.0, 1.0)

# ---- 频域一致性 ----
_RADIAL_CACHE = {}
def fft_mag(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3: x = x.unsqueeze(0)
    X = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
    X = torch.fft.fftshift(X, dim=(-2, -1))
    return torch.abs(X)

def make_radial_weight(H: int, W: int, device, lo: float = 0.2, hi: float = 1.0, p: float = 2.0) -> torch.Tensor:
    key = (int(H), int(W), str(device))
    w = _RADIAL_CACHE.get(key, None)
    if w is None:
        ys = torch.linspace(-(H - 1) / 2.0, (H - 1) / 2.0, steps=H, device=device, dtype=torch.float32)
        xs = torch.linspace(-(W - 1) / 2.0, (W - 1) / 2.0, steps=W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        r = torch.sqrt(xx * xx + yy * yy); r = r / (r.max() + 1e-6)
        w = (lo + (hi - lo) * (r ** p)).contiguous()
        _RADIAL_CACHE[key] = w
    return w

def frequency_consistency_loss(rec: torch.Tensor, gt: torch.Tensor, hi_boost: float = 1.0) -> torch.Tensor:
    if rec.dim() == 3: rec = rec.unsqueeze(0)
    if gt.dim()  == 3: gt  = gt .unsqueeze(0)
    ds = int(getattr(CFG, "FREQ_DOWNSAMPLE", 1))
    if ds > 1:
        rec = torch.nn.functional.avg_pool2d(rec, ds, ds)
        gt  = torch.nn.functional.avg_pool2d(gt , ds, ds)
    B, C, H, W = rec.shape
    w = make_radial_weight(H, W, rec.device, lo=0.2, hi=1.0, p=2.0).view(1,1,H,W)
    Mr = fft_mag(rec); Mg = fft_mag(gt)
    return (w * (Mr - Mg).float()).pow(2).flatten(1).mean()

# ---- RL 动作映射（eval_envs也复用）----
def map_action_to_controls(action: torch.Tensor, deg_limit: float, mod_range_pct: float, cycle_range_pct: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(action): action = torch.tensor(action, dtype=torch.float32)
    a = action
    if a.dim() == 1: a = a.unsqueeze(0)
    import math as _m
    B = a.size(0); a = a.to(dtype=torch.float32)
    rad = float(deg_limit) * _m.pi / 180.0
    da = a[:, 0:3] * rad; db = a[:, 3:6] * rad
    mod_scale = 1.0 + float(mod_range_pct)  * (a[:, 6] if a.size(1) >= 7 else 0.0)
    cyc_scale = 1.0 + float(cycle_range_pct) * (a[:, 7] if a.size(1) >= 8 else 0.0)
    mod_scale = torch.clamp(mod_scale, min=1e-3).to(a.device)
    cyc_scale = torch.clamp(cyc_scale, min=1e-3).to(a.device)
    return da, db, mod_scale, cyc_scale
