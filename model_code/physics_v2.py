# -*- coding: utf-8 -*-
"""
physics_v2.py — SIM 物理前向（3 相位 × 2 方向 = 6 RAW）
支持：
- enable_zernike: 是否启用 Zernike 像差（ON/OFF）
- zernike_order:  最高 Zernike 阶（含径向阶）
- zernike_scale:  系数总体尺度（越大像差越强）
- zernike_fixed:  True=固定一组像差；False=每次前向（或每 batch）轻微扰动
- use_noise:      是否在 RAW 上叠加噪声（高斯 + 少量泊松近似）
说明：
- 这里的 OTF = 振幅（高斯近似低通） × 相位项（由 Zernike 波前产生）
- 这是“合理且稳定”的近似实现，非严格显微光学推导，但对鲁棒训练充足
"""
from __future__ import annotations
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------- Zernike 多项式 --------------------------
# 极坐标 (r, theta)；r∈[0,1] 单位圆，theta∈[-pi, pi]
def _zernike_radial(n: int, m: int, r: torch.Tensor) -> torch.Tensor:
    # n: radial order, m: azimuthal frequency, |m|<=n, (n-m) even
    m = abs(m)
    R = torch.zeros_like(r)
    for k in range((n - m)//2 + 1):
        # R_n^m(r) = sum_{k=0}^{(n-m)/2} (-1)^k (n-k)! / [ k! ((n+m)/2 - k)! ((n-m)/2 - k)! ] * r^{n-2k}
        num = math.factorial(n - k)
        den = math.factorial(k) * math.factorial((n + m)//2 - k) * math.factorial((n - m)//2 - k)
        R = R + ((-1)**k) * (num / den) * (r ** (n - 2*k))
    return R

def _zernike(n: int, m: int, r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # 规范化 Zernike（未做能量归一，这里只做结构形状）
    # m>0: R_n^m(r)*cos(mθ); m<0: R_n^{|m|}(r)*sin(|m|θ); m=0: R_n^0(r)
    R = _zernike_radial(n, m, r)
    if m > 0:
        return R * torch.cos(m * theta)
    elif m < 0:
        return R * torch.sin(abs(m) * theta)
    else:
        return R

def _gen_zernike_phase(h: int, w: int,
                       device: torch.device,
                       order: int = 4,
                       coeffs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    生成单位圆孔径内的 Zernike 相位 φ(r,θ)。order=4 ~ 含 n<=4 的项。
    coeffs: [K]，顺序采用 (n,m) 字典序遍历（略过 (n-m) 为奇的非法项）。
    """
    ys = torch.linspace(-(h-1)/2, (h-1)/2, steps=h, device=device, dtype=torch.float32)
    xs = torch.linspace(-(w-1)/2, (w-1)/2, steps=w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    # 频域归一化半径 r∈[0,1]
    rr = torch.sqrt(xx*xx + yy*yy); rr = rr / (rr.max() + 1e-6)
    th = torch.atan2(yy, xx)

    # 单位圆外的相位置零
    mask = (rr <= 1.0).float()

    # 逐项叠加
    terms = []
    for n in range(order + 1):
        for m in range(-n, n+1, 2):  # 只取 (n-m) 偶数的合法项
            if (n - m) % 2 == 0:
                terms.append((n, m))
    K = len(terms)
    if coeffs is None:
        coeffs = torch.zeros(K, device=device, dtype=torch.float32)

    phi = torch.zeros((h, w), device=device, dtype=torch.float32)
    for i, (n, m) in enumerate(terms):
        Z = _zernike(n, m, rr, th) * mask
        phi = phi + coeffs[i] * Z
    return phi  # 实数相位（弧度）


# -------------------------- SIM 物理前向 --------------------------
class SIMForward(nn.Module):
    """
    生成 6 幅 RAW：3 相位（alpha）× 2 方向（orient 0/90 度，使用 beta 做相位）
    forward(gt, alpha[B,3], beta[B,3], mod_scale[B], cycle_scale[B]) -> RAW [B,6,H,W]
    """
    def __init__(self,
                 use_noise: bool = True,
                 enable_zernike: bool = False,
                 zernike_order: int = 4,
                 zernike_scale: float = 0.15,
                 zernike_fixed: bool = True,
                 base_cycles: float = 24.0,
                 otf_sigma: float = 0.18):
        super().__init__()
        self.use_noise = bool(use_noise)
        self.enable_zernike = bool(enable_zernike)
        self.zernike_order = int(zernike_order)
        self.zernike_scale = float(zernike_scale)
        self.zernike_fixed = bool(zernike_fixed)
        self.base_cycles = float(base_cycles)
        self.otf_sigma = float(otf_sigma)

        self.register_buffer("_cached_phi", None, persistent=False)  # 固定像差时缓存

    # -------- 新增：运行时控制接口（供 test/eval 动态切换） --------
    def set_enable_zernike(self, flag: bool) -> None:
        self.enable_zernike = bool(flag)
        # 切换像差时，清除缓存以重新生成 φ
        self._cached_phi = None

    def get_enable_zernike(self) -> bool:
        return bool(self.enable_zernike)

    def set_zernike_scale(self, scale: float) -> None:
        self.zernike_scale = float(scale)
        self._cached_phi = None

    # --------------------------------------------------------------

    def _build_otf(self, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回（振幅 A(f)、像差相位 φ(f)）
        A 采用高斯近似；φ 由 Zernike 产生；二者都在频域定义。
        """
        # 振幅 — 高斯低通
        ys = torch.linspace(-(h-1)/2, (h-1)/2, steps=h, device=device, dtype=torch.float32)
        xs = torch.linspace(-(w-1)/2, (w-1)/2, steps=w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        rr2 = (xx**2 + yy**2)
        rr2 = rr2 / (rr2.max() + 1e-6)  # 归一化到 [0,1]
        A = torch.exp(- rr2 / (2.0 * (self.otf_sigma**2))).contiguous()

        # 相位 — Zernike
        if self.enable_zernike:
            if self.zernike_fixed and (self._cached_phi is not None) and \
               (self._cached_phi.shape[-2] == h) and (self._cached_phi.shape[-1] == w):
                phi = self._cached_phi
            else:
                # 系数：随阶次衰减，近轴项更强；总体由 zernike_scale 控制
                terms = []
                for n in range(self.zernike_order + 1):
                    for m in range(-n, n+1, 2):
                        if (n - m) % 2 == 0:
                            terms.append((n, m))
                K = len(terms)
                coeffs = torch.zeros(K, device=device, dtype=torch.float32)
                for i, (n, m) in enumerate(terms):
                    decay = 1.0 / (1.0 + n)  # 简单随阶次衰减
                    coeffs[i] = self.zernike_scale * decay * torch.randn((), device=device)
                phi = _gen_zernike_phase(h, w, device, self.zernike_order, coeffs)
                if self.zernike_fixed:
                    self._cached_phi = phi
        else:
            phi = torch.zeros((h, w), device=device, dtype=torch.float32)

        return A, phi

    def _apply_otf(self, img: torch.Tensor, A: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        频域滤波：Y = FFT(img) * A * exp(iφ) -> IFFT
        输入 img: [B,1,H,W] or [B,C,H,W]
        """
        X = torch.fft.fft2(img.float(), dim=(-2, -1), norm="ortho")
        X = torch.fft.fftshift(X, dim=(-2, -1))
        H = A * torch.exp(1j * phi)
        Y = X * H
        Y = torch.fft.ifftshift(Y, dim=(-2, -1))
        y = torch.fft.ifft2(Y, dim=(-2, -1), norm="ortho").real
        return y.to(img.dtype)

    @torch.no_grad()
    def _add_noise(self, raw: torch.Tensor) -> torch.Tensor:
        """
        近似噪声：少量高斯 + 轻微泊松近似（把 raw 当作强度，噪声幅度与 sqrt(I) 成比例）
        """
        if not self.use_noise: return raw
        gauss = 0.01 * torch.randn_like(raw)
        # 简单泊松近似：以 sqrt(max(I,1e-3)) 为幅度
        poi = 0.005 * torch.sqrt(raw.clamp_min(1e-3)) * torch.randn_like(raw)
        out = (raw + gauss + poi).clamp(0.0, 1.0)
        return out

    def forward(self,
                gt: torch.Tensor,            # [B,1,H,W] 或 [B,3,H,W]（会取灰度）
                alpha: torch.Tensor,         # [B,3] 三个相位（方向 0°）
                beta: torch.Tensor,          # [B,3] 三个相位（方向 90°）
                mod_scale: torch.Tensor,     # [B]   调制度缩放（>0）
                cycle_scale: torch.Tensor    # [B]   周期缩放（>0）
                ) -> torch.Tensor:
        if gt.size(1) != 1:
            gt = gt.mean(dim=1, keepdim=True)
        B, C, H, W = gt.shape
        device = gt.device

        # 频域 OTF（包含 Zernike 相位）
        A, phi = self._build_otf(H, W, device)

        # 基础空间频率（每幅图的条纹周期数）
        kx0 = 2.0 * math.pi * self.base_cycles / float(W)
        ky0 = 2.0 * math.pi * self.base_cycles / float(H)

        # 栅格
        xs = torch.linspace(0, float(W-1), steps=W, device=device).view(1, 1, 1, W)
        ys = torch.linspace(0, float(H-1), steps=H, device=device).view(1, 1, H, 1)

        raws = []
        for i in range(3):
            # 两个正交方向：0° 用 alpha，相位=alpha[:,i]；90° 用 beta，相位=beta[:,i]
            # 周期缩放（每个 batch 一个缩放）：kx = kx0 * cyc; ky = ky0 * cyc
            cyc = cycle_scale.view(B, 1, 1, 1)
            kx = kx0 * cyc
            ky = ky0 * cyc

            # mod_scale
            mod = mod_scale.view(B, 1, 1, 1).clamp_min(1e-3)

            # │ orient 0°：沿 x 条纹
            # I = gt * (1 + m * cos(kx*x + alpha))
            pha_a = alpha[:, i].view(B, 1, 1, 1)
            patt_a = (1.0 + mod * torch.cos(kx * xs + pha_a)).to(gt.dtype)
            img_a = (gt * patt_a).clamp(0.0, 1.0)
            img_a = self._apply_otf(img_a, A, phi)

            # │ orient 90°：沿 y 条纹
            pha_b = beta[:, i].view(B, 1, 1, 1)
            patt_b = (1.0 + mod * torch.cos(ky * ys + pha_b)).to(gt.dtype)
            img_b = (gt * patt_b).clamp(0.0, 1.0)
            img_b = self._apply_otf(img_b, A, phi)

            raws.extend([img_a, img_b])

        raw = torch.cat(raws, dim=1)  # [B,6,H,W]
        raw = self._add_noise(raw)
        return raw
