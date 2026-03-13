# -*- coding: utf-8 -*-
"""
wienersim_core.py

频域 Wiener-SIM 基线（不依赖训练）：
- 适配 RL-SIM 的 physics_v2.SIMForward 生成的 6 帧 RAW：
  3 phases × 2 orientations (0°: α, 90°: β)，帧顺序为：
  [a0, b0, a1, b1, a2, b2]。
- 采用经典 3 相位解调 + 频移 + Wiener 反卷积 + 频谱加权融合。
- 假定两条条纹的空间频率幅值相同，方向分别为 x / y。

注意：
- 本实现只在模拟环境中用作“解析频域 SIM”参考基线，
  不用于真实光机标定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ====================== FFT / OTF 工具 ======================

def fft2c(x: np.ndarray) -> np.ndarray:
    """中心化 FFT2：spatial -> frequency."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def ifft2c(X: np.ndarray) -> np.ndarray:
    """中心化 IFFT2：frequency -> spatial."""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(X, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def otf_circular(
    shape_hw: Tuple[int, int],
    pixel_um: float,
    wavelength_um: float,
    na: float,
) -> np.ndarray:
    """
    经典圆孔 incoherent OTF（Hopkins 公式），返回复数数组（实值）。
    """
    H, W = shape_hw
    fy = np.fft.fftshift(np.fft.fftfreq(H, d=pixel_um))
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=pixel_um))
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    f = np.sqrt(FX ** 2 + FY ** 2)  # cycles / µm
    fc = 2.0 * na / wavelength_um   # cutoff
    rho = f / fc

    OTF = np.zeros_like(rho, dtype=np.float64)
    inside = rho <= 1.0
    r = rho[inside]
    # Hopkins closed form
    OTF[inside] = (2.0 / np.pi) * (
        np.arccos(r) - r * np.sqrt(1.0 - r**2)
    )
    return OTF.astype(np.complex128)


def fourier_shift_spectrum(F: np.ndarray, dk_yx: Tuple[float, float]) -> np.ndarray:
    """
    在频域中执行子像素频移：通过空间域相位斜坡实现。
    dk_yx: (Δky, Δkx)，单位：cycles/pixel。
    """
    H, W = F.shape[-2], F.shape[-1]
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]
    ky, kx = float(dk_yx[0]), float(dk_yx[1])
    phase = np.exp(1j * 2.0 * np.pi * (ky * yy + kx * xx))
    return fft2c(ifft2c(F) * phase)


def harmonic_demodulation(F_lj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    3‑phase 解调：
    - 输入 F_lj 形状 (J=3, H, W)，假设相位为 0, 2π/3, 4π/3 或其整体平移。
    - 输出 0、+1、−1 次谐波分量。
    """
    assert F_lj.shape[0] == 3, "目前只支持 J=3"
    J = 3
    j = np.arange(J, dtype=np.float64)
    w = np.exp(-1j * 2.0 * np.pi * j / J)  # e^{-i 2π j / 3}

    F0 = F_lj.mean(axis=0)
    Fp1 = (F_lj * w[:, None, None]).mean(axis=0)
    Fm1 = (F_lj * np.conjugate(w)[:, None, None]).mean(axis=0)
    return F0, Fp1, Fm1


# ====================== 配置与主函数 ======================

@dataclass
class WienerSimConfig:
    """
    Wiener‑SIM 解析重建配置。
    这里的参数含义：
    - pixel_um, wavelength_um, na: 成像系统参数
    - kappa: |k| / fc，其中 fc = 2NA/λ（incoherent cutoff，cycles/µm）
    - mod_depth: 条纹调制度
    - wiener_const: Wiener 正则常数
    """
    pixel_um: float = 0.065
    wavelength_um: float = 0.520
    na: float = 1.49
    kappa: float = 0.80
    mod_depth: float = 0.90
    wiener_const: float = 1e-3


def wiener_sim_recon(raw6: np.ndarray, cfg: WienerSimConfig) -> np.ndarray:
    """
    从 6 帧 RAW（3 phases × 2 orientations）进行 Wiener‑SIM 解析重建。

    参数
    ----
    raw6 : np.ndarray
        形状 (B, 6, H, W)，数值已归一化到 [0,1]。
        帧顺序需满足 physics_v2.SIMForward 的约定：
        [a0, b0, a1, b1, a2, b2]。
    cfg : WienerSimConfig
        Wiener‑SIM 配置。

    返回
    ----
    recon : np.ndarray
        形状 (B, H, W)，归一化到 [0,1] 的重建结果。
    """
    assert raw6.ndim == 4, "raw6 必须是 (B,6,H,W)"
    B, C, H, W = raw6.shape
    assert C == 6, "当前实现假设 RAW 为 6 帧（3 相位 × 2 方向）"

    # 2 orientations × 3 phases
    L, J = 2, 3
    raw_lj = raw6.reshape(B, L, J, H, W)

    # OTF（与 MAP‑SIM 一致的圆孔模型）
    Hc = otf_circular((H, W), cfg.pixel_um, cfg.wavelength_um, cfg.na)
    Hstar = np.conjugate(Hc)
    absH2 = np.abs(Hc) ** 2

    # 频域网格（用于加权窗口）
    fy = np.fft.fftshift(np.fft.fftfreq(H, d=cfg.pixel_um))
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=cfg.pixel_um))
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    f_radius = np.sqrt(FX**2 + FY**2)
    fc = 2.0 * cfg.na / cfg.wavelength_um  # cutoff (cycles/µm)

    # 简单的频谱加权：中心衰减较弱，靠近截止频率位置略加强侧带
    w0 = np.exp(-(f_radius / (0.6 * fc)) ** 4)          # baseband window
    wsb = np.exp(-((f_radius - 0.8 * fc) / (0.4 * fc)) ** 4)  # sideband window

    # 条纹空间频率（cycles/pixel）
    k_mag_cpp = cfg.kappa * fc * cfg.pixel_um
    # orientation 0°: k = (0, kx), orientation 90°: k = (ky, 0)
    kvecs = np.array([[0.0, k_mag_cpp], [k_mag_cpp, 0.0]], dtype=np.float64)

    m = float(cfg.mod_depth)
    m = max(m, 1e-3)
    scale_sideband = 2.0 / m  # (m/2) 的反向缩放

    eps = 1e-8
    gamma = float(cfg.wiener_const)

    recon = np.zeros((B, H, W), dtype=np.float64)

    for b in range(B):
        S_hat = np.zeros((H, W), dtype=np.complex128)
        W_acc = np.zeros((H, W), dtype=np.float64)

        for l in range(L):
            # 该方向的 3 相位帧
            F_lj = fft2c(raw_lj[b, l])  # (3,H,W)
            F0, Fp1, Fm1 = harmonic_demodulation(F_lj)

            # baseband：H* F0 / (|H|^2 + γ)
            denom = absH2 + gamma
            S0 = Hstar * F0 / denom

            # sidebands：补偿调制度，频移回中心
            Sp = scale_sideband * Hstar * Fp1 / denom
            Sm = scale_sideband * Hstar * Fm1 / denom

            ky, kx = kvecs[l]
            Sp_c = fourier_shift_spectrum(Sp, (ky, kx))
            Sm_c = fourier_shift_spectrum(Sm, (-ky, -kx))

            # 累积（简单线性融合）
            S_hat += w0 * S0 + wsb * (Sp_c + Sm_c)
            W_acc += w0 + 2.0 * wsb

        S_hat /= (W_acc + eps)
        img = np.real(ifft2c(S_hat))
        recon[b] = img

    # 归一化到 [0,1]
    recon -= recon.min(axis=(-2, -1), keepdims=True)
    maxv = recon.max(axis=(-2, -1), keepdims=True)
    maxv[maxv < eps] = 1.0
    recon /= maxv
    return recon.astype(np.float32)
