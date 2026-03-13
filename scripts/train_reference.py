# -*- coding: utf-8 -*-
"""
多实验（APCRL_Zon_SSIM / APCRL_Zoff / APCOnly / RLOnly）训练脚本（SAC 版本，带 history.csv 日志）
- 物理前向 -> 重建网络 -> 物理再投影 -> 重建/物理/频域损失
- RL 阶段使用 SAC 调节 [相位(6) + 调制度 + 周期] 共 8 维连续动作
- 每个实验会把每个 epoch 的统计写入 work_all/<RunName>/history.csv，便于后续绘图
保留并沿用你工程里的接口与实现：SIMForward.forward、SCUNetRecon、utils.ssim_batch
"""

import os, math, time, random, csv
from collections import deque
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- AMP 兼容导入（兼容 PyTorch 2.0+ 的 torch.amp 与旧版 torch.cuda.amp）----
try:
    from torch.amp import autocast as _autocast_amp
    from torch.amp import GradScaler as _GradScaler_amp
    def amp_autocast(device_type: str, enabled: bool):
        return _autocast_amp(device_type, enabled=enabled)
    def make_scaler(enabled: bool):
        return _GradScaler_amp("cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _autocast_amp
    from torch.cuda.amp import GradScaler as _GradScaler_amp
    def amp_autocast(device_type: str, enabled: bool):
        return _autocast_amp(enabled=enabled)
    def make_scaler(enabled: bool):
        return _GradScaler_amp(enabled=enabled)

# ------------------------------ Config ------------------------------
class CFG:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_DIR = r"E:\data\GT"
    WORK_DIR  = r"E:\code\RL-SIM\work_all"
    MIXED_PRECISION = True
    CHANNELS_LAST = True
    NUM_WORKERS = 4
    BATCH_SIZE  = 8
    PATCH_SIZE  = 256
    SEED = 2024

    # 前两阶段长度
    EPOCHS_WARMUP = 4
    EPOCHS_FIXED  = 4

    # 重建网络优化
    LR_RECON = 2e-4
    WEIGHT_DECAY = 0.0
    CLIP_GRAD = 1.0

    # 物理/频域一致性权重（当 enable_apc=False 时会强制为 0）
    LAMBDA_PHYS_WARMUP = 0.0
    LAMBDA_PHYS_FIXED  = 0.10
    LAMBDA_PHYS_RL     = 0.10
    LAMBDA_FREQ_WARMUP = 0.0
    LAMBDA_FREQ_FIXED  = 0.05
    LAMBDA_FREQ_RL     = 0.05

    # 环境扰动强度（随 s 从 0.30 线性涨到 1.0）
    STRESS_MAX_DEG       = 30.0
    STRESS_MAX_MOD_PCT   = 0.35
    STRESS_MAX_CYCLE_PCT = 0.12
    STRESS_BLEACH_MAX    = 0.50
    EXTRA_STRIPE_PROB    = 0.30
    EXTRA_GAUSS_PROB     = 0.20
    EXTRA_BLEACH_PROB    = 0.10
    FREQ_HI_BOOST        = 1.2

    # --- SAC 超参数 ---
    RL_STATE_DIM   = 4
    RL_ACTION_DIM  = 8
    RL_HIDDEN      = 256
    RL_GAMMA       = 0.99
    RL_TAU         = 0.005
    RL_LR_ACTOR    = 1e-4
    RL_LR_CRITIC   = 1e-4
    RL_LR_ALPHA    = 5e-4
    RL_REPLAY_SIZE = 100_000
    RL_BATCH_SIZE  = 256
    RL_UPDATES_PER_STEP = 1
    RL_START_STEPS = 50
    RL_CLIP_STD    = 1e-3
    RL_ENTROPY_TGT = None
    RL_DEG_LIMIT   = 15.0
    RL_MOD_RANGE_PCT   = 0.20
    RL_CYCLE_RANGE_PCT = 0.08
    RL_ACT_L2       = 5e-4

    ENABLE_COMPILE = False  # 没有 Triton 时会自动回退

# 每个实验的总 epoch（Warmup+Fixed+RL/Fixed3 = epochs_total）
RUNS: Dict[str, Dict[str, Any]] = {
    "APCRL_Zon_SSIM": dict(enable_zernike=True,  enable_rl=True,  enable_apc=True,  epochs_total=76),
    "APCRL_Zoff"   : dict(enable_zernike=False, enable_rl=True,  enable_apc=True,  epochs_total=76),
    "APCOnly"      : dict(enable_zernike=True,  enable_rl=False, enable_apc=True,  epochs_total=76),
    "RLOnly"       : dict(enable_zernike=True,  enable_rl=True,  enable_apc=False, epochs_total=76),
}

# ------------------------------ Utils ------------------------------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    except Exception:
        pass

class EMA:
    def __init__(self, m=0.05): self.m=m; self.v=None
    def update(self, x): x=float(x); self.v=x if self.v is None else (1-self.m)*self.v + self.m*x
    def value(self): return 0.0 if self.v is None else float(self.v)

def charbonnier_loss(x, y, eps=1e-3): return torch.mean(torch.sqrt((x - y)**2 + eps*eps))

def grad_l1_loss(x, y):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    gx = y[..., :, 1:] - y[..., :, :-1]
    gy = y[..., 1:, :] - y[..., :-1, :]
    return (dx - gx).abs().mean() + (dy - gy).abs().mean()

def frequency_consistency_loss(rec, gt, hi_boost=1.0):
    x = rec.float(); y = gt.float()
    B, C, H, W = x.shape
    rec_fft = torch.fft.fft2(x, norm='ortho')
    gt_fft  = torch.fft.fft2(y, norm='ortho')
    rec_amp = torch.abs(rec_fft); gt_amp  = torch.abs(gt_fft)
    yy = torch.linspace(-1.0, 1.0, H, device=x.device)
    xx = torch.linspace(-1.0, 1.0, W, device=x.device)
    Y, X = torch.meshgrid(yy, xx, indexing='ij')
    radius = torch.sqrt(X**2 + Y**2)
    weight = (1.0 + (hi_boost - 1.0) * radius).view(1,1,H,W)
    return ((rec_amp - gt_amp)**2 * weight).mean().to(rec.dtype)

def canonical_patterns(B, device):
    phi = torch.tensor([0.0, 2*math.pi/3, 4*math.pi/3], device=device)
    alpha = phi.unsqueeze(0).expand(B, -1).contiguous()
    beta  = (phi + math.pi/2).unsqueeze(0).expand(B, -1).contiguous()
    return alpha, beta

def map_action_to_controls(action: torch.Tensor, B: int,
                           deg_limit=15.0, mod_range_pct=0.2, cycle_range_pct=0.08):
    a = action.tanh()
    a_alpha = a[:, 0:3]; a_beta = a[:, 3:6]
    d_alpha = (a_alpha * deg_limit) * math.pi / 180.0
    d_beta  = (a_beta  * deg_limit) * math.pi / 180.0
    mod_scale = 1.0 + mod_range_pct * a[:, 6]
    cyc_scale = 1.0 + cycle_range_pct * a[:, 7]
    return d_alpha, d_beta, mod_scale, cyc_scale

def try_compile(model: nn.Module):
    if not CFG.ENABLE_COMPILE:
        print("[compile] 已禁用 torch.compile"); return model, False
    try:
        import torch._dynamo  # noqa
        m = torch.compile(model, mode="max-autotune")
        print("[compile] torch.compile 已启用")
        return m, True
    except Exception as e:
        print(f"[compile] 回退到 eager（原因：{repr(e)}）。")
        return model, False

def print_data_info(prefix: str, loader):
    try:
        n_imgs = len(getattr(loader, "dataset", []))
    except Exception:
        n_imgs = "NA"
    print(f"[data] {prefix} {n_imgs} imgs | iters: {len(loader)} | patch={CFG.PATCH_SIZE}")

# ---------- 导入你现有模块 ----------
from physics_v2 import SIMForward               # 物理前向（你的实现）
from model_recon_scunet import SCUNetRecon     # 重建网络（你的实现）
from data import build_dataloaders              # 数据加载（你的实现）
from utils import ssim_batch                    # SSIM 计算（你的实现）  # ← 保持与工程一致  :contentReference[oaicite:2]{index=2}

# ------------------------------ SAC Agent ------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, act=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), act(),
            nn.Linear(hidden, hidden), act(),
            nn.Linear(hidden, out_dim)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

class Actor(nn.Module):
    def __init__(self, sdim, adim, hidden=256, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.fc1 = nn.Linear(sdim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu  = nn.Linear(hidden, adim)
        self.log_std = nn.Linear(hidden, adim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        for m in [self.fc1, self.fc2, self.mu, self.log_std]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, s):
        x = F.gelu(self.fc1(s))
        x = F.gelu(self.fc2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        std = log_std.exp().clamp(min=CFG.RL_CLIP_STD)
        return mu, std

    def sample(self, s):
        mu, std = self(s)
        eps = torch.randn_like(mu)
        z = mu + std * eps
        a = torch.tanh(z)
        logp_gauss = -0.5 * (((z - mu) / (std + 1e-8))**2 + 2*torch.log(std + 1e-8) + math.log(2*math.pi))
        logp_gauss = logp_gauss.sum(-1, keepdim=True)
        log_det = torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        logp = logp_gauss - log_det
        return a, logp

class QCritic(nn.Module):
    def __init__(self, sdim, adim, hidden=256):
        super().__init__()
        self.q = MLP(sdim + adim, 1, hidden)
    def forward(self, s, a): return self.q(torch.cat([s, a], dim=-1))

class SAC:
    def __init__(self, sdim, adim, device):
        self.device = device
        self.actor  = Actor(sdim, adim, 256).to(device)
        self.q1     = QCritic(sdim, adim, 256).to(device)
        self.q2     = QCritic(sdim, adim, 256).to(device)
        self.q1_t   = QCritic(sdim, adim, 256).to(device)
        self.q2_t   = QCritic(sdim, adim, 256).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=CFG.RL_LR_ACTOR)
        self.opt_q     = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=CFG.RL_LR_CRITIC)

        target_entropy = -float(adim) if CFG.RL_ENTROPY_TGT is None else float(CFG.RL_ENTROPY_TGT)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=CFG.RL_LR_ALPHA)
        self.target_entropy = target_entropy

        self.buf_s  = deque(maxlen=CFG.RL_REPLAY_SIZE)
        self.buf_a  = deque(maxlen=CFG.RL_REPLAY_SIZE)
        self.buf_r  = deque(maxlen=CFG.RL_REPLAY_SIZE)
        self.buf_ns = deque(maxlen=CFG.RL_REPLAY_SIZE)
        self.buf_d  = deque(maxlen=CFG.RL_REPLAY_SIZE)

    @property
    def alpha(self): return self.log_alpha.exp()

    def act(self, s_np, eval_mode=False):
        s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            if eval_mode:
                mu, std = self.actor(s); a = torch.tanh(mu)
            else:
                a, _ = self.actor.sample(s)
        return a.squeeze(0).cpu().numpy()

    def push(self, s, a, r, ns, d):
        self.buf_s.append(np.array(s, dtype=np.float32))
        self.buf_a.append(np.array(a, dtype=np.float32))
        self.buf_r.append(np.array([r], dtype=np.float32))
        self.buf_ns.append(np.array(ns, dtype=np.float32))
        self.buf_d.append(np.array([d], dtype=np.float32))

    def _sample_batch(self, batch_size):
        idx = np.random.randint(0, len(self.buf_s), size=batch_size)
        s  = torch.as_tensor(np.array([self.buf_s[i]  for i in idx]), device=self.device, dtype=torch.float32)
        a  = torch.as_tensor(np.array([self.buf_a[i]  for i in idx]), device=self.device, dtype=torch.float32)
        r  = torch.as_tensor(np.array([self.buf_r[i]  for i in idx]), device=self.device, dtype=torch.float32)
        ns = torch.as_tensor(np.array([self.buf_ns[i] for i in idx]), device=self.device, dtype=torch.float32)
        d  = torch.as_tensor(np.array([self.buf_d[i]  for i in idx]), device=self.device, dtype=torch.float32)
        return s, a, r, ns, d

    def update(self, updates=1):
        if len(self.buf_s) < max(CFG.RL_START_STEPS, CFG.RL_BATCH_SIZE):
            return {}
        info = {}
        for _ in range(max(1, updates)):
            s, a, r, ns, d = self._sample_batch(CFG.RL_BATCH_SIZE)

            with torch.no_grad():
                na, nlogp = self.actor.sample(ns)
                q1_t = self.q1_t(ns, na); q2_t = self.q2_t(ns, na)
                q_t  = torch.min(q1_t, q2_t) - self.alpha.detach() * nlogp
                y = r + (1.0 - d) * CFG.RL_GAMMA * q_t

            q1 = self.q1(s, a); q2 = self.q2(s, a)
            q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            self.opt_q.zero_grad(set_to_none=True)
            q_loss.backward(); self.opt_q.step()

            ca, clogp = self.actor.sample(s)
            q1a = self.q1(s, ca); q2a = self.q2(s, ca)
            actor_loss = (self.alpha.detach() * clogp - torch.min(q1a, q2a)).mean()
            self.opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward(); self.opt_actor.step()

            alpha_loss = (-(self.log_alpha * (clogp + self.target_entropy).detach())).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward(); self.alpha_opt.step()

            with torch.no_grad():
                for tp, p in zip(self.q1_t.parameters(), self.q1.parameters()):
                    tp.data.mul_(1.0 - CFG.RL_TAU).add_(CFG.RL_TAU * p.data)
                for tp, p in zip(self.q2_t.parameters(), self.q2.parameters()):
                    tp.data.mul_(1.0 - CFG.RL_TAU).add_(CFG.RL_TAU * p.data)
            info = {"q_loss": float(q_loss.detach().cpu()),
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "alpha": float(self.alpha.detach().cpu())}
        return info

# ------------------------------ CSV 日志工具 ------------------------------
def _init_history_csv(csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","stage","s",
                        "lambda_phys","lambda_freq",
                        "loss_rec","loss_phys","loss_freq","loss_total",
                        "val_psnr","val_ssim","time_sec"])

def _append_history_row(csv_path: str, row: dict):
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            row["epoch"], row["stage"], f'{row["s"]:.4f}',
            f'{row["lam_phys"]:.6f}', f'{row["lam_freq"]:.6f}',
            f'{row["l_rec"]:.6f}', f'{row["l_phys"]:.6f}', f'{row["l_freq"]:.6f}', f'{row["l_total"]:.6f}',
            f'{row["psnr_val"]:.4f}', f'{row["ssim_val"]:.6f}', f'{row["time_sec"]:.2f}'
        ])

# ------------------------------ 训练一个实验 ------------------------------
def run_one_experiment(name: str, rcfg: Dict[str, Any], tr_loader, va_loader, device):
    enable_zk  = bool(rcfg.get("enable_zernike", True))
    enable_rl  = bool(rcfg.get("enable_rl", True))
    enable_apc = bool(rcfg.get("enable_apc", True))
    epochs_total = int(rcfg.get("epochs_total", 76))

    E1, E2 = CFG.EPOCHS_WARMUP, CFG.EPOCHS_FIXED
    E3 = max(0, epochs_total - E1 - E2)
    stage3_name = "RL" if enable_rl else "Fixed"
    print(f"\n========== Run: {name} | Zernike={'ON' if enable_zk else 'OFF'} | RL={'ON' if enable_rl else 'OFF'} | APC={'ON' if enable_apc else 'OFF'} ==========")
    print(f"[schedule] epochs_total={epochs_total} (warmup={E1}, fixed={E2}, {'rl' if enable_rl else 'fixed3'}={E3})")

    # 物理前向
    phys_clean = SIMForward(use_noise=False).to(device).eval()
    phys_train = SIMForward(use_noise=True ).to(device).train()
    try:
        if hasattr(phys_clean, "set_enable_zernike"):
            phys_clean.set_enable_zernike(enable_zk); phys_train.set_enable_zernike(enable_zk)
        elif hasattr(phys_clean, "enable_zernike"):
            phys_clean.enable_zernike=enable_zk; phys_train.enable_zernike=enable_zk
        print(f"[zernike] 设置 enable_zernike = {enable_zk}")
    except Exception:
        print(f"[zernike] 设置 enable_zernike = {enable_zk}")

    # 重建网络
    net = SCUNetRecon(base_ch=48).to(device)
    if CFG.CHANNELS_LAST: net = net.to(memory_format=torch.channels_last)
    net, _ = try_compile(net)
    opt = torch.optim.Adam(net.parameters(), lr=CFG.LR_RECON, weight_decay=CFG.WEIGHT_DECAY)
    scaler = make_scaler(enabled=(device.type=="cuda" and CFG.MIXED_PRECISION))

    # SAC
    agent = SAC(CFG.RL_STATE_DIM, CFG.RL_ACTION_DIM, device) if enable_rl else None

    # EMA
    ema_rec  = EMA(0.05); ema_phys = EMA(0.05); ema_freq = EMA(0.05)

    # 路径 & CSV
    work_dir = os.path.join(CFG.WORK_DIR, name); os.makedirs(work_dir, exist_ok=True)
    best_recon_path = os.path.join(work_dir, "best_recon_ssim.pth")
    best_actor_path = os.path.join(work_dir, "best_actor_ssim.pth")
    hist_path = os.path.join(work_dir, "history.csv")
    _init_history_csv(hist_path)

    best_ssim = -1.0

    # 强度调度
    def strength(ep):
        if ep <= E1: return 0.0
        if ep <= E1 + E2: return 0.30
        t = (ep - (E1 + E2)) / max(1, E3)
        return float(min(1.0, 0.30 + 0.70 * t))

    # 训练
    for ep in range(1, epochs_total + 1):
        s = strength(ep)
        if ep <= E1: stage, lam_phys, lam_freq = "Warmup", CFG.LAMBDA_PHYS_WARMUP, CFG.LAMBDA_FREQ_WARMUP
        elif ep <= E1 + E2: stage, lam_phys, lam_freq = "Fixed", CFG.LAMBDA_PHYS_FIXED, CFG.LAMBDA_FREQ_FIXED
        else: stage, lam_phys, lam_freq = stage3_name, CFG.LAMBDA_PHYS_RL, CFG.LAMBDA_FREQ_RL
        if not enable_apc:
            lam_phys = 0.0; lam_freq = 0.0

        print(f"\n[Epoch {ep:03d}] Stage={stage} | s={s:.2f} | λ_phys={lam_phys:.2f} λ_freq={lam_freq:.2f}")
        net.train(); phys_train.train()

        sum_lr = sum_lp = sum_lf = 0.0; nb = 0; t0 = time.time()

        for gt_rgb in tr_loader:
            gt_rgb = gt_rgb.to(device, non_blocking=True)
            if CFG.CHANNELS_LAST: gt_rgb = gt_rgb.contiguous(memory_format=torch.channels_last)
            gt = gt_rgb.mean(dim=1, keepdim=True)
            B = gt.size(0)
            alpha, beta = canonical_patterns(B, device)

            # RL 控制量（默认 1）
            mod_scale_rl = torch.ones(B, device=device)
            cyc_scale_rl = torch.ones(B, device=device)

            if enable_rl and stage == "RL":
                state = np.array([ema_rec.value(), ema_phys.value(), ema_freq.value(), s], dtype=np.float32)
                a_np  = agent.act(state, eval_mode=False)
                a     = torch.tensor(a_np, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
                d_alpha, d_beta, mod_scale_rl, cyc_scale_rl = map_action_to_controls(
                    a, B, CFG.RL_DEG_LIMIT, CFG.RL_MOD_RANGE_PCT, CFG.RL_CYCLE_RANGE_PCT
                )
                alpha = (alpha + d_alpha) % (2*math.pi)
                beta  = (beta  + d_beta ) % (2*math.pi)

            with torch.no_grad():
                raw_clean = phys_clean(gt, alpha, beta, mod_scale=mod_scale_rl, cycle_scale=cyc_scale_rl)
                if CFG.CHANNELS_LAST: raw_clean = raw_clean.contiguous(memory_format=torch.channels_last)

            with torch.no_grad():
                rad = (CFG.STRESS_MAX_DEG * s) * math.pi / 180.0
                d_alpha_env = (2*torch.rand(B, 3, device=device) - 1.0) * rad
                d_beta_env  = (2*torch.rand(B, 3, device=device) - 1.0) * rad
                alpha_tr = (alpha + d_alpha_env) % (2*math.pi)
                beta_tr  = (beta  + d_beta_env)  % (2*math.pi)
                mod_jit = 1.0 + (2*torch.rand(B, device=device) - 1.0) * (CFG.STRESS_MAX_MOD_PCT * s)
                cyc_jit = 1.0 + (2*torch.rand(B, device=device) - 1.0) * (CFG.STRESS_MAX_CYCLE_PCT * s)
                mod_scale_tr = (mod_scale_rl * mod_jit).clamp_min(1e-3)
                cyc_scale_tr = (cyc_scale_rl * cyc_jit).clamp_min(1e-3)

                raw_train = phys_train(gt, alpha_tr, beta_tr, mod_scale=mod_scale_tr, cycle_scale=cyc_scale_tr)
                if CFG.CHANNELS_LAST: raw_train = raw_train.contiguous(memory_format=torch.channels_last)

                if random.random() < CFG.EXTRA_STRIPE_PROB:
                    H, W = raw_train.shape[-2], raw_train.shape[-1]
                    xs = torch.linspace(0, 2*math.pi*28, W, device=device)
                    stripe = torch.cos(xs)[None,None,None,:].expand(B, raw_train.size(1), H, W)
                    raw_train = (raw_train + 0.04*s*stripe).clamp(0.0, 1.0)
                if random.random() < CFG.EXTRA_GAUSS_PROB:
                    raw_train = (raw_train + 0.02*s*torch.randn_like(raw_train)).clamp(0.0, 1.0)
                if random.random() < CFG.EXTRA_BLEACH_PROB:
                    K = raw_train.size(1)
                    decay = torch.linspace(1.0, 1.0 - CFG.STRESS_BLEACH_MAX*s, K, device=device).view(1,K,1,1)
                    raw_train = (raw_train * decay).clamp(0.0, 1.0)

            with amp_autocast("cuda", enabled=(device.type=="cuda" and CFG.MIXED_PRECISION)):
                logits = net(raw_train)
                rec = torch.sigmoid(logits).clamp(0.0, 1.0)
                l_rec  = 0.8*charbonnier_loss(rec, gt) + 0.2*grad_l1_loss(rec, gt)
                pred_raw = phys_clean(rec, alpha, beta, mod_scale=mod_scale_rl, cycle_scale=cyc_scale_rl)
                l_phys = F.mse_loss(pred_raw, raw_clean)
                l_freq = frequency_consistency_loss(rec, gt, hi_boost=CFG.FREQ_HI_BOOST) if lam_freq>1e-9 else torch.zeros((), device=device)
                loss = l_rec + lam_phys*l_phys + lam_freq*l_freq

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if CFG.CLIP_GRAD is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), CFG.CLIP_GRAD)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if CFG.CLIP_GRAD is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), CFG.CLIP_GRAD)
                opt.step()

            nb += 1
            sum_lr += float(l_rec.detach().cpu())
            sum_lp += float(l_phys.detach().cpu())
            sum_lf += float(l_freq.detach().cpu())
            # 这里用 epoch 累计平均推进 EMA，让 RL 看到平滑指标
            ema_rec.update(sum_lr/nb); ema_phys.update(sum_lp/nb); ema_freq.update(sum_lf/nb)

            if enable_rl and stage == "RL":
                with torch.no_grad():
                    act_l2 = (a**2).mean().item() if 'a' in locals() else 0.0
                    reward = -(l_rec + lam_phys*l_phys + lam_freq*l_freq).item() - CFG.RL_ACT_L2*act_l2
                    next_state = np.array([ema_rec.value(), ema_phys.value(), ema_freq.value(), s], dtype=np.float32)
                    done = 0.0
                    agent.push(state, a_np, reward, next_state, done)
                agent.update(CFG.RL_UPDATES_PER_STEP)

        # 验证
        net.eval(); phys_clean.eval()
        sum_psnr = sum_ssim = nimg = 0
        with torch.no_grad():
            for gt_rgb in va_loader:
                gt_rgb = gt_rgb.to(device, non_blocking=True)
                if CFG.CHANNELS_LAST: gt_rgb = gt_rgb.contiguous(memory_format=torch.channels_last)
                gt = gt_rgb.mean(dim=1, keepdim=True)
                B = gt.size(0)
                alpha, beta = canonical_patterns(B, device)
                ones = torch.ones(B, device=device)
                raw = phys_clean(gt, alpha, beta, mod_scale=ones, cycle_scale=ones)
                if CFG.CHANNELS_LAST: raw = raw.contiguous(memory_format=torch.channels_last)
                rec = torch.sigmoid(net(raw)).clamp(0.0, 1.0)
                mse = F.mse_loss(rec, gt, reduction='none').mean(dim=[1,2,3])
                psnr = -10*torch.log10(mse + 1e-8)
                ssim = ssim_batch(rec, gt)  # 来自 utils.py  :contentReference[oaicite:3]{index=3}
                sum_psnr += psnr.sum().item(); sum_ssim += ssim.sum().item(); nimg += B

        psnr_val = sum_psnr / max(1, nimg)
        ssim_val = sum_ssim / max(1, nimg)
        dt = time.time() - t0; nb = max(1, nb)
        l_rec_ep  = sum_lr/nb; l_phys_ep = sum_lp/nb; l_freq_ep = sum_lf/nb
        l_total_ep = l_rec_ep + lam_phys*l_phys_ep + lam_freq*l_freq_ep
        print(f"Epoch {ep:03d} [{stage}] Val: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f} | "
              f"L(rec/phys/freq)={l_rec_ep:.4f}/{l_phys_ep:.4f}/{l_freq_ep:.4f} | {dt:.1f}s")

        # 写 CSV
        _append_history_row(hist_path, dict(
            epoch=ep, stage=stage, s=s,
            lam_phys=lam_phys, lam_freq=lam_freq,
            l_rec=l_rec_ep, l_phys=l_phys_ep, l_freq=l_freq_ep, l_total=l_total_ep,
            psnr_val=psnr_val, ssim_val=ssim_val, time_sec=dt
        ))

        # 保存最优
        if ssim_val > best_ssim:
            best_ssim = ssim_val
            torch.save({"model": net.state_dict()}, best_recon_path)
            if enable_rl:
                torch.save({"actor": agent.actor.state_dict()}, best_actor_path)
            print(f"★ 保存更优模型({name}): SSIM={best_ssim:.4f} -> {best_recon_path}"
                  + (f"; Actor -> {best_actor_path}" if enable_rl else ""))

    print(f"\n完成 {name} 训练！最佳 SSIM={best_ssim:.4f} | 模型: {best_recon_path}")
    return best_ssim, best_recon_path, (best_actor_path if enable_rl else None)

# ------------------------------ 主函数 ------------------------------
def main():
    os.makedirs(CFG.WORK_DIR, exist_ok=True)
    set_seed(CFG.SEED)
    device = torch.device(CFG.DEVICE)
    print(f"Device: {device.type} | AMP: {'on' if CFG.MIXED_PRECISION and device.type=='cuda' else 'off'}")
    print(f"[data] Train dir: {CFG.TRAIN_DIR}")

    loaders = build_dataloaders(
        gt_dir=CFG.TRAIN_DIR, work_dir=CFG.WORK_DIR,
        batch_size=CFG.BATCH_SIZE, patch_size=CFG.PATCH_SIZE,
        num_workers=CFG.NUM_WORKERS, seed=CFG.SEED
    )
    if isinstance(loaders, (list, tuple)) and len(loaders) >= 2:
        tr_loader = loaders[0]; va_loader = loaders[1]
    else:
        raise RuntimeError("build_dataloaders 返回格式不符合预期")
    print_data_info("Train", tr_loader)
    print_data_info("Val  ", va_loader)

    results = {}
    for name, rcfg in RUNS.items():
        best_ssim, mp, ap = run_one_experiment(name, rcfg, tr_loader, va_loader, device)
        results[name] = dict(best_ssim=best_ssim, model=mp, actor=ap)

    print("\n=== All runs finished ===")
    for name, r in results.items():
        line = f"{name}: best SSIM={r['best_ssim']:.4f} | model={r['model']}"
        if r['actor'] is not None: line += f" | actor={r['actor']}"
        print(line)

if __name__ == "__main__":
    main()
