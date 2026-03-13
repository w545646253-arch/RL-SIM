# -*- coding: utf-8 -*-
"""
test_369.py  (DATASET mode + optional fixed split, reviewer-facing, input-channel adaptive at INPUT only)

Reproduce Fig.1 (3/6/9 frames) on a held-out set.
- Nested angle sets (2D-SIM style):
  K3 : angles=[0]
  K6 : angles=[0,60]
  K9 : angles=[0,60,120]
- GT normalization: minmax (per-image)
- crop: center 256×256
- forward: true multi-angle stripe forward (no rotation interpolation)
- split: (optional) uses splits_fixed/val_list(.txt) if provided; otherwise generates a deterministic split and stores it.

Folder assumptions:
supplyment/
  model_code/
    model_recon_scunet.py, utils.py
  splits_fixed/               (optional)
    val_list.txt (or val_list)
  test_image/                 (demo set)
  work_all/
    APCRL_3/best_ssim.pth
    APCRL_6/best_ssim.pth
    APCRL_9/best_ssim.pth
  test/
    test_369.py

Run:
python test/test_369.py
"""

import os, sys, math, csv, random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tifffile as tiff
except Exception:
    tiff = None
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
MODEL_CODE = os.path.join(REPO_ROOT, "model_code")
if os.path.isdir(MODEL_CODE) and MODEL_CODE not in sys.path:
    sys.path.insert(0, MODEL_CODE)

from model_recon_scunet import SCUNetRecon
from utils import psnr_batch, ssim_batch


class CFG:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 2024

    GT_DIR = os.path.join(REPO_ROOT, "test_image")

    SPLIT_DIR = os.path.join(REPO_ROOT, "splits_fixed")
    VAL_LIST_CAND = [
        os.path.join(SPLIT_DIR, "val_list.txt"),
        os.path.join(SPLIT_DIR, "val_list"),
    ]
    SPLIT_RATIO_TRAIN = 0.85

    PATCH_SIZE = 256
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    CHANNELS_LAST = True
    PAD_MULT = 16

    BASE_CYCLES = 24.0
    OTF_SIGMA = 0.18

    WORK_ALL = os.path.join(REPO_ROOT, "work_all")
    CKPT_K3 = os.path.join(WORK_ALL, "APCRL_3", "best_ssim.pth")
    CKPT_K6 = os.path.join(WORK_ALL, "APCRL_6", "best_ssim.pth")
    CKPT_K9 = os.path.join(WORK_ALL, "APCRL_9", "best_ssim.pth")

    OUT_DIR = os.path.join(REPO_ROOT, "outputs_369")
    OUT_CSV = os.path.join(OUT_DIR, "fig1_k369_fixedsplit.csv")
    GEN_SPLIT_DIR = os.path.join(OUT_DIR, "splits")
    GEN_VAL_LIST = os.path.join(GEN_SPLIT_DIR, "val_list.txt")


EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"GT_DIR not found: {root}")
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(EXTS):
                paths.append(os.path.join(dp, fn))
    paths = sorted(list(set(paths)))
    if not paths:
        raise FileNotFoundError(f"No images found in: {root}")
    return paths


def imread(path: str) -> np.ndarray:
    if (tiff is not None) and path.lower().endswith((".tif", ".tiff")):
        return np.asarray(tiff.imread(path))
    return np.asarray(Image.open(path))


def norm01_minmax(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    if x.ndim == 3:
        x = x[..., :3].mean(axis=2)
    vmax = float(x.max()) if x.size > 0 else 1.0
    vmax = max(vmax, 1e-6)
    x = x / vmax
    return np.clip(x, 0.0, 1.0)


def to_tensor_chw(x01_hw: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x01_hw[None, ...]).float()


def center_crop_chw(x: torch.Tensor, patch: Optional[int]) -> torch.Tensor:
    if patch is None:
        return x
    C, H, W = x.shape
    if H <= patch or W <= patch:
        return x
    top = (H - patch) // 2
    left = (W - patch) // 2
    return x[:, top:top + patch, left:left + patch]


def pad_to_mult(x_bchw: torch.Tensor, mult: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    _, _, H, W = x_bchw.shape
    tH = int(math.ceil(H / mult) * mult)
    tW = int(math.ceil(W / mult) * mult)
    ph = max(0, tH - H)
    pw = max(0, tW - W)
    top = ph // 2
    bottom = ph - top
    left = pw // 2
    right = pw - left
    if ph > 0 or pw > 0:
        x_pad = F.pad(x_bchw, (left, right, top, bottom), mode="reflect")
    else:
        x_pad = x_bchw
    return x_pad, (top, bottom, left, right)


def maybe_channels_last(x: torch.Tensor) -> torch.Tensor:
    if not CFG.CHANNELS_LAST:
        return x.contiguous()
    if x.dim() == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x.contiguous()


def _pick_existing_val_list() -> Optional[str]:
    for p in CFG.VAL_LIST_CAND:
        if os.path.isfile(p):
            return p
    return None


def _resolve_listfile_paths(list_path: str, base_dir: str) -> List[str]:
    with open(list_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    out = []
    for s in lines:
        if os.path.isabs(s) and os.path.isfile(s):
            out.append(s)
        else:
            p = os.path.join(base_dir, s)
            if os.path.isfile(p):
                out.append(p)
    return out


def build_val_paths() -> Tuple[List[str], str]:
    val_file = _pick_existing_val_list()
    if val_file is not None:
        paths = _resolve_listfile_paths(val_file, CFG.GT_DIR)
        if len(paths) >= 1:
            return paths, f"fixed:{os.path.basename(val_file)}"

    if os.path.isfile(CFG.GEN_VAL_LIST):
        paths = _resolve_listfile_paths(CFG.GEN_VAL_LIST, CFG.GT_DIR)
        if len(paths) >= 1:
            return paths, f"generated:{CFG.GEN_VAL_LIST}"

    all_paths = list_images(CFG.GT_DIR)
    idx = list(range(len(all_paths)))
    random.shuffle(idx)
    n_tr = int(CFG.SPLIT_RATIO_TRAIN * len(idx))
    val_paths = [all_paths[i] for i in idx[n_tr:]]

    ensure_dir(CFG.GEN_SPLIT_DIR)
    with open(CFG.GEN_VAL_LIST, "w", encoding="utf-8") as f:
        for p in val_paths:
            rel = os.path.relpath(p, CFG.GT_DIR).replace("\\", "/")
            f.write(rel + "\n")

    return val_paths, f"generated:{CFG.GEN_VAL_LIST}"


class ValSet(torch.utils.data.Dataset):
    def __init__(self, paths: List[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = imread(self.paths[idx])
        x01 = norm01_minmax(arr)
        ten = to_tensor_chw(x01)
        ten = center_crop_chw(ten, CFG.PATCH_SIZE)
        return ten


def build_val_loader(val_paths: List[str]) -> torch.utils.data.DataLoader:
    ds = ValSet(val_paths)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False
    )
    return loader


def _replace_conv(parent: nn.Module, name: str, conv: nn.Conv2d, new_in_ch: int):
    new_conv = nn.Conv2d(
        new_in_ch,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)
    parent._modules[name] = new_conv


def adapt_input_layers_only(net: nn.Module, target_in_ch: int) -> nn.Module:
    # e1.conv1.0
    if hasattr(net, "e1") and hasattr(net.e1, "conv1"):
        conv1 = net.e1.conv1
        if isinstance(conv1, nn.Sequential) and len(conv1) > 0 and isinstance(conv1[0], nn.Conv2d):
            if conv1[0].in_channels != target_in_ch:
                _replace_conv(conv1, "0", conv1[0], target_in_ch)

    # head.0
    if hasattr(net, "head"):
        h = getattr(net, "head")
        if isinstance(h, nn.Sequential) and len(h) > 0 and isinstance(h[0], nn.Conv2d):
            if h[0].in_channels != target_in_ch:
                _replace_conv(h, "0", h[0], target_in_ch)

    return net


class SIMForwardAngles(nn.Module):
    def __init__(self, angles_deg: List[float], base_cycles: float, otf_sigma: float):
        super().__init__()
        self.angles_deg = [float(a) for a in angles_deg]
        self.base_cycles = float(base_cycles)
        self.otf_sigma = float(otf_sigma)

    def _build_otf(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        ys = torch.linspace(-(H - 1) / 2, (H - 1) / 2, steps=H, device=device, dtype=torch.float32)
        xs = torch.linspace(-(W - 1) / 2, (W - 1) / 2, steps=W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        rr2 = (xx ** 2 + yy ** 2)
        rr2 = rr2 / (rr2.max() + 1e-6)
        A = torch.exp(-rr2 / (2.0 * (self.otf_sigma ** 2))).contiguous()
        return A

    def _apply_otf(self, img: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        X = torch.fft.fft2(img.float(), dim=(-2, -1), norm="ortho")
        X = torch.fft.fftshift(X, dim=(-2, -1))
        Y = X * A
        Y = torch.fft.ifftshift(Y, dim=(-2, -1))
        y = torch.fft.ifft2(Y, dim=(-2, -1), norm="ortho").real
        return y.to(img.dtype)

    def forward(self, gt: torch.Tensor) -> torch.Tensor:
        B, _, H, W = gt.shape
        device = gt.device
        A = self._build_otf(H, W, device)

        xs = torch.linspace(0, float(W - 1), steps=W, device=device).view(1, 1, 1, W)
        ys = torch.linspace(0, float(H - 1), steps=H, device=device).view(1, 1, H, 1)

        k_mag = 2.0 * math.pi * float(CFG.BASE_CYCLES) / float(W)
        phases = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]

        frames = []
        for phi in phases:
            for ang in self.angles_deg:
                th = math.radians(ang)
                kx = k_mag * math.cos(th)
                ky = k_mag * math.sin(th)
                patt = 1.0 + torch.cos(kx * xs + ky * ys + phi)
                img = (gt * patt).clamp(0.0, 1.0)
                img = self._apply_otf(img, A)
                frames.append(img)
        return torch.cat(frames, dim=1)


def load_scunet(ckpt_path: str, in_ch: int, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    net = SCUNetRecon(base_ch=48).to(device)
    if in_ch != 6:
        net = adapt_input_layers_only(net, in_ch)
    if CFG.CHANNELS_LAST:
        net = net.to(memory_format=torch.channels_last)

    net.load_state_dict(state, strict=False)
    net.eval()
    return net


@torch.no_grad()
def eval_one(net: nn.Module, forward: SIMForwardAngles, loader) -> Tuple[float, float]:
    device = next(net.parameters()).device
    ps_all, ss_all = [], []
    for gt_chw in loader:
        gt = gt_chw.to(device, non_blocking=True).unsqueeze(1) if gt_chw.dim()==3 else gt_chw.to(device, non_blocking=True)
        gt = maybe_channels_last(gt)
        gt_pad, _ = pad_to_mult(gt, CFG.PAD_MULT)

        raw = maybe_channels_last(forward(gt_pad))
        rec = torch.sigmoid(net(raw)).clamp(0.0, 1.0)

        ps_all.append(psnr_batch(rec, gt_pad))
        ss_all.append(ssim_batch(rec, gt_pad))
    return torch.cat(ps_all).mean().item(), torch.cat(ss_all).mean().item()


def main():
    set_seed(CFG.SEED)
    ensure_dir(CFG.OUT_DIR)

    val_paths, src = build_val_paths()
    print(f"[split] {src} | Val={len(val_paths)}")
    loader = build_val_loader(val_paths)
    print(f"[data] iters={len(loader)} | patch={CFG.PATCH_SIZE}")

    device = torch.device(CFG.DEVICE)

    net3 = load_scunet(CFG.CKPT_K3, 3, device)
    net6 = load_scunet(CFG.CKPT_K6, 6, device)
    net9 = load_scunet(CFG.CKPT_K9, 9, device)

    f3 = SIMForwardAngles([0.0], CFG.BASE_CYCLES, CFG.OTF_SIGMA).to(device).eval()
    f6 = SIMForwardAngles([0.0, 60.0], CFG.BASE_CYCLES, CFG.OTF_SIGMA).to(device).eval()
    f9 = SIMForwardAngles([0.0, 60.0, 120.0], CFG.BASE_CYCLES, CFG.OTF_SIGMA).to(device).eval()

    p3, s3 = eval_one(net3, f3, loader)
    p6, s6 = eval_one(net6, f6, loader)
    p9, s9 = eval_one(net9, f9, loader)

    print(f"[Fig1] K3  PSNR={p3:.4f} SSIM={s3:.6f}")
    print(f"[Fig1] K6  PSNR={p6:.4f} SSIM={s6:.6f}")
    print(f"[Fig1] K9  PSNR={p9:.4f} SSIM={s9:.6f}")

    with open(CFG.OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "PSNR", "SSIM"])
        w.writerow([3, f"{p3:.4f}", f"{s3:.6f}"])
        w.writerow([6, f"{p6:.4f}", f"{s6:.6f}"])
        w.writerow([9, f"{p9:.4f}", f"{s9:.6f}"])

    print("[save]", CFG.OUT_CSV)


if __name__ == "__main__":
    main()
