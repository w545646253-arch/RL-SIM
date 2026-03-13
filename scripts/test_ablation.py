# -*- coding: utf-8 -*-
"""
test_ablation.py  (DATASET-only, reviewer-facing)

Ablation study reproduction (APCRL vs APCOnly vs RLOnly), DATASET mode only:
- Fig.5(c): Gaussian noise sweep (sigma = 0.00..0.10 step 0.02)
- Fig.5(e): vertical sinusoidal stripe sweep (A = 0.00..0.10 step 0.02)
- Table 2: phase detuning (uniform offset applied to ALL patterns): {0,5,10,15,20,30} deg
- Table 3: photobleaching ratio: {0.1,0.2,0.3,0.4,0.5}

Locked protocol:
- GT normalization: minmax (per-image)
- Zernike: OFF (consistent across ablations)
- raw6 order: native interleaved output of physics_v2.SIMForward
- same forward + perturbations for all ablation models

Folder assumptions:
supplyment/
  model_code/
    physics_v2.py, model_recon_scunet.py, utils.py
  work_all/
    APCRL/best_recon_ssim.pth
    APCOnly/best_recon_ssim.pth
    RLOnly/best_recon_ssim.pth
  splits_fixed/
    val_list(.txt)  (optional; absolute or relative paths)
  test/
    test_ablation.py  (this file)

Run:
python test/test_ablation.py
"""

import os, sys, math, csv, random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import tifffile as tiff
except Exception:
    tiff = None
from PIL import Image

# ---------------- repo root & import path ----------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
MODEL_CODE = os.path.join(REPO_ROOT, "model_code")
if os.path.isdir(MODEL_CODE) and MODEL_CODE not in sys.path:
    sys.path.insert(0, MODEL_CODE)

from physics_v2 import SIMForward
from model_recon_scunet import SCUNetRecon
from utils import psnr_batch, ssim_batch


# ============================== CONFIG ==============================
class CFG:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 2024

    # Dataset root (for quick sanity check you can keep test_image;
    # for full reproduction set to your full GT dataset directory)
    GT_DIR = os.path.join(REPO_ROOT, "test_image")

    # outputs
    OUT_ROOT = os.path.join(REPO_ROOT, "outputs_ablation")

    # optional fixed split file (preferred if valid)
    SPLIT_DIR = os.path.join(REPO_ROOT, "splits_fixed")
    VAL_LIST_CAND = [
        os.path.join(SPLIT_DIR, "val_list.txt"),
        os.path.join(SPLIT_DIR, "val_list"),
    ]

    # fallback split ratio (only used when val_list is missing/invalid)
    SPLIT_RATIO_TRAIN = 0.85

    # dataloader
    PATCH_SIZE = 256
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    CHANNELS_LAST = True

    # forward
    ENABLE_ZERNIKE = False
    BASE_CYCLES = 28.0
    OTF_SIGMA = 0.18
    PAD_MULT = 16

    # checkpoints (K=6 ablation)
    WORK_ALL = os.path.join(REPO_ROOT, "work_all")
    CKPT_APCRL   = os.path.join(WORK_ALL, "APCRL",   "best_recon_ssim.pth")   # full
    CKPT_APCONLY = os.path.join(WORK_ALL, "APCOnly", "best_recon_ssim.pth")
    CKPT_RLONLY  = os.path.join(WORK_ALL, "RLOnly",  "best_recon_ssim.pth")

    # sweep levels
    NOISE_SIGMAS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
    STRIPE_AMPS  = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
    PHASE_OFFSETS_DEG = [0, 5, 10, 15, 20, 30]
    BLEACH_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]
    STRIPE_CYCLES = 28


# ============================== Utilities ==============================
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


def norm01(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    if x.ndim == 3:
        x = x[..., :3].mean(axis=2)
    vmax = float(x.max()) if x.size > 0 else 1.0
    vmax = max(vmax, 1e-6)
    x = x / vmax
    return np.clip(x, 0.0, 1.0)


def to_tensor_chw(x01_hw: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x01_hw[None, ...]).float()  # [1,H,W]


def center_crop_chw(x: torch.Tensor, patch: Optional[int]) -> torch.Tensor:
    if patch is None:
        return x
    C, H, W = x.shape
    if H <= patch or W <= patch:
        return x
    top = (H - patch) // 2
    left = (W - patch) // 2
    return x[:, top:top+patch, left:left+patch]


def pad_to_mult(x_bchw: torch.Tensor, mult: int) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
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
    # only 4D can use channels_last
    if not CFG.CHANNELS_LAST:
        return x.contiguous()
    if x.dim() == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x.contiguous()


def add_gaussian_noise(raw: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return raw
    return (raw + float(sigma) * torch.randn_like(raw)).clamp(0.0, 1.0)


def add_vertical_stripe(raw: torch.Tensor, amp: float, cycles: int) -> torch.Tensor:
    if amp <= 0:
        return raw
    if raw.dim() != 4:
        raise RuntimeError(f"stripe expects 4D raw [B,K,H,W], got {tuple(raw.shape)}")
    B, K, H, W = raw.shape
    xs = torch.linspace(0, 2*math.pi*cycles, W, device=raw.device, dtype=raw.dtype)
    stripe = torch.sin(xs)[None, None, None, :].expand(B, K, H, W)
    return (raw + float(amp) * stripe).clamp(0.0, 1.0)


def apply_bleach(raw: torch.Tensor, ratio: float) -> torch.Tensor:
    if ratio <= 0:
        return raw
    if raw.dim() != 4:
        raise RuntimeError(f"bleach expects 4D raw [B,K,H,W], got {tuple(raw.shape)}")
    B, K, H, W = raw.shape
    decay = torch.linspace(1.0, 1.0 - float(ratio), K, device=raw.device, dtype=raw.dtype).view(1, K, 1, 1)
    return (raw * decay).clamp(0.0, 1.0)


# ============================== Split handling ==============================
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
    """
    Returns (val_paths, source_tag)
    Priority:
    1) Use splits_fixed/val_list(.txt) if it resolves to >=1 file
    2) Else, generate a new split from GT_DIR and write to OUT_ROOT/splits/val_list.txt (relative)
    """
    val_file = _pick_existing_val_list()
    if val_file is not None:
        paths = _resolve_listfile_paths(val_file, CFG.GT_DIR)
        if len(paths) >= 1:
            return paths, f"fixed:{os.path.basename(val_file)}"

    # fallback split
    all_paths = list_images(CFG.GT_DIR)
    idx = list(range(len(all_paths)))
    random.shuffle(idx)
    n_tr = int(CFG.SPLIT_RATIO_TRAIN * len(idx))
    val_paths = [all_paths[i] for i in idx[n_tr:]]

    split_dir = os.path.join(CFG.OUT_ROOT, "splits")
    ensure_dir(split_dir)
    out_val = os.path.join(split_dir, "val_list.txt")
    with open(out_val, "w", encoding="utf-8") as f:
        for p in val_paths:
            rel = os.path.relpath(p, CFG.GT_DIR).replace("\\", "/")
            f.write(rel + "\n")

    return val_paths, f"generated:{out_val}"


# ============================== Dataset / Loader ==============================
class ValSet(torch.utils.data.Dataset):
    def __init__(self, paths: List[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = imread(self.paths[idx])
        x01 = norm01(arr)
        ten = to_tensor_chw(x01)  # [1,H,W]
        ten = center_crop_chw(ten, CFG.PATCH_SIZE)
        return ten


def build_val_loader(val_paths: List[str]) -> torch.utils.data.DataLoader:
    ds = ValSet(val_paths)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False
    )
    print(f"[data] Val={len(val_paths)} | iters={len(loader)} | patch={CFG.PATCH_SIZE}")
    return loader


# ============================== Model loading ==============================
def load_k6_net(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    net = SCUNetRecon(base_ch=48).to(device)
    if CFG.CHANNELS_LAST:
        net = net.to(memory_format=torch.channels_last)
    net.load_state_dict(state, strict=False)
    net.eval()
    return net


@torch.no_grad()
def eval_model_on_raw(net: torch.nn.Module, raw: torch.Tensor, gt: torch.Tensor):
    if raw.dim() != 4:
        raise RuntimeError(f"Network expects 4D input [B,K,H,W], got {tuple(raw.shape)}")
    rec = torch.sigmoid(net(raw)).clamp(0.0, 1.0)
    return psnr_batch(rec, gt), ssim_batch(rec, gt)


# ============================== Main ablation suite ==============================
@torch.no_grad()
def run_ablation_suite():
    device = torch.device(CFG.DEVICE)

    # fixed dataset loader
    val_paths, src = build_val_paths()
    print("[split]", src)
    loader = build_val_loader(val_paths)

    # forward
    phys = SIMForward(
        use_noise=False,
        enable_zernike=bool(CFG.ENABLE_ZERNIKE),
        base_cycles=float(CFG.BASE_CYCLES),
        otf_sigma=float(CFG.OTF_SIGMA),
    ).to(device).eval()

    # models
    netA = load_k6_net(CFG.CKPT_APCRL, device)
    netB = load_k6_net(CFG.CKPT_APCONLY, device)
    netC = load_k6_net(CFG.CKPT_RLONLY, device)

    ensure_dir(CFG.OUT_ROOT)

    def canonical_patterns(B: int, device: torch.device):
        phi = torch.tensor([0.0, 2*math.pi/3, 4*math.pi/3], device=device, dtype=torch.float32)
        alpha = phi.unsqueeze(0).expand(B, -1).contiguous()
        beta  = (phi + math.pi/2).unsqueeze(0).expand(B, -1).contiguous()
        return alpha, beta

    # ---------- baseline ----------
    base_csv = os.path.join(CFG.OUT_ROOT, "ablation_baseline.csv")
    with open(base_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "PSNR_APCRL", "SSIM_APCRL", "PSNR_APCOnly", "SSIM_APCOnly", "PSNR_RLOnly", "SSIM_RLOnly"])

        psA=[]; ssA=[]; psB=[]; ssB=[]; psC=[]; ssC=[]
        for gt_chw in loader:
            gt = gt_chw.to(device, non_blocking=True)  # [B,1,H,W]
            gt = maybe_channels_last(gt)

            gt_pad, _ = pad_to_mult(gt, CFG.PAD_MULT)

            B = gt_pad.size(0)
            alpha, beta = canonical_patterns(B, device)
            ones = torch.ones(B, device=device, dtype=torch.float32)

            raw6 = phys(gt_pad, alpha, beta, mod_scale=ones, cycle_scale=ones)  # [B,6,H,W]
            raw6 = maybe_channels_last(raw6)

            p,s = eval_model_on_raw(netA, raw6, gt_pad); psA.append(p); ssA.append(s)
            p,s = eval_model_on_raw(netB, raw6, gt_pad); psB.append(p); ssB.append(s)
            p,s = eval_model_on_raw(netC, raw6, gt_pad); psC.append(p); ssC.append(s)

        PS_A = torch.cat(psA).mean().item(); SS_A = torch.cat(ssA).mean().item()
        PS_B = torch.cat(psB).mean().item(); SS_B = torch.cat(ssB).mean().item()
        PS_C = torch.cat(psC).mean().item(); SS_C = torch.cat(ssC).mean().item()

        w.writerow(["baseline", f"{PS_A:.4f}", f"{SS_A:.6f}", f"{PS_B:.4f}", f"{SS_B:.6f}", f"{PS_C:.4f}", f"{SS_C:.6f}"])
        print(f"[baseline] SSIM A/B/C = {SS_A:.4f}/{SS_B:.4f}/{SS_C:.4f}")

    # helper: sweep
    def write_sweep(csv_name: str, header: List[str], levels: List[float], apply_fn):
        path_csv = os.path.join(CFG.OUT_ROOT, csv_name)
        with open(path_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)

            for lv in levels:
                # make stochastic noise deterministic per level (optional but helpful)
                seed_lv = CFG.SEED + int(round(float(lv) * 1000))
                torch.manual_seed(seed_lv)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_lv)

                psA=[]; ssA=[]; psB=[]; ssB=[]; psC=[]; ssC=[]
                for gt_chw in loader:
                    gt = gt_chw.to(device, non_blocking=True)
                    gt = maybe_channels_last(gt)
                    gt_pad, _ = pad_to_mult(gt, CFG.PAD_MULT)

                    B = gt_pad.size(0)
                    alpha, beta = canonical_patterns(B, device)
                    ones = torch.ones(B, device=device, dtype=torch.float32)

                    raw6 = phys(gt_pad, alpha, beta, mod_scale=ones, cycle_scale=ones)
                    raw6 = apply_fn(raw6, lv)
                    raw6 = maybe_channels_last(raw6)

                    p,s = eval_model_on_raw(netA, raw6, gt_pad); psA.append(p); ssA.append(s)
                    p,s = eval_model_on_raw(netB, raw6, gt_pad); psB.append(p); ssB.append(s)
                    p,s = eval_model_on_raw(netC, raw6, gt_pad); psC.append(p); ssC.append(s)

                PS_A = torch.cat(psA).mean().item(); SS_A = torch.cat(ssA).mean().item()
                PS_B = torch.cat(psB).mean().item(); SS_B = torch.cat(ssB).mean().item()
                PS_C = torch.cat(psC).mean().item(); SS_C = torch.cat(ssC).mean().item()

                w.writerow([f"{lv:.2f}",
                            f"{PS_A:.4f}", f"{SS_A:.6f}",
                            f"{PS_B:.4f}", f"{SS_B:.6f}",
                            f"{PS_C:.4f}", f"{SS_C:.6f}"])
                print(f"[{csv_name.replace('.csv','')}] lv={lv} | SSIM A/B/C = {SS_A:.4f}/{SS_B:.4f}/{SS_C:.4f}")

    # Fig5 noise
    write_sweep(
        "fig5_noise_sweep.csv",
        ["noise_sigma", "PSNR_APCRL", "SSIM_APCRL", "PSNR_APCOnly", "SSIM_APCOnly", "PSNR_RLOnly", "SSIM_RLOnly"],
        CFG.NOISE_SIGMAS,
        lambda raw, sigma: add_gaussian_noise(raw, float(sigma))
    )

    # Fig5 stripe
    write_sweep(
        "fig5_stripe_sweep.csv",
        ["stripe_amp", "PSNR_APCRL", "SSIM_APCRL", "PSNR_APCOnly", "SSIM_APCOnly", "PSNR_RLOnly", "SSIM_RLOnly"],
        CFG.STRIPE_AMPS,
        lambda raw, amp: add_vertical_stripe(raw, float(amp), CFG.STRIPE_CYCLES)
    )

    # Table2 phase offsets
    table2 = os.path.join(CFG.OUT_ROOT, "table2_phase_offsets.csv")
    with open(table2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase_offset_deg", "PSNR_APCRL", "SSIM_APCRL", "PSNR_APCOnly", "SSIM_APCOnly", "PSNR_RLOnly", "SSIM_RLOnly"])

        for deg in CFG.PHASE_OFFSETS_DEG:
            off = float(deg) * math.pi / 180.0
            psA=[]; ssA=[]; psB=[]; ssB=[]; psC=[]; ssC=[]
            for gt_chw in loader:
                gt = gt_chw.to(device, non_blocking=True)
                gt = maybe_channels_last(gt)
                gt_pad, _ = pad_to_mult(gt, CFG.PAD_MULT)

                B = gt_pad.size(0)
                base = torch.tensor([0.0, 2*math.pi/3, 4*math.pi/3], device=device).view(1,3).expand(B,3)
                alpha = (base + off) % (2*math.pi)
                beta  = ((base + math.pi/2) + off) % (2*math.pi)
                ones = torch.ones(B, device=device, dtype=torch.float32)

                raw6 = phys(gt_pad, alpha, beta, mod_scale=ones, cycle_scale=ones)
                raw6 = maybe_channels_last(raw6)

                p,s = eval_model_on_raw(netA, raw6, gt_pad); psA.append(p); ssA.append(s)
                p,s = eval_model_on_raw(netB, raw6, gt_pad); psB.append(p); ssB.append(s)
                p,s = eval_model_on_raw(netC, raw6, gt_pad); psC.append(p); ssC.append(s)

            PS_A = torch.cat(psA).mean().item(); SS_A = torch.cat(ssA).mean().item()
            PS_B = torch.cat(psB).mean().item(); SS_B = torch.cat(ssB).mean().item()
            PS_C = torch.cat(psC).mean().item(); SS_C = torch.cat(ssC).mean().item()

            w.writerow([deg, f"{PS_A:.4f}", f"{SS_A:.6f}", f"{PS_B:.4f}", f"{SS_B:.6f}", f"{PS_C:.4f}", f"{SS_C:.6f}"])
            print(f"[Table2] deg={deg:>2} | SSIM A/B/C = {SS_A:.4f}/{SS_B:.4f}/{SS_C:.4f}")

    # Table3 bleaching
    write_sweep(
        "table3_bleaching.csv",
        ["bleach_ratio", "PSNR_APCRL", "SSIM_APCRL", "PSNR_APCOnly", "SSIM_APCOnly", "PSNR_RLOnly", "SSIM_RLOnly"],
        CFG.BLEACH_RATIOS,
        lambda raw, ratio: apply_bleach(raw, float(ratio))
    )

    print(f"[DONE] outputs saved to: {CFG.OUT_ROOT}")


def main():
    set_seed(CFG.SEED)
    ensure_dir(CFG.OUT_ROOT)
    print(f"[env] mode=DATASET device={CFG.DEVICE} norm=minmax zernike={'ON' if CFG.ENABLE_ZERNIKE else 'OFF'} patch={CFG.PATCH_SIZE}")
    run_ablation_suite()


if __name__ == "__main__":
    main()
