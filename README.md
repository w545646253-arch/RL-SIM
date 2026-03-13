# RL-SIM reviewer package

This package is a lightweight **reviewer reproducibility package** for the manuscript
**Physics-guided reinforcement learning for structured illumination microscopy**.

It is designed to support:
- inference with pretrained checkpoints,
- reproduction of the frame-number comparison,
- reproduction of the paired-FRC-based Fig. 1(g) summary,
- generation of representative paired-FRC curves for the Supplementary Information.

## Directory structure

```text
RL-SIM/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ CITATION.cff
├─ model_code/
├─ scripts/
├─ checkpoints/
├─ minimal_example_data/
└─ results_example/
```

## What must be added locally before use

This archive contains the code and templates that were available in the chat materials.
Before uploading to GitHub or running the package end-to-end, you should additionally copy:

1. the three pretrained checkpoints into `checkpoints/`;
2. the six representative `.tif` images into `minimal_example_data/`.

Expected checkpoint filenames:
- `APCRL_Zoff_K3_best.pth`
- `APCRL_Zoff_K6_best.pth`
- `APCRL_Zoff_K9_best.pth`

Expected minimal example data:
- 2 microtubule images
- 2 CCP images
- 2 ER images

## Reproducing Fig. 1(g)

Run from the repository root:

```bash
python scripts/paired_frc_fig1g.py
```

This generates:
- `results_example/fig1g_paired_frc/paired_frc_results.xlsx`
- `results_example/fig1g_paired_frc/paired_frc_per_sample.csv`
- `results_example/fig1g_paired_frc/paired_frc_source_curves.csv`
- `results_example/fig1g_paired_frc/fig1g_frc_resolution_halfbit_nm.png`
- `results_example/fig1g_paired_frc/supp_fig_s1_representative_paired_frc.png`

## Scope note

The included `scripts/train_reference.py` is provided only as a method reference.
The present reviewer package should be interpreted as a **minimal figure-level and inference-level reproducibility package**, not as a full retraining release.
