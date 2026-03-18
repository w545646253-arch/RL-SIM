# RL-SIM reviewer package

This package is a lightweight **reviewer reproducibility package** for the manuscript
**Physics-guided reinforcement learning for structured illumination microscopy**.

It is designed to support:

- inference with pretrained checkpoints;
- reproduction of the frame-number comparison and the paired-FRC-based Fig. 3(g) summary in the revised manuscript;
- generation of the GT-referenced FRC-style cross-method summary used for Supplementary Fig. S3 and Supplementary Table S5;
- export of source-data spreadsheets for redrawing the revised Supplementary Fig. S3 curves and boxplots.

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
   └─ supp_fig_s3_frc_vs_gt/
      ├─ baseline_paired_frc_comparison.py
      └─ supp_table_s5_frc_vs_gt_subset.xlsx
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

### `results_example/supp_fig_s3_frc_vs_gt`

This folder contains the source-data workbook and the companion script used to generate the revised GT-referenced FRC-style cross-method summary for Supplementary Fig. S3 and Supplementary Table S5.

- `baseline_paired_frc_comparison.py` generates the GT-referenced FRC-style curves, per-image resolution statistics, and the source-data workbook used for the revised supplementary frequency-domain summary.
- `supp_table_s5_frc_vs_gt_subset.xlsx` contains the wide-format curve data, per-image statistics, and redraw-ready source data for Supplementary Fig. S3.

The folder name `supp_fig_s3_frc_vs_gt` is retained for continuity with earlier reviewer-package versions. In the revised manuscript, however, the contents of this folder correspond to the GT-referenced FRC-style baseline summary used for Supplementary Fig. S3 / Table S5 rather than only the earlier paired-FRC example.


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
