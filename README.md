---
language:
  - en
license: other
license_name: research-use
tags:
  - medical-imaging
  - prostate-cancer
  - segmentation
  - knowledge-distillation
  - reinforcement-learning
  - mri
  - bi-parametric-mri
  - nnunet
  - multi-teacher
  - federated-learning
datasets:
  - pi-cai
  - prostate-mri-us-biopsy
  - stanford-prostate-mri
pipeline_tag: image-segmentation
library_name: pytorch
metrics:
  - roc_auc
  - pr_auc
model-index:
  - name: multiGRAFT
    results:
      - task:
          type: image-segmentation
          name: Clinically Significant Prostate Cancer Detection
        dataset:
          name: PI-CAI + TCIA + Stanford (T1+T2+T3, 899 patients)
          type: multi-site-biopsy
        metrics:
          - type: roc_auc
            value: 0.940
            name: ROC-AUC
          - type: pr_auc
            value: 0.814
            name: PR-AUC
          - type: sensitivity
            value: 0.753
            name: Sensitivity
  - name: modelGRAFT
    results:
      - task:
          type: image-segmentation
          name: Clinically Significant Prostate Cancer Detection
        dataset:
          name: PI-CAI + TCIA (T1+T2, 547 patients)
          type: multi-site-biopsy
        metrics:
          - type: roc_auc
            value: 0.921
            name: ROC-AUC (in-distribution)
      - task:
          type: image-segmentation
          name: Clinically Significant Prostate Cancer Detection
        dataset:
          name: Stanford (D3+T3, 1756 patients)
          type: out-of-distribution
        metrics:
          - type: roc_auc
            value: 0.836
            name: ROC-AUC (out-of-distribution)

---

# GRAFT — Guided Reinforcement Learning with Agentic Fused Teachers

GRAFT is a reinforcement learning–guided knowledge distillation framework for
clinically significant prostate cancer (csPCa) detection on bi-parametric MRI.
It trains a single lightweight 3D U-Net student model by adaptively distilling
knowledge from multiple teacher models, using a policy-gradient RL agent that
dynamically weights teacher contributions based on their predictive accuracy
and mutual disagreement. The result is a compact, deployable detector that
matches or surpasses full teacher ensembles while requiring only a single
forward pass at inference — achieving **>1,000× speedup** over the Teacher Average.

This release contains the training and evaluation code for the **5-fold nnUNet
ensemble** configuration (foldGRAFT) used in the GRAFT paper. The teachers are
five fold-specific nnUNet checkpoints; the student is a full-resolution 3D U-Net
trained jointly with a small policy network that learns how to weight each
teacher's intermediate features at every training step.

> Dataset / image / teacher-feature paths in this release have been replaced
> with `<PLACEHOLDER>` strings. Before running anything, edit the launch script
> and the dataset YAML to point at your own data.

**Paper:** Li, C.X., et al. *GRAFT: Guided Reinforcement Learning with Agentic
Fused Teachers.* (2026). [[GitHub](https://github.com/pimed/GRAFT)]

---

## Method

For each training batch the student forwards the 3D MR volume and produces
intermediate decoder features. The teachers' features for that same case are
loaded from disk (pre-extracted once and cached). A tiny **policy network**
observes per-teacher state — cosine similarity, BCE accuracy, and (optionally)
inter-teacher disagreement — and outputs a softmax weight vector
`w ∈ Δ^{N_teachers}`. The distillation loss is a weighted feature MSE

```
L_student = L_seg + w * Σᵢ wᵢ · D(F_student, F_teacherᵢ)
```

combined with cross-entropy / Dice / focal supervised losses on the labels. The
policy is updated with REINFORCE-style returns where the reward is the
per-episode improvement in validation Dice (and optionally training loss
reduction). See `meta_teacher_optimizer.py` and the `train_agent_episode`
function in `train_loops_new_policy.py`.

### Model Variants

| Variant | Description | # Teachers |
|---|---|---|
| **modelGRAFT** | Distills across different model architectures (nnUNet, ProViCNet, ProstAtlasDiff) | 3 |
| **foldGRAFT** | Distills across 5-fold cross-validation nnUNet models *(this release)* | 5 |
| **fedGRAFT** | Federated multi-site distillation; sites exchange intermediate features, not images or weights | 3 (1 per site) |
| **multiGRAFT** | Cascaded distillation from 15 teachers (3 architectures × 5 folds); best overall performance | 15 |

---

## Intended Use

- Automated detection and segmentation of clinically significant prostate
  cancer (ISUP Grade Group ≥ 2) on bi-parametric MRI as a **decision-support
  tool** for radiologists.
- Efficient deployment alternative to full teacher model ensembles — a single
  forward pass replaces 3–15 independent inferences.
- Multi-institutional and privacy-preserving deployment via fedGRAFT, where
  sites share only intermediate features, not patient images or model weights.

**Out of scope:** This model is not intended for autonomous clinical diagnosis
without radiologist review. Not validated for prostate cancer staging, Gleason
grading, non-prostate anatomical sites, or DCE sequences.

---

## Evaluation Results

Lesion-level detection using a sextant biopsy–inspired protocol (true positive:
90th percentile of predicted voxels within ground-truth lesion boundary
classified as cancer).

### Experiment 1 — modelGRAFT

| Model | ROC-AUC (T1+T2, in-dist.) | ROC-AUC (D3+T3, out-of-dist.) |
|---|---|---|
| nnUNet | 0.911 | 0.817 |
| ProViCNet | 0.802 | 0.756 |
| ProstAtlasDiff | 0.885 | 0.834 |
| Teacher Average | 0.917 | 0.822 |
| **modelGRAFT** | **0.926** | **0.836** |

### Experiment 2 — foldGRAFT

| Model | ROC-AUC (T1+T2) | ROC-AUC (D3+T3) |
|---|---|---|
| nnUNet-5Fold | 0.910 | 0.789 |
| **foldGRAFT** | **0.903** | **0.833** |

### Experiment 3 — fedGRAFT

| Model | ROC-AUC (T1+T2+T3) |
|---|---|
| FedAvg | 0.820 |
| Single-teacher distillation | 0.839 |
| **fedGRAFT** | **0.889** |

### Experiment 4 — multiGRAFT

| Model | ROC-AUC(T1+T2+T3)| RP |
|---|---|---|
| foldGRAFT-nnUNet | 0.901 | 0.898 | 
| foldGRAFT-ProViCNet | 0.840 | 0.895 | 
| foldGRAFT-ProstAtlasDiff | 0.856 | 0.928 | 
| **multiGRAFT** | **0.940** | **0.929** |

multiGRAFT detected **7.5% more clinically significant cancers than radiologists**
on the independent radical prostatectomy cohort (93 patients, GE scanners).

### Inference Runtime

| Model | GPU (RTX A6000) | CPU (Xeon Gold 5320) |
|---|---|---|
| **modelGRAFT** | **0.06 s** | **1.64 s** |
| nnUNet | ~1 s | ~75 s |
| ProViCNet | ~4.4 s | ~17 s |
| ProstAtlasDiff | ~30 s | ~1,500 s |
| Teacher Average | ~95 s | ~3,000 s |

---

## Limitations

- All experiments were conducted on **retrospective datasets**; prospective
  validation has not yet been performed.
- Teacher pool is currently limited to three architectural families; behavior with larger
  or more diverse pools is unexplored.
- A formal privacy analysis for fedGRAFT has not yet been conducted.
- Missed lesions tend to be smaller (median 226.5 mm³); small-lesion
  sensitivity may be a limitation in screening contexts.

---

## Repository Layout

```
GRAFT/
├── run_5fold_201_binary_focal_neg1.sh   # Example launch script (5-fold, binary, focal)
├── train_student_rl_new_policy.py       # Main DDP training entry point
├── train_loops_new_policy.py            # Train / val / episode-RL inner loops
├── meta_teacher_optimizer.py            # Policy-gradient teacher-weight optimizer
├── validation_visualizer.py             # Sample qualitative validation panels
├── utils.py                             # Logging, AMP helpers, distillation losses
├── metrics.py                           # Dice / IoU / lesion-level metrics
├── setting.py                           # Optional 2D-teacher checkpoint registry
├── dataset/
│   ├── pimed_dataset_configs_local_region_5fold_binary_201.yaml  # Edit me
│   ├── pimedloader_3d.py                # Base 3D prostate-MR dataloader
│   ├── pimedloader_3d_with_features.py  # Adds teacher feature loading + LRU cache
│   ├── fixed_subset_sampler.py          # Per-rank fixed-size sampler for DDP
│   ├── region_converter.py              # 3-class ↔ region-based label encoding
│   └── base.py
├── distiller_zoo/
│   ├── contrastive_kd.py                # Contrastive feature KD loss
│   ├── feature_mse_mtkd_rl.py           # Per-teacher weighted feature MSE
│   ├── normalized_feature_losses.py     # Channel-normalized feature distance
│   └── segmentation_3d_losses.py        # Dice / focal / region-based seg losses
└── models/
    ├── fullres_3d_unet.py               # nnUNet-style full-resolution 3D U-Net (student)
    ├── unet3d.py                        # Lightweight 3D U-Net variants
    ├── swin_unetr3d.py                  # 3D Swin-UNETR backbones
    ├── policy.py                        # Policy network producing per-teacher weights
    ├── util.py / util_learned_resize.py # Building blocks
    └── (2D classification models kept for compatibility)
```

---

## Inputs You Need to Supply

1. **Multi-cohort 3D prostate-MR dataset** in nnUNet-style layout:
   - `imagesTr/` — `<case>_0000.nii.gz`, `<case>_0001.nii.gz`, `<case>_0002.nii.gz`
     (T2w / ADC / DWI stacked channels)
   - `masksTr/` — prostate masks
   - `labelsTr/` — lesion / cancer labels

2. **Per-cohort 5-fold split JSONs** — case-name lists for each fold.

3. **Per-case z-score normalization stats** — a JSON
   `{ "<case_id>": { "mean": <float>, "std": <float> }, ... }`.

4. **Pre-trained nnUNet 5-fold teachers**, plus their **extracted decoder
   features** saved per case as `.pt` (one directory per fold). The extractor
   used to produce these files is not included in this release — any compatible
   per-case feature dump that keys on the same case IDs as `imagesTr/` will work.


All five locations are configured in
[dataset/pimed_dataset_configs_local_region_5fold_binary_201.yaml](dataset/pimed_dataset_configs_local_region_5fold_binary_201.yaml).

### MRI Preprocessing

Input volumes are expected to be preprocessed as follows before feature extraction:

| Property | Value |
|---|---|
| Modalities | T2w, ADC, high-b DWI |
| In-plane resolution | 0.5 × 0.5 mm (B-Spline resampled) |
| Slice thickness | 3.0 mm |
| Field of view | 128 × 128 mm axial (256 × 256 voxels), 20 slices centered on the prostate |
| Normalization | Per-channel zero mean, unit standard deviation |

---

## Environment

The code targets PyTorch ≥ 2.0 with CUDA. Recommended:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install \
    "torch>=2.0" torchvision \
    numpy scipy scikit-image scikit-learn pandas \
    SimpleITK torchio nibabel matplotlib tqdm pyyaml
```

Multi-GPU training uses `torch.distributed.run` with NCCL.

---

## Training

1. Open [run_5fold_201_binary_focal_neg1.sh](run_5fold_201_binary_focal_neg1.sh)
   and edit the activation line for your Python environment and
   `CUDA_VISIBLE_DEVICES` / `NUM_GPUS`.

2. Open [dataset/pimed_dataset_configs_local_region_5fold_binary_201.yaml](dataset/pimed_dataset_configs_local_region_5fold_binary_201.yaml)
   and replace every `<PATH_TO_...>` placeholder with a real path on your system.

3. Launch:

   ```bash
   bash run_5fold_201_binary_focal_neg1.sh
   ```

The script defaults to 2 GPUs, batch size 2 × 2-step gradient accumulation,
200 epochs, BF16 mixed precision, full-resolution 3D U-Net student, deep
supervision, region-based training, episode-based RL rewards (γ = 0.95,
α = 0.7), and feature distillation from the second-to-last decoder stage
(`--distill-features neg1`).

### Key Command-Line Flags

| Flag | Purpose |
|---|---|
| `--arch` | Student architecture (default `fullres_3d_unet`; see `models/__init__.py`) |
| `--data-configs` | Path to the dataset YAML |
| `--teacher-name-list` | Teacher keys matching `teachers_features_paths` in the YAML |
| `--distill-features` | Which decoder stage(s) to distill: `neg1`, `neg2`, or both |
| `--use-region-based-training` | Use prostate-region label encoding |
| `--use-deep-supervision` | Enable nnUNet-style deep supervision on the student |
| `--use-episode-reward` | Use validation-based episode returns instead of per-step rewards |
| `--use-disagreement-reward` | Add inter-teacher disagreement to the policy's state vector |
| `--reward-alpha` | Reward blend: 1.0 = pure Dice gain, 0.0 = pure loss reduction |
| `--agent-step` | Number of optimizer steps before each agent update |
| `--agent-warmup-epochs` | Epochs to linearly blend uniform → learned teacher weights |
| `--ce-weight / --kd-weight / --feat-weight` | Loss term weights |
| `--bf16` / `--fp16` | Mixed-precision mode |
| `--checkpoint-dir` | Where to write `checkpoint_best.pth` + per-epoch state |
| `--innovation-suffix` | String appended to the auto-generated experiment subfolder |

For the full list run:

```bash
python train_student_rl_new_policy.py --help
```

---

## Notes & Gotchas

- Teacher feature directories must contain one `.pt` (or `.npz`) per
  training/validation case, keyed by the same case ID used in the nnUNet
  `imagesTr/` filenames. Feature shapes must match the student's corresponding
  decoder stage.
- `pimedloader_3d_with_features.py` uses an LRU cache (`--cache-size`,
  default 300 cases/GPU) so feature loading does not blow up RAM when
  shuffling is enabled.
- If you train without the RL agent (`--agent-warmup-epochs` ≥ total epochs),
  GRAFT reduces to a uniformly-weighted multi-teacher feature distillation
  baseline.
- `setting.py` exists for compatibility with legacy 2D classification teachers;
  it is **not** needed for the 3D segmentation pipeline.

---

## Citation

If you use this code or model, please cite:

```bibtex
@article{li2026graft,
  title   = {{GRAFT}: Guided Reinforcement Learning with Agentic Fused Teachers},
  author  = {Li, Cynthia X. and others},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Funding

This work was supported by Stanford University (Departments of Radiology and
Urology) and by the National Cancer Institute, National Institutes of Health
(R37CA260346).

## License

Released for research use.
