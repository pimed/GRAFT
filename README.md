# GRAFT — RL-Guided Ensemble Knowledge Distillation for 3D Prostate MR Segmentation

GRAFT is a 3D segmentation training pipeline that distills an ensemble of
pre-trained "teacher" networks into a single 3D U-Net **student**, using a
policy-gradient agent to learn per-teacher feature weights from validation
feedback.

This release contains the training and evaluation code for the **5-fold
nnUNet ensemble** configuration used in the GRAFT paper. The teachers are five
fold-specific nnUNet checkpoints; the student is a full-resolution 3D U-Net
trained jointly with a small policy network that learns how to weight each
teacher's intermediate features at every training step.

> The dataset / image / teacher-feature paths in this release have been
> replaced with `<PLACEHOLDER>` strings. Before running anything, edit the
> launch script and the dataset YAML to point at your own data.

---

## Repository layout

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

## Method, in one paragraph

For each training batch the student forwards the 3D MR volume and produces
intermediate decoder features. The teachers' features for that same case are
loaded from disk (pre-extracted once and cached). A tiny **policy network**
observes per-teacher state — cosine similarity, BCE accuracy, and (optionally)
inter-teacher disagreement — and outputs a softmax weight vector
`w ∈ Δ^{N_teachers}`. The distillation loss is a weighted feature MSE
`L_feat = Σᵢ wᵢ · ‖F_student − F_teacherᵢ‖²`, combined with cross-entropy /
Dice / focal supervised losses on the labels. The policy is updated with
REINFORCE-style returns where the reward is the per-episode improvement in
validation Dice (and optionally training loss reduction). See `meta_teacher_optimizer.py`
and the `train_agent_episode` function in `train_loops_new_policy.py`.

---

## Inputs you need to supply

1. **Multi-cohort 3D prostate-MR dataset** in nnUNet-style layout:
   - `imagesTr/` — `<case>_0000.nii.gz`, `<case>_0001.nii.gz`, `<case>_0002.nii.gz`
     (T2 / ADC / DWI stacked channels)
   - `masksTr/` — prostate masks
   - `labelsTr/` — lesion / cancer labels

2. **Per-cohort 5-fold split JSONs** — case-name lists for each fold.

3. **Per-case z-score normalization stats** — a JSON
   `{ "<case_id>": { "mean": <float>, "std": <float> }, ... }`.

4. **Pre-trained nnUNet 5-fold teachers**, plus their **extracted decoder
   features** saved per case as `.pt` (one directory per fold). The extractor
   used to produce these files is not included in this release — any
   compatible per-case feature dump that keys on the same case IDs as
   `imagesTr/` will work.

5. **(Optional)** A single nnUNet `checkpoint_best.pth` to initialize the
   student encoder weights. Skipping this just leaves the student randomly
   initialized.

All five locations are configured in
[dataset/pimed_dataset_configs_local_region_5fold_binary_201.yaml](dataset/pimed_dataset_configs_local_region_5fold_binary_201.yaml).

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
   and edit:
   - the activation line for your Python environment,
   - `CUDA_VISIBLE_DEVICES` / `NUM_GPUS`.

2. Open [dataset/pimed_dataset_configs_local_region_5fold_binary_201.yaml](dataset/pimed_dataset_configs_local_region_5fold_binary_201.yaml)
   and replace every `<PATH_TO_...>` placeholder with a real path on your
   system.

3. Launch:

   ```bash
   bash run_5fold_201_binary_focal_neg1.sh
   ```

The script defaults to 2 GPUs, batch size 2 × 2-step gradient accumulation,
200 epochs, BF16 mixed precision, full-resolution 3D U-Net student, deep
supervision, region-based training, episode-based RL rewards (γ = 0.95,
α = 0.7), and feature distillation from the second-to-last decoder stage
(`--distill-features neg1`).

### Key command-line flags

| Flag                          | Purpose                                                                 |
| ----------------------------- | ----------------------------------------------------------------------- |
| `--arch`                      | Student architecture (default `fullres_3d_unet`; see `models/__init__.py`) |
| `--data-configs`              | Path to the dataset YAML                                                |
| `--teacher-name-list`         | Teacher keys matching `teachers_features_paths` in the YAML             |
| `--distill-features`          | Which decoder stage(s) to distill: `neg1`, `neg2`, or both              |
| `--use-region-based-training` | Use prostate-region label encoding                                      |
| `--use-deep-supervision`      | Enable nnUNet-style deep supervision on the student                     |
| `--use-episode-reward`        | Use validation-based episode returns instead of per-step rewards        |
| `--use-disagreement-reward`   | Add inter-teacher disagreement to the policy's state vector             |
| `--reward-alpha`              | Reward blend: 1.0 = pure Dice gain, 0.0 = pure loss reduction           |
| `--agent-step`                | Number of optimizer steps before each agent update                      |
| `--agent-warmup-epochs`       | Epochs to linearly blend uniform → learned teacher weights              |
| `--ce-weight / --kd-weight / --feat-weight` | Loss term weights                                          |
| `--bf16` / `--fp16`           | Mixed-precision mode                                                    |
| `--checkpoint-dir`            | Where to write `checkpoint_best.pth` + per-epoch state                  |
| `--innovation-suffix`         | String appended to the auto-generated experiment subfolder              |

For the full list run:

```bash
python train_student_rl_new_policy.py --help
```

---

## Notes & gotchas

- The teacher feature directories must contain one `.pt` (or `.npz`) per
  training/validation case, keyed by the same case ID used in the nnUNet
  `imagesTr/` filenames. Feature shapes must match the student's
  corresponding decoder stage.
- `pimedloader_3d_with_features.py` uses an LRU cache (`--cache-size`,
  default 300 cases/GPU) so feature loading does not blow up RAM when
  shuffling is enabled.
- If you train without the RL agent (`--agent-warmup-epochs` ≥ total epochs),
  GRAFT reduces to a uniformly-weighted multi-teacher feature distillation
  baseline.
- `setting.py` exists for compatibility with legacy 2D classification
  teachers; it is **not** needed for the 3D segmentation pipeline.

---

## Citation

If you use this code, please cite the GRAFT paper.

## License

Released for research use. See `LICENSE` (add one before publishing).
