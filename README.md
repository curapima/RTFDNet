# RXDistill (RTFDNet)

Official code release for the paper:

**RTFDNet: Fusion-Decoupling for Robust RGB-T Segmentation**  
Kunyu Tan, Mingjian Liang  
[arXiv:2603.09149](https://arxiv.org/abs/2603.09149) | [PDF](https://arxiv.org/pdf/2603.09149v1)

This repository currently provides **code + configs only** (no pretrained weights).

## 1. Overview

RXDistill is built on top of MMSegmentation and focuses on robust RGB-T semantic segmentation under modality degradation.

Core ideas in the paper:
- Synergistic Feature Fusion (SFF)
- Cross-Modal Decouple Regularization (CMDR)
- Region Decouple Regularization (RDR)

Project-specific modules in this repo:
- Dual-modal backbone: `BIMixVisionTransformer`  
  `mmseg/models/backbones/cross_segform_model.py`
- Segmentor: `EncoderDecoder_mult`  
  `mmseg/models/segmentors/encoder_decoder_mult.py`
- Custom losses: `AKDLoss`, `RegionL1`, `M_CrossEntropyLoss`  
  `mmseg/models/losses/`
- Custom RGB-X loader and datasets  
  `mmseg/datasets/loadimg/LoadImageFromFile_rgbx.py`, `mmseg/datasets/`

## 2. Environment Setup

Recommended: Python 3.8-3.10 + CUDA-enabled PyTorch.

```bash
# 1) install mmcv/mmengine via OpenMIM
pip install -U openmim
mim install -r requirements/mminstall.txt

# 2) install project deps
pip install -r requirements.txt
pip install -v -e .
```

Optional logging backend:
```bash
pip install -U swanlab
```

## 3. Data Format

This project uses `.npy` multimodal inputs + `.png` masks.

Expected directory layout for each dataset root:

```text
<DATA_ROOT>/
  rgbx_np/
    training/
      xxx.npy
    validation/
      xxx.npy
  annotations/
    training/
      xxx.png
    validation/
      xxx.png
```

Input assumption:
- each sample is a 6-channel tensor stored in `.npy`
- channels are split as `RGB = [:3]`, `X modality = [3:6]` in the backbone

You must update paths in your config before running:
- `data_root`
- `pretrained` (if you use local pretrained backbone path)
- `work_dir` (optional but recommended)

## 4. Available Main Configs

Located in `configs/segformer/`:

| Config | Dataset Type | Classes |
| --- | --- | --- |
| `EAEF_mit-b2_1xb8-40K_FMB_512x512.py` | `FMB_ADE20KDataset` | 14 |
| `EAEF_mit-b2_1xb8-40K_MF_512x512.py` | `MSRS_ADE20KDataset` | 9 |
| `EAEF_mit-b2_1xb8-40K_MSRS_512x512.py` | `MSRS_ADE20KDataset` | 9 |
| `EAEF_mit-b2_1xb8-40K_PST_512x512.py` | `PST_ADE20KDataset` | 5 |
| `EAEF_mit-b4_1xb8-40K_FMB_512x512.py` | `FMB_ADE20KDataset` | 14 |
| `EAEF_mit-b4_1xb8-40K_MF_512x512.py` | `MSRS_ADE20KDataset` | 9 |
| `EAEF_mit-b4_1xb8-40K_PST_512x512.py` | `PST_ADE20KDataset` | 5 |

## 5. Training

### Method A (your required workflow)

Edit default config in `tools/train.py`:

```python
parser.add_argument(
    '--config',
    default='configs/segformer/EAEF_mit-b4_1xb8-40K_PST_512x512.py'
)
```

Then run:

```bash
python tools/train.py
```

### Method B (recommended CLI)

```bash
python tools/train.py \
  --config configs/segformer/EAEF_mit-b2_1xb8-40K_MF_512x512.py \
  --work-dir work_dirs/mf_b2_exp1
```

### Method C (batch multiple configs)

Edit `CONFIGS` in `train.sh`, then:

```bash
bash train.sh
```

### Multi-GPU training

```bash
torchrun --nproc_per_node=4 tools/train.py \
  --config configs/segformer/EAEF_mit-b2_1xb8-40K_MF_512x512.py \
  --launcher pytorch
```

Or:

```bash
bash tools/dist_train.sh configs/segformer/EAEF_mit-b2_1xb8-40K_MF_512x512.py 4
```

## 6. Evaluation / Testing

```bash
python tools/test.py \
  --config configs/segformer/EAEF_mit-b2_1xb8-40K_MF_512x512.py \
  --checkpoint work_dirs/mf_b2_exp1/best_mIoU_epoch_xxx.pth
```

Useful options:
- `--tta`: enable test-time augmentation
- `--show`: visualize online
- `--show-dir <path>`: save visualization images
- `--out <path>`: save predictions for offline analysis

Multi-GPU testing:

```bash
torchrun --nproc_per_node=4 tools/test.py \
  --config configs/segformer/EAEF_mit-b2_1xb8-40K_MF_512x512.py \
  --checkpoint work_dirs/mf_b2_exp1/best_mIoU_epoch_xxx.pth \
  --launcher pytorch
```

Or:

```bash
bash tools/dist_test.sh \
  configs/segformer/EAEF_mit-b2_1xb8-40K_MF_512x512.py \
  work_dirs/mf_b2_exp1/best_mIoU_epoch_xxx.pth 4
```

## 7. Logging and Visualization

- Local logs/checkpoints are saved under `work_dirs/`.
- SwanLab backend is already integrated in `mmseg/visualization/swanlab_log_hook.py`.
- To disable SwanLab, remove `swanlabVisBackend` from `vis_backends` in your config.

## 8. Common Issues

1. `FileNotFoundError` with `/root/...` paths  
   Fix absolute paths in config (`data_root`, `pretrained`, `work_dir`) to your local machine.

2. `No module named swanlab`  
   Install it (`pip install -U swanlab`) or remove SwanLab backend from config.

3. Shape/channel mismatch  
   Ensure input `.npy` follows RGB-X format with 6 channels and correct label ids.

## 9. Citation

If this repository helps your work, please cite:

```bibtex
@article{tan2026rtfdnet,
  title={RTFDNet: Fusion-Decoupling for Robust RGB-T Segmentation},
  author={Tan, Kunyu and Liang, Mingjian},
  journal={arXiv preprint arXiv:2603.09149},
  year={2026}
}
```

## 10. Acknowledgements

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [OpenMMLab](https://openmmlab.com/)
- [SegFormer](https://arxiv.org/abs/2105.15203)
