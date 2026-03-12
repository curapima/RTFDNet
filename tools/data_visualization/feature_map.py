# -*- coding: utf-8 -*-
import os
import os.path as osp
from argparse import ArgumentParser
from typing import Any, List, Optional

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32))
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn) * 255.0
    return arr.clip(0, 255).astype(np.uint8)

def voc_like_palette(n: int) -> np.ndarray:
    """PASCAL VOC 风格调色板（RGB）"""
    n = int(max(n, 256))
    pal = np.zeros((n, 3), dtype=np.uint8)
    for j in range(n):
        lab = j
        r = g = b = 0
        i = 0
        while lab:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            lab >>= 3
            i += 1
        pal[j] = [r, g, b]
    return pal

def save_gray_and_color(label_2d_uint: np.ndarray, out_prefix: str, num_classes_hint: Optional[int] = None):
    cv2.imwrite(out_prefix + "_gray.png", label_2d_uint)
    n_cls = int(num_classes_hint) if (num_classes_hint is not None) else 256
    palette_rgb = voc_like_palette(n_cls)  # RGB
    color_rgb = palette_rgb[label_2d_uint]  # HxWx3
    color_bgr = color_rgb[..., ::-1]
    cv2.imwrite(out_prefix + "_color.png", color_bgr)

def basename_noext(p: str) -> str:
    return osp.splitext(osp.basename(p))[0]


def _to_hwc_uint8(x: np.ndarray) -> np.ndarray:
    """把任意 (H,W), (H,W,C), (C,H,W) 转为 (H,W,3) uint8（逐通道归一化）。"""
    if x.ndim == 2:
        g = normalize_to_uint8(x)
        return cv2.merge([g, g, g])
    if x.ndim == 3:
        # CHW -> HWC ?
        if x.shape[0] in (1, 3, 4) and x.shape[1] > 16 and x.shape[2] > 16:
            x = np.transpose(x, (1, 2, 0))
        if x.shape[2] == 1:
            g = normalize_to_uint8(x[..., 0])
            return cv2.merge([g, g, g])
        c3 = x[..., :3].astype(np.float32)
        chans = [normalize_to_uint8(c3[..., i]) for i in range(3)]
        return cv2.merge(chans)
    # fallback
    g = normalize_to_uint8(x.astype(np.float32))
    return cv2.merge([g, g, g])

def load_vis_bg(img_path: str) -> np.ndarray:
    ext = osp.splitext(img_path)[1].lower()
    if ext == ".npy":
        arr = np.load(img_path)
        hwc = _to_hwc_uint8(arr)
        return hwc[..., ::-1]  # RGB->BGR
    else:
        return mmcv.imread(img_path, 'color')  # BGR


def save_channel_heatmaps_cam(
    t_chw: torch.Tensor,
    out_dir: str,
    prefix: str,
    base_bgr: Optional[np.ndarray] = None,
    upsample_to: Optional[tuple] = None,
    alpha: float = 0.35
):
    """
    对 (C,H,W) 的每个通道：ReLU → 归一化 → JET 伪彩；可选叠加到 base_bgr。
    产物：
      {prefix}_chXX_gray.png
      {prefix}_chXX_color.png
      {prefix}_chXX_overlay.png（若有 base_bgr）
    """
    ensure_dir(out_dir)
    t_cpu = t_chw.detach().cpu()
    C, H, W = t_cpu.shape

    if upsample_to is not None:
        tgt_W, tgt_H = int(upsample_to[0]), int(upsample_to[1])
    elif base_bgr is not None:
        tgt_W, tgt_H = base_bgr.shape[1], base_bgr.shape[0]
    else:
        tgt_W, tgt_H = W, H

    for c in range(C):
        ch = t_cpu[c].numpy().astype(np.float32)
        ch = np.maximum(ch, 0.0)           # ReLU
        gray = normalize_to_uint8(ch)

        if (gray.shape[1] != tgt_W) or (gray.shape[0] != tgt_H):
            gray_r = cv2.resize(gray, (tgt_W, tgt_H), interpolation=cv2.INTER_CUBIC)
        else:
            gray_r = gray

        color = cv2.applyColorMap(gray_r, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_ch{c:02d}_gray.png"), gray_r)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_ch{c:02d}_color.png"), color)

        if base_bgr is not None:
            if (base_bgr.shape[1] != tgt_W) or (base_bgr.shape[0] != tgt_H):
                base_r = cv2.resize(base_bgr, (tgt_W, tgt_H), interpolation=cv2.INTER_CUBIC)
            else:
                base_r = base_bgr
            overlay = (alpha * color + (1.0 - alpha) * base_r).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_ch{c:02d}_overlay.png"), overlay)


def save_channels_merged_cam(
    t_chw: torch.Tensor,
    out_dir: str,
    prefix: str = "merged",
    base_bgr: Optional[np.ndarray] = None,
    upsample_to: Optional[tuple] = None,   # (W,H)
    alpha: float = 0.35,
    reduce: str = "max",                   # max / mean / sum / l2
    per_channel_norm: bool = False
):
    """
    将 (C,H,W) 的多通道特征合成为一张 CAM 风格热力图:
      1) ReLU 保留正响应
      2) (可选) 对每通道各自 min-max 到 [0,1]
      3) 通道维做 reduce 合并: max / mean / sum / l2
      4) 全局 min-max 归一化 → JET 上色 → (可选)叠加到底图

    导出：
      {prefix}_gray.png
      {prefix}_color.png
      {prefix}_overlay.png（若有 base_bgr）
    """
    ensure_dir(out_dir)
    x = t_chw.detach().cpu().float()       # (C,H,W)
    x = torch.relu(x)

    if per_channel_norm:
        C = x.shape[0]
        x2 = []
        for c in range(C):
            ch = x[c]
            ch = ch - ch.min()
            den = ch.max()
            if den > 1e-12:
                ch = ch / den
            x2.append(ch)
        x = torch.stack(x2, dim=0)

    if reduce == "max":
        merged = x.max(dim=0).values
    elif reduce == "mean":
        merged = x.mean(dim=0)
    elif reduce == "sum":
        merged = x.sum(dim=0)
    elif reduce == "l2":
        merged = torch.sqrt((x * x).sum(dim=0) + 1e-12)
    else:
        raise ValueError(f"Unknown reduce: {reduce}")

    merged = merged - merged.min()
    den = merged.max()
    if den > 1e-12:
        merged = merged / den
    merged_np = merged.numpy().astype(np.float32)

    if upsample_to is not None:
        tgt_W, tgt_H = int(upsample_to[0]), int(upsample_to[1])
    elif base_bgr is not None:
        tgt_W, tgt_H = base_bgr.shape[1], base_bgr.shape[0]
    else:
        tgt_H, tgt_W = merged_np.shape[0], merged_np.shape[1]

    gray_u8 = (cv2.resize(merged_np, (tgt_W, tgt_H), interpolation=cv2.INTER_CUBIC) * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)

    cv2.imwrite(os.path.join(out_dir, f"{prefix}_gray.png"), gray_u8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_color.png"), color)

    if base_bgr is not None:
        if (base_bgr.shape[1] != tgt_W) or (base_bgr.shape[0] != tgt_H):
            base_r = cv2.resize(base_bgr, (tgt_W, tgt_H), interpolation=cv2.INTER_CUBIC)
        else:
            base_r = base_bgr
        overlay = (alpha * color + (1.0 - alpha) * base_r).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_overlay.png"), overlay)


def export_mmseg_result_cam(
    result: SegDataSample,
    out_dir: str,
    base_bgr: Optional[np.ndarray] = None,
    merge_reduce: str = "max",
    per_channel_norm: bool = False,
):
    ensure_dir(out_dir)
    allowlist = [
        "seg_logits", "seg_logits_x", "seg_logits_fuse"
    ]
    discovered: List[str] = []
    if hasattr(result, "_data_fields"):
        discovered.extend(list(result._data_fields))
    elif hasattr(result, "keys"):
        try:
            discovered.extend(list(result.keys()))
        except Exception:
            pass
        
    seen = set()
    field_names: List[str] = []
    for k in allowlist + discovered:
        if k and (k not in seen):
            seen.add(k)
            field_names.append(k)

    for field in field_names:
        try:
            node = result.get(field, None)
        except Exception:
            continue
        if node is None or not hasattr(node, "data"):
            continue
        t = node.data
        if not isinstance(t, torch.Tensor):
            continue

        t = t.detach().cpu()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim >= 4:
            t = t[0]
            if t.ndim == 2:
                t = t.unsqueeze(0)
        elif t.ndim != 3:
            print(f"[{field}] 跳过：不支持的维度 {tuple(t.shape)}")
            continue

        C, H, W = t.shape
        fld_out = os.path.join(out_dir, field)
        ensure_dir(fld_out)

        if C > 1:
            save_channels_merged_cam(
                t, fld_out, prefix="logit_merged",
                base_bgr=base_bgr,
                upsample_to=(base_bgr.shape[1], base_bgr.shape[0]) if base_bgr is not None else None,
                alpha=0.35,
                reduce=merge_reduce,
                per_channel_norm=per_channel_norm
            )
            label = torch.argmax(t, dim=0).to(torch.uint8).numpy()
            save_gray_and_color(label, os.path.join(fld_out, "argmax"), num_classes_hint=C)

        else:
            t2d = t[0]
            if torch.is_floating_point(t2d):
                gray = normalize_to_uint8(t2d.numpy())
                color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(fld_out, "chan0_gray.png"), gray)
                cv2.imwrite(os.path.join(fld_out, "chan0_color.png"), color)
                if base_bgr is not None:
                    tgt = (base_bgr.shape[1], base_bgr.shape[0])
                    if (gray.shape[1] != tgt[0]) or (gray.shape[0] != tgt[1]):
                        gray = cv2.resize(gray, tgt, interpolation=cv2.INTER_CUBIC)
                        color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                    overlay = (0.35 * color + 0.65 * base_bgr).astype(np.uint8)
                    cv2.imwrite(os.path.join(fld_out, "chan0_overlay.png"), overlay)
            else:
                label = t2d.to(torch.uint8).numpy()
                save_gray_and_color(label, os.path.join(fld_out, "label"), num_classes_hint=256)

    print(f"✅ CAM Output：{os.path.abspath(out_dir)}")

def main():
    img_list = os.listdir("/root/autodl-tmp/ade_mf/rgbx_np/validation/")
    for i in range(len(img_list)):
        image_path = "/root/autodl-tmp/ade_mf/rgbx_np/validation/" + img_list[i] 
        parser = ArgumentParser(description="MMSeg CAM")
        parser.add_argument('--img', default=image_path, help='Image file (.png/.jpg/.npy)')
        parser.add_argument('--config', default=r'/root/RXDistill/configs/segformer/EAEF_mit-b2_1xb8-40K_MF_512x512.py', help='config path')
        parser.add_argument('--checkpoint', default=r'/root/RXDistill/work_dirs/EAEF_mit-b2_1xb8-40K_MF_512x512/best_mIoU_epoch_191.pth')
        parser.add_argument('--device', default='cuda:0', help='cuda:0 / cpu')
        parser.add_argument('--merge-reduce', choices=['max', 'mean', 'sum', 'l2'], default='max')
        parser.add_argument('--per-channel-norm', action='store_true', default=False)
        parser.add_argument('--save-split', action='store_true', default=False)
        args = parser.parse_args()
        register_all_modules()
        model = init_model(args.config, args.checkpoint, device=args.device)
        if args.device == 'cpu':
            model = revert_sync_batchnorm(model)
        result: SegDataSample = inference_model(model, args.img)
        base_bgr = load_vis_bg(args.img)
        vis_root = f"vis_{basename_noext(args.img)}"
        export_mmseg_result_cam(
            result,
            out_dir=os.path.join(vis_root, "tensors_cam"),
            base_bgr=base_bgr,
            merge_reduce=args.merge_reduce,
            per_channel_norm=args.per_channel_norm,
        )

if __name__ == "__main__":
    main()
