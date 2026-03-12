# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader

def convert_mit(ckptr,ckptx,ckpt2):
    # Process the concat between q linear weights and kv linear weights
    for k, v in ckpt2.items():
        print(k)
        if k.split('.')[1] == 'layers_rgb':
            ckpt2[k] = ckptr[k]
        if k.split('.')[1] == 'fusion_conv':
            ckpt2[k] = ckptr[k]
        if k.split('.')[1] == 'convs':
            ckpt2[k] = ckptr[k]
        if k.split('.')[1] == 'conv_seg':
            ckpt2[k] = ckptr[k]
        if k.split('.')[1] == 'fuse_module':
            ckpt2[k] = ckptx[k]
        if k.split('.')[1] == 'layers_x':
            ckpt2[k] = ckptx[k]
        if k.split('.')[1] == 'fusion_conv_x':
            k2 = k.replace('fusion_conv_x', 'fusion_conv')
            ckpt2[k] = ckptx[k2]
        if k.split('.')[1] == 'convs_x':
            k2 = k.replace('convs_x', 'convs')
            ckpt2[k] = ckptx[k2]
    return ckpt2

def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained segformer to '
        'MMSegmentation style.')
    parser.add_argument('--src',default=r'D:\UniEAEF\pretrain\Combind\rgbt_best_mIoU_epoch_35.pth',help='src model path or url')
    parser.add_argument('--src2', default=r'D:\UniEAEF\pretrain\Combind\rgb_best_mIoU_epoch_48.pth',help='src model path or url')
    parser.add_argument('--src3', default=r'D:\UniEAEF\pretrain\Combind\strict_46.pth',help='src model path or url')
    parser.add_argument('--dst',default='D:/UniEAEF/pretrain/mix_eaef.pth', help='save path')
    args = parser.parse_args()

    checkpoint_rgbt = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    checkpoint_rgb = CheckpointLoader.load_checkpoint(args.src2, map_location='cpu')
    checkpoint_strict = CheckpointLoader.load_checkpoint(args.src3, map_location='cpu')

    if 'state_dict' in checkpoint_rgb:
        state_dict_rgb = checkpoint_rgb['state_dict']
    elif 'model' in checkpoint_rgb:
        state_dict_rgb = checkpoint_rgb['model']
    else:
        state_dict_rgb = checkpoint_rgb


    if 'state_dict' in checkpoint_rgbt:
        state_dict_rgbt = checkpoint_rgbt['state_dict']
    elif 'model' in checkpoint_rgbt:
        state_dict_rgbt = checkpoint_rgbt['model']
    else:
        state_dict_rgbt = checkpoint_rgbt

    if 'state_dict' in checkpoint_strict:
        state_dict_s = checkpoint_strict['state_dict']
    elif 'model' in checkpoint_strict:
        state_dict_s = checkpoint_strict['model']
    else:
        state_dict_s = checkpoint_strict

    weight = convert_mit(state_dict_rgb,state_dict_rgbt,state_dict_s)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
