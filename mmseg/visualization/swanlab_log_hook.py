# Copyright (c) OpenMMLab. All rights reserved.
import copy
import functools
import logging
import os
import os.path as osp
import platform
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import MutableMapping
from typing import Any, Callable, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.fileio import dump
from mmengine.logging import MMLogger, print_log
from mmseg.registry import VISBACKENDS
from mmengine.visualization.vis_backend import BaseVisBackend


def force_init_env(old_func: Callable) -> Any:
    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        if not hasattr(obj, '_init_env'):
            raise AttributeError(f'{type(obj)} does not have _init_env method.')
        if not getattr(obj, '_env_initialized', False):
            print_log(
                'Attribute `_env_initialized` is not defined in '
                f'{type(obj)} or `{type(obj)}._env_initialized is False, '
                '`_init_env` will be called and the flag will be set to True',
                logger='current',
                level=logging.DEBUG)
            obj._init_env()  # type: ignore
            obj._env_initialized = True  # type: ignore
        return old_func(obj, *args, **kwargs)
    return wrapper


def _to_float(v):
    """Convert torch/np scalars to float for swanlab.log."""
    if isinstance(v, torch.Tensor):
        v = v.detach()
        return float(v.item() if v.numel() == 1 else v.mean().item())
    if isinstance(v, np.ndarray):
        return float(v.item() if v.size == 1 else v.mean())
    return float(v)


@VISBACKENDS.register_module()
class swanlabVisBackend(BaseVisBackend):
    def __init__(self,
                 name: str = 'visualizer',
                 save_dir: str = None,
                 init_kwargs: Optional[dict] = None,
                 define_metric_cfg: Union[dict, list, None] = None,
                 commit: Optional[bool] = True,
                 log_code_name: Optional[str] = None,
                 watch_kwargs: Optional[dict] = None):
        super().__init__(save_dir)
        self._init_kwargs = copy.deepcopy(init_kwargs)
        self._define_metric_cfg = define_metric_cfg
        self._commit = commit
        self._log_code_name = log_code_name
        self._watch_kwargs = watch_kwargs if watch_kwargs is not None else {}
        self._swanlab = None
        self._run = None

    def _init_env(self):
        """Setup env for swanlab."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore

        if self._init_kwargs is None:
            self._init_kwargs = {'logdir': self._save_dir}
        else:
            self._init_kwargs.setdefault('logdir', self._save_dir)

        try:
            import swanlab
        except ImportError:
            raise ImportError('Please run "pip install -U swanlab" to install swanlab')

        self._swanlab = swanlab
        self._run = swanlab.init(**self._init_kwargs)

        if self._define_metric_cfg is not None:
            try:
                if isinstance(self._define_metric_cfg, dict):
                    for metric, summary in self._define_metric_cfg.items():
                        self._swanlab.define_metric(metric, summary=summary)  # type: ignore[attr-defined]
                elif isinstance(self._define_metric_cfg, list):
                    for metric_cfg in self._define_metric_cfg:
                        self._swanlab.define_metric(**metric_cfg)  # type: ignore[attr-defined]
                else:
                    raise ValueError('define_metric_cfg should be dict or list')
            except AttributeError:
                print_log('swanlab.define_metric not available, skip metric definition.',
                          logger='current', level=logging.WARNING)

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return swanlab module (has swanlab.log/Image/Text...)."""
        return self._swanlab

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to swanlab."""
        cfg_dict = config.to_dict() if isinstance(config, Config) else dict(config)
        self._swanlab.config.update(cfg_dict)

    @force_init_env
    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict], **kwargs) -> None:
        """Record model info to swanlab (no native watch API)."""
        try:
            params = sum(p.numel() for p in model.parameters())
            self._swanlab.log({'model/params': int(params)}, step=0)
        except Exception:
            pass
        try:
            arch_txt = repr(model)
            self._swanlab.log({'model/arch': self._swanlab.Text(arch_txt)})
        except Exception as e:
            print_log(f'add_graph failed to log model text: {e}', logger='current', level=logging.WARNING)

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to swanlab. image should be RGB (H,W,3)/uint8."""
        bgr2rgb = kwargs.pop('bgr2rgb', True)
        caption = kwargs.pop('caption', None)
        size = kwargs.pop('size', None)

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                if bgr2rgb:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        img = self._swanlab.Image(image, caption=caption, size=size)
        self._swanlab.log({name: img}, step=step)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record scalar to swanlab."""
        try:
            v = _to_float(value)
        except Exception:
            v = float(value)
        self._swanlab.log({name: v}, step=step)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record multiple scalars to swanlab."""
        data = {}
        for k, v in scalar_dict.items():
            try:
                data[k] = _to_float(v)
            except Exception:
                data[k] = float(v)
        self._swanlab.log(data, step=step)

    def close(self) -> None:
        """Close swanlab experiment."""
        if hasattr(self, '_swanlab') and self._swanlab is not None:
            try:
                self._swanlab.finish()
            except Exception as e:
                print_log(f'swanlab.finish() failed: {e}', logger='current', level=logging.WARNING)
