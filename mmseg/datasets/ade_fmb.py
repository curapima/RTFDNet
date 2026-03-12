# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FMB_ADE20KDataset(BaseSegDataset):
    """ADE20K dataset.
    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('Road', 'Sidewalk', 'Building', 'T-lamp', 'T-Sign' ,'Vegetation', 'Sky', 'Person','Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle' , 'pole'),
        palette=[[64, 0, 128], [64, 64, 0], [0, 128, 192],[0, 0, 192], [128, 128, 0], [64, 64, 128] ,[192, 21, 35],[90, 128, 128],
                 [55, 45, 128],[78, 65, 69],[0, 102, 200], [61, 230, 250], [255, 6, 51],[192, 64, 0]])

    def __init__(self,
                 img_suffix='.npy',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
