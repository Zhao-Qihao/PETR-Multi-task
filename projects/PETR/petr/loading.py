# Copyright (c) 2024 Your Company. All rights reserved.
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from mmcv.transforms.base import BaseTransform
from mmdet.datasets.transforms import LoadAnnotations
from mmengine.registry import TRANSFORMS
from einops import rearrange


@TRANSFORMS.register_module()
class LoadMapsFromFiles(BaseTransform):
    """Load map masks from files.

    Required Keys:
        - map_filename

    Modified Keys:
        - maps
        - gt_map
        - map_shape

    Args:
        k (int, optional): Not used. Defaults to None.
    """

    def __init__(self, k: Optional[int] = None):
        self.k = k

    def transform(self, results: dict) -> dict:
        """Call function to load map masks from files.

        Args:
            results (dict): Result dict containing the map filename.

        Returns:
            dict: Updated result dict with loaded map data.
        """
        pts_file_path = results['lidar_points']['lidar_path']
        # map_filename = results['map_filename']
        map_filename = '/data1051n/algorithm/home/zhaoqh/petr_extr_data/HDmaps-final/' + pts_file_path.replace('.bin', '.npz')
        maps = np.load(map_filename)
        map_mask = maps['arr_0'].astype(np.float32)

        # [C, H, W]
        maps = map_mask.transpose((2, 0, 1))
        results['gt_map'] = maps

        # Flatten and reshape
        maps = rearrange(maps, 'c (h h1) (w w2) -> (h w) c h1 w2', h1=16, w2=16)
        maps = maps.reshape(256, 3 * 256)

        results['maps'] = maps
        results['map_shape'] = maps.shape

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(k={self.k})'
        return repr_str


@TRANSFORMS.register_module()
class LoadMapsFromFiles_flattenf200f3(BaseTransform):
    """Load and flatten map masks into binary format.

    Required Keys:
        - map_filename

    Modified Keys:
        - maps
        - gt_map
        - map_shape

    Args:
        k (int, optional): Not used. Defaults to None.
    """

    def __init__(self, k: Optional[int] = None):
        self.k = k

    def transform(self, results: dict) -> dict:
        """ Call function to load and flatten map masks.

        Args:
            results (dict): Result dict containing the map filename.

        Returns:
            dict: Updated result dict with loaded map data.
        """
        pts_file_path = results['lidar_points']['lidar_path'].split('/')[-1]
        # map_filename = results['map_filename']
        map_filename = 'data/nuscenes/HDmaps-final/' + pts_file_path.replace('.bin', '.npz')
        maps = np.load(map_filename)
        map_mask = maps['arr_0'].astype(np.float32)

        # [C, H, W]
        maps = map_mask.transpose((2, 0, 1))
        results['gt_map'] = maps

        # Reshape and binarize
        maps = maps.reshape(3, 200 * 200)
        maps[maps >= 0.5] = 1
        maps[maps < 0.5] = 0
        maps = 1 - maps

        results['maps'] = maps
        results['map_shape'] = maps.shape

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(k={self.k})'
        return repr_str