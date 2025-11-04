from mmdet3d.datasets.transforms import Pack3DDetInputs
from mmcv.transforms import BaseTransform
import mmcv
from mmengine.registry import TRANSFORMS
from mmdet3d.datasets.transforms.formating import to_tensor
import json
import numpy as np
from mmdet3d.structures import LiDARInstance3DBoxes, DepthInstance3DBoxes
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes
import torch

@TRANSFORMS.register_module()
class Pack3DDetAndMapInputs(Pack3DDetInputs):
    def __init__(self, shape=(320, 800), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = shape


    def transform(self, results):
        packed = super().transform(results)
        
        # 确保 results 中包含 'maps' 字段
        # assert 'maps' in results, \
        #     "Key 'maps' is missing in results. Please check the pipeline to ensure LoadMapsFromFiles is used."

        # # 添加 maps 到 inputs 中
        # packed['data_samples'].maps = to_tensor(results['maps'])
        # packed['data_samples'].gt_map = to_tensor(results['gt_map'])
        packed['data_samples'].set_metainfo({'maps': to_tensor(results['maps'])})
        packed['data_samples'].set_metainfo({'gt_map': to_tensor(results['gt_map'])})
        if 'img_shape' in results and 'pad_shape' in results:
            packed['data_samples'].set_metainfo({'img_shape': self.shape})
            packed['data_samples'].set_metainfo({'pad_shape': self.shape})

        if 'lidar2ego' in results['lidar_points']:
            lidar2ego = to_tensor(results['lidar_points']['lidar2ego'])
            ego2global = to_tensor(results['ego2global'])
            lidar2global = ego2global @ lidar2ego
            packed['data_samples'].set_metainfo({'lidar2ego': lidar2ego})
            # packed['data_samples'].ego2global = ego2global
            packed['data_samples'].set_metainfo({'lidar2global': lidar2global})
        if 'timestamp' in results:
            packed['data_samples'].set_metainfo({'timestamp': torch.as_tensor((results['timestamp']))})
        if 'instance_inds' in results:
            packed['data_samples'].set_metainfo({'instance_inds': to_tensor(results['instance_inds'])})
        if 'token' in results:
            packed['data_samples'].set_metainfo({'token': results['token']})
        if 'scene_token' in results:
            packed['data_samples'].set_metainfo({'scene_token': results['scene_token']})
        return packed


@TRANSFORMS.register_module()
class TrackletRangeFilter(BaseTransform):
    """Filter objects by the range.
        The difference between TrackletRangeFilter and ObjectRangeFilter is that TrackletRangeFilter add instance_inds filter.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        instance_inds = input_dict['instance_inds']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(bool)]
        instance_inds = instance_inds[mask.numpy().astype(bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        input_dict['instance_inds'] = instance_inds

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str

@TRANSFORMS.register_module()
class NormalizeMultiviewImage(BaseTransform):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str