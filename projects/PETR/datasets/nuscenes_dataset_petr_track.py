# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
from typing import Callable, List, Union

import numpy as np
import torch
import json

from mmengine.fileio import load
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData



@DATASETS.register_module()
class NuScenesDatasetPETRTrack(Det3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
                to convert to the FOV-based data type to support image-based
                detector.
            - 'fov_image_based': Only load the instances inside the default
                cam, and need to convert to the FOV-based data type to support
                image-based detector.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        with_velocity (bool): Whether to include velocity prediction
            into the experiments. Defaults to True.
        use_valid_flag (bool): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
    """
    METAINFO = {
        'classes':
        ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
        'version':
        'v1.0-trainval',
        'palette': [
            (255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 127, 80),  # Coral
            (233, 150, 70),  # Darksalmon
            (220, 20, 60),  # Crimson
            (255, 61, 99),  # Red
            (0, 0, 230),  # Blue
            (47, 79, 79),  # Darkslategrey
            (112, 128, 144),  # Slategrey
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 box_type_3d: str = 'LiDAR',
                 load_type: str = 'frame_based',
                 modality: dict = dict(
                     use_camera=False,
                     use_lidar=True,
                 ),
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 with_velocity: bool = True,
                 use_valid_flag: bool = False,
                 load_interval=1,
                 sample_mode='fixed_interval',
                 sample_interval=1,
                 force_continuous=False,
                 num_frames_per_sample=3,
                 **kwargs) -> None:
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.sample_mode = sample_mode
        self.sample_interval = sample_interval
        self.force_continuous = force_continuous
        self.num_frames_per_sample = num_frames_per_sample
        self.with_velocity = with_velocity

        # TODO: Redesign multi-view data process in the future
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type

        assert box_type_3d.lower() in ('lidar', 'camera')
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            modality=modality,
            pipeline=pipeline,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        self.num_samples = len(self.data_address) - (self.num_frames_per_sample - 1) * \
            self.sample_interval
        
        return self.num_samples

    def _filter_with_mask(self, ann_info: dict) -> dict:
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos.

        Returns:
            dict: Annotations after filtering.
        """
        filtered_annotations = {}
        if self.use_valid_flag:
            filter_mask = ann_info['bbox_3d_isvalid']
        else:
            filter_mask = ann_info['num_lidar_pts'] > 0

        for key in ann_info.keys():
            if key != 'instances':
                filtered_annotations[key] = ann_info[key][filter_mask]
            else:
                filtered_annotations[key] = ann_info[key]
        return filtered_annotations

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        metainfo = annotations['metainfo']
        raw_data_list = sorted(annotations['data_list'], key=lambda e: e['timestamp'])

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another()
                continue
            return data

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_sample - 1) * sample_interval + 1, sample_interval
        return default_range
        
    def pack_ret(self, ret: dict):
        assert 'data_samples' and 'inputs' in ret, \
            "Keys 'data_samples' and 'inputs' are missing in ret. Please check the pipeline"

        data_samples = Det3DDataSample()
        metainfo_keys = ret['data_samples'][0].metainfo.keys()
        metainfo_dict = {}
        for key in metainfo_keys:
            metainfo_dict[key] = [ds.metainfo[key] for ds in ret['data_samples']]
        data_samples.set_metainfo(metainfo_dict)

        gt_instances_3d = InstanceData()
        if not self.test_mode:
            keys = ['labels_3d', 'bboxes_3d']
            for key in keys:
                gt_instances_3d[key] = [getattr(ds, 'gt_instances_3d', None)[key] for ds in ret['data_samples']]
            data_samples.gt_instances_3d = gt_instances_3d
        else:
            keys = ['gt_labels_3d', 'gt_bboxes_3d']
            for key in keys:
                gt_instances_3d[key] = [getattr(ds, 'eval_ann_info', None)[key] for ds in ret['data_samples']]
            data_samples.gt_instances_3d = gt_instances_3d


        img_tensors = [item['img'] for item in ret['inputs']]
        inputs = {'img': torch.stack(img_tensors, dim=0)}


        packed_results = dict()
        packed_results['data_samples'] = data_samples
        packed_results['inputs'] = inputs
        return packed_results

    def prepare_train_data(self, index):
        start, end, interval = self._get_sample_range(index)
        # print(f'start:{start}, end:{end}')

        # force to use continuous frame in the same scene
        if self.force_continuous and 'scene_token' in self.get_data_info(start):
            reassign = False
            try:
                if end > len(self.data_address):
                    reassign = True
                elif self.get_data_info(start)['scene_token'] != \
                self.get_data_info(end)['scene_token']:
                    reassign = True
            except:
                print("END:{}, Total NUM:{}".format(end, len(self.data_address)))
            
            if reassign:
               # reassign frame index
               frame_len = end - start
               start = start - frame_len
               end = start + frame_len

        ret = None
        for i in range(start, end, interval):
            data_i = self.prepare_train_data_single(i)
            if data_i is None:
                return None

            if ret is None:
                ret = {key: [] for key in data_i.keys()}

            for key, value in data_i.items():
                ret[key].append(value)
        ret = self.pack_ret(ret)
        return ret

    def prepare_train_data_single(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        example = self.pipeline(input_dict)

        if self.filter_empty_gt and (example is None or 
                                        ~(example['data_samples'].gt_instances_3d.labels_3d != -1).any()):
            return None

        return example

    def prepare_test_data_single(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        start, end, interval = self._get_sample_range(index)

        ret = None
        for i in range(start, end, interval):
            data_i = self.prepare_test_data_single(i)

            if ret is None:
                ret = {key: [] for key in data_i.keys()}

            for key, value in data_i.items():
                ret[key].append(value)
        ret = self.pack_ret(ret)
        return ret

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        input_dict = super().get_data_info(idx)
        if self.test_mode:
            path = '/data1051n/algorithm/home/zhaoqh/petr_extr_data/instance_inds_val.json'
        else:
            path = '/data1051n/algorithm/home/zhaoqh/petr_extr_data/instance_inds_train.json'
        with open(path, 'r') as f:
            instance_data_list = json.load(f)
        # 根据 token 查找对应的 instance_inds
        token = input_dict['token']
        instance_inds = None
        
        # 遍历列表查找匹配的 token
        for item in instance_data_list:
            if item['token'] == token:
                if self.use_valid_flag:
                    filter_mask = np.array(item['valid_flag'], dtype=bool)
                else:
                    filter_mask = input_dict['ann_info']['num_lidar_pts'] > 0
                instance_inds = np.array(item['instance_inds'])
                # if not self.test_mode:
                instance_inds = instance_inds[filter_mask]
                scene_token = item['scene_token']
                break
        if 'ann_info' not in input_dict:
            ann_info = input_dict['eval_ann_info']
        else:
            ann_info = input_dict['ann_info']

        for item in ann_info:
            if item != 'instances':
                ann_info[item] = ann_info[item][filter_mask]
        input_dict.update(
            instance_inds=instance_inds,
            ann_info=ann_info,
            scene_token=scene_token
        )

        return input_dict

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is not None:

            # ann_info = self._filter_with_mask(ann_info)

            if self.with_velocity:
                gt_bboxes_3d = ann_info['gt_bboxes_3d']
                gt_velocities = ann_info['velocities']
                nan_mask = np.isnan(gt_velocities[:, 0])
                gt_velocities[nan_mask] = [0.0, 0.0]
                gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocities],
                                              axis=-1)
                ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        else:
            # empty instance
            ann_info = dict()
            if self.with_velocity:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            else:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['attr_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # TODO: Unify the coordinates
        if self.load_type in ['fov_image_based', 'mv_image_based']:
            gt_bboxes_3d = CameraInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5))
        else:
            gt_bboxes_3d = LiDARInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        return ann_info

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.load_type == 'mv_image_based':
            data_list = []
            if self.modality['use_lidar']:
                info['lidar_points']['lidar_path'] = \
                    osp.join(
                        self.data_prefix.get('pts', ''),
                        info['lidar_points']['lidar_path'])

            if self.modality['use_camera']:
                for cam_id, img_info in info['images'].items():
                    if 'img_path' in img_info:
                        if cam_id in self.data_prefix:
                            cam_prefix = self.data_prefix[cam_id]
                        else:
                            cam_prefix = self.data_prefix.get('img', '')
                        img_info['img_path'] = osp.join(
                            cam_prefix, img_info['img_path'])

            for idx, (cam_id, img_info) in enumerate(info['images'].items()):
                camera_info = dict()
                camera_info['images'] = dict()
                camera_info['images'][cam_id] = img_info
                if 'cam_instances' in info and cam_id in info['cam_instances']:
                    camera_info['instances'] = info['cam_instances'][cam_id]
                else:
                    camera_info['instances'] = []
                # TODO: check whether to change sample_idx for 6 cameras
                #  in one frame
                camera_info['sample_idx'] = info['sample_idx'] * 6 + idx
                camera_info['token'] = info['token']
                camera_info['ego2global'] = info['ego2global']

                if not self.test_mode:
                    # used in traing
                    camera_info['ann_info'] = self.parse_ann_info(camera_info)
                if self.test_mode and self.load_eval_anns:
                    camera_info['eval_ann_info'] = \
                        self.parse_ann_info(camera_info)
                data_list.append(camera_info)
            return data_list
        else:
            data_info = super().parse_data_info(info)
            return data_info
