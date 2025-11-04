# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/DQTrack/blob/main/LICENSE
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
import torch
import torch.nn.functional as F

from mmdet.models.task_modules import BaseBBoxCoder
from mmdet3d.registry import TASK_UTILS
from projects.PETR.core.bbox.util import denormalize_bbox


@TASK_UTILS.register_module()
class DETRTrack3DCoder(BaseBBoxCoder):
    """Bbox coder for DETR3D.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=None):
        
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, track_scores, obj_idxes, num_classes, eval_det_only):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].

        Returns:
            list[dict]: Decoded boxes.
        """
        if eval_det_only:  # use NMSFreeCoder
            max_num = self.max_num

            cls_scores = cls_scores.sigmoid()
            scores, indexes = cls_scores.view(-1).topk(max_num)
            labels = indexes % num_classes
            bbox_index = indexes // num_classes
            bbox_preds = bbox_preds[bbox_index]

            final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
            final_scores = scores
            final_preds = labels
            track_scores = track_scores[bbox_index]
            obj_idxes = obj_idxes[bbox_index]

        else:          # use DETRTrack3DCoder
            max_num = min(cls_scores.size(0), self.max_num)
            cls_scores = cls_scores.sigmoid()
            _, indexs = cls_scores.max(dim=-1)
            labels = indexs % num_classes

            _, bbox_index = track_scores.topk(max_num)
            
            labels = labels[bbox_index]
            bbox_preds = bbox_preds[bbox_index]
            track_scores = track_scores[bbox_index]
            obj_idxes = obj_idxes[bbox_index]

            scores = track_scores
            
            final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
            final_scores = track_scores
            final_preds = labels


        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            track_scores = track_scores[mask]
            obj_idxes = obj_idxes[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'track_scores': track_scores,
                'obj_idxes': obj_idxes,
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts, eval_det_only=False):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
                Note: before sigmoid!
            bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].

        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['cls_scores']
        all_bbox_preds = preds_dicts['bbox_preds']
        track_scores = preds_dicts['track_scores']
        obj_idxes = preds_dicts['obj_idxes']
        num_classes = preds_dicts['num_classes']        
        predictions_list = []
        predictions_list.append(self.decode_single(
            all_cls_scores, all_bbox_preds,
            track_scores, obj_idxes, num_classes, eval_det_only))
        
        return predictions_list
