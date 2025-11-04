# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch
from mmengine.structures import InstanceData

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox3d2result
from .grid_mask import GridMask
import copy
import numpy as np
import matplotlib.pyplot as plt

def IOU (intputs, targets, eps=1e-6):
    intputs = intputs.bool()
    targets = targets.bool()
    inter = (intputs & targets).sum(-1)
    union = (intputs | targets).sum(-1)
    # iou = (numerator + eps) / (denominator + eps - numerator)
    return inter.cpu(),union.cpu()

@MODELS.register_module()
class PETR(MVXTwoStageDetector):
    """PETR."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 pts_seg_head=None,
                 pts_det_seg_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None,
                 **kwargs):
        super(PETR,
              self).__init__(pts_voxel_layer, pts_middle_encoder,
                             pts_fusion_layer, img_backbone, pts_backbone,
                             img_neck, pts_neck, pts_bbox_head, img_roi_head,
                             img_rpn_head, train_cfg, test_cfg, init_cfg,
                             data_preprocessor)
        self.train_cfg = train_cfg
        if pts_bbox_head is not None:
            self.pts_bbox_head = MODELS.build(pts_bbox_head)
        else:
            self.pts_bbox_head = None

        if pts_seg_head is not None:
            self.pts_seg_head = MODELS.build(pts_seg_head)
        else:
            self.pts_seg_head = None

        if pts_det_seg_head is not None:
            self.pts_det_seg_head = MODELS.build(pts_det_seg_head)
        else:
            self.pts_det_seg_head = None

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        # # Freeze all parameters
        # for param in self.parameters():
        #     param.requires_grad = False
        # # Unfreeze segmentation head
        # if self.pts_seg_head is not None:
        #     for param in self.pts_seg_head.parameters():
        #         param.requires_grad = True

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    # @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          maps,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        losses = {}
        if self.pts_bbox_head is not None:
            outs_bbox = self.pts_bbox_head(pts_feats, img_metas)
            loss_inputs_bbox = [gt_bboxes_3d, gt_labels_3d, outs_bbox]
            losses_bbox = self.pts_bbox_head.loss_by_feat(*loss_inputs_bbox)
            losses.update(losses_bbox)

        if self.pts_seg_head is not None:
            outs_seg = self.pts_seg_head(pts_feats, img_metas)
            loss_inputs_seg = [gt_bboxes_3d, gt_labels_3d, outs_seg, maps]
            losses_seg = self.pts_seg_head.loss_by_feat(*loss_inputs_seg)
            losses.update(losses_seg)
        
        if self.pts_det_seg_head is not None:
            outs_det_seg = self.pts_det_seg_head(pts_feats, img_metas)
            loss_inputs_det_seg = [gt_bboxes_3d, gt_labels_3d, outs_det_seg, maps]
            losses_det_seg = self.pts_det_seg_head.loss_by_feat(*loss_inputs_det_seg)
            losses.update(losses_det_seg)
        return losses

    def _forward(self, mode='loss', **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        raise NotImplementedError('tensor mode is yet to add')

    def loss(self,
             inputs=None,
             data_samples=None,
             mode=None,
             points=None,
             img_metas=None,
             gt_bboxes_3d=None,
             gt_labels_3d=None,
             gt_labels=None,
             gt_bboxes=None,
             img=None,
             proposals=None,
             gt_bboxes_ignore=None,
             img_depth=None,
             img_mask=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        img = inputs['imgs']
        batch_img_metas = [ds.metainfo for ds in data_samples]
        batch_gt_instances_3d = [ds.gt_instances_3d for ds in data_samples]
        maps = [gt.maps for gt in data_samples]
        gt_bboxes_3d = [gt.bboxes_3d for gt in batch_gt_instances_3d]
        gt_labels_3d = [gt.labels_3d for gt in batch_gt_instances_3d]
        gt_bboxes_ignore = None

        batch_img_metas = self.add_lidar2img(img, batch_img_metas)

        img_feats = self.extract_feat(img=img, img_metas=batch_img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, maps, batch_img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def predict(self, inputs=None, data_samples=None, mode=None, **kwargs):
        img = inputs['imgs']
        batch_img_metas = [ds.metainfo for ds in data_samples]
        maps = [gt.maps for gt in data_samples]
        gt_map = [gt.gt_map for gt in data_samples]
        for var, name in [(batch_img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        batch_img_metas = self.add_lidar2img(img, batch_img_metas)

        results_list_3d = self.simple_test(batch_img_metas, gt_map, img, maps, **kwargs)

        for i, data_sample in enumerate(data_samples):
            results_list_3d_i = InstanceData(
                metainfo=results_list_3d[i]['pts_bbox'] if 'pts_bbox' in results_list_3d[i] else None)
            data_sample.pred_instances_3d = results_list_3d_i
            data_sample.pred_instances = InstanceData()
            data_sample.ret_iou = results_list_3d[i]['ret_iou']

        return data_samples

    def simple_test_pts(self, x, img_metas, gt_map, maps, rescale=False):
        """Test function of point cloud branch."""
        if self.pts_bbox_head:
            outs_bbox = self.pts_bbox_head(x, img_metas)
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs_bbox, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ] if self.pts_bbox_head else []

        outs_seg = self.pts_seg_head(x, img_metas)
        lane_preds=outs_seg['all_lane_preds'][5]    # 取最后一个decoder layer的输出
        # gt_maps = [gt.reshape(3, 200, 200) for gt in gt_map]
        gt_maps = gt_map
        maps = [gt.reshape(3, 200, 200) for gt in maps]

        with torch.no_grad():
            # TODO(Itachi): 把下面可视化代码封装起来
            batch_size = len(lane_preds)
            ret_ious_batch=[]
            for i in range(batch_size):
                lane_pred=lane_preds[i] #[B,N,H,W]    一个batch保留一张图片可视化
                
                # n,w=lane_preds.size()
                
                f_lane=lane_pred.sigmoid()
                f_lane[f_lane>=0.43]=1
                f_lane[f_lane<0.43]=0
                f_lane_show=copy.deepcopy(f_lane).reshape(3,200,200)
                gt_map_show=copy.deepcopy(maps[i])
                
                f_lane=f_lane.view(3,-1)  # [3, H*W]
                f_lane=1-f_lane
                gt_map=gt_maps[i].reshape(3,-1).to(x.device)  # [3, H*W]
                
                        
                inter,union=IOU(f_lane,gt_map)
                # ret_iou=inter/union
                # ret_iou=[ret_iou[0].item(),ret_iou[1].item(),ret_iou[2].item()]
                ret_ious=[inter,union]
                ret_ious_batch.append(ret_ious)


                show_res=False
                if show_res:
                    gt = 1 - gt_map_show
                    gt = (gt * 255).cpu().numpy().astype(np.uint8)
                    gt = gt.transpose(1, 2, 0)

                    pre = 1 - f_lane_show
                    pre = (pre * 255).cpu().numpy().astype(np.uint8)
                    pre = pre.transpose(1, 2, 0)

                    # 创建一个大图：2行4列，用于展示 GT 和 PRE 的对比
                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

                    titles = ['Combined', 'Drive', 'Lane', 'Vehicle']

                    # 第一行：Ground Truth
                    axes[0, 0].imshow(gt)
                    axes[0, 0].set_title("GT: Combined")
                    axes[0, 0].axis('off')

                    for i in range(3):
                        axes[0, i+1].imshow(gt[:, :, i], cmap='gray')
                        axes[0, i+1].set_title(f"GT: {titles[i+1]}")
                        axes[0, i+1].axis('off')

                    # 第二行：Prediction
                    axes[1, 0].imshow(pre)
                    axes[1, 0].set_title("Pre: Combined")
                    axes[1, 0].axis('off')

                    for i in range(3):
                        axes[1, i+1].imshow(pre[:, :, i], cmap='gray')
                        axes[1, i+1].set_title(f"Pre: {titles[i+1]}")
                        axes[1, i+1].axis('off')

                    plt.tight_layout()
                    save_path = './res/' + img_metas[0]['lidar_path'].split('/')[-1].split('.')[0] + '_compare.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    print(f"Saved comparison visualization to {save_path}")

        # 构造返回结果，包含检测 + map 分割结果
        results = {
            'bbox_results': bbox_results,  
            'iou_results': ret_ious_batch,  # 转换为 numpy 或 tensor 用于后续评估
        }
        return results

    def simple_test(self, img_metas, gt_map=None, img=None, maps=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # NOTE(Itachi): 创建一个长度为batch_size的列表bbox_list，列表元素为dict
        bbox_list = [dict() for i in range(len(img_metas))]
        results = self.simple_test_pts(img_feats, img_metas, gt_map, maps, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, results['bbox_results']):
            result_dict['pts_bbox'] = pts_bbox
        for result_dict, ret_iou in zip(bbox_list, results['iou_results']):
            result_dict['ret_iou'] = ret_iou
        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    # may need speed-up
    def add_lidar2img(self, img, batch_input_metas):
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for meta in batch_input_metas:
            lidar2img_rts = []
            # obtain lidar to image transformation matrix
            for i in range(len(meta['cam2img'])):
                lidar2cam_rt = torch.tensor(meta['lidar2cam'][i]).double()
                intrinsic = torch.tensor(meta['cam2img'][i]).double()
                viewpad = torch.eye(4).double()
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                # The extrinsics mean the transformation from lidar to camera.
                # If anyone want to use the extrinsics as sensor to lidar,
                # please use np.linalg.inv(lidar2cam_rt.T)
                # and modify the ResizeCropFlipImage
                # and LoadMultiViewImageFromMultiSweepsFiles.
                lidar2img_rts.append(lidar2img_rt)
            meta['lidar2img'] = lidar2img_rts
            img_shape = meta['img_shape'][:3]
            meta['img_shape'] = [img_shape] * len(img[0])

        return batch_input_metas
