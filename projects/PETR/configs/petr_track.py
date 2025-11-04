_base_ = [
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/cyclic-20e.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
custom_imports = dict(imports=['projects.PETR.petr', 'projects.PETR.datasets', 'projects.PETR.core.bbox.coders'], allow_failed_imports=False)
backend_args = None

randomness = dict(seed=1, deterministic=False, diff_rank_seed=False)
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
bev_stride = 4
track_frame = 3
fp16_enabled = True

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# metainfo = dict(classes=class_names)

track_names = [
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian',
]

velocity_error = {
  'car':6,
  'truck':6,
  'bus':6,
  'trailer':5,
  'pedestrian':5,
  'motorcycle':12,
  'bicycle':5,  
}
velocity_error = [velocity_error[_name] for _name in track_names]
eval_det_only = True

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='PETRTrackerDQ',
    # data_preprocessor=dict(
    #     type='Det3DDataPreprocessor',
    #     mean=[103.530, 116.280, 123.675],
    #     std=[57.375, 57.120, 58.395],
    #     bgr_to_rgb=False,
    #     pad_size_divisor=32),
    # not use grid mask
    use_grid_mask=True,
    tracker_cfg=dict(
        track_frame=track_frame,
        num_track=300,
        num_query=900,
        num_cams=6,
        embed_dims=256,
        ema_decay=0.5,
        train_det_only=False,
        train_track_only=True,
        class_dict=dict(
            all_classes=class_names,
            track_classes=track_names,
        ),
        pos_cfg=dict(
            pos_trans='ffn',
            fuse_type="sum",
            final_trans="linear",
        ),
        query_trans=dict(
            with_att=True,
            with_pos=True,
            min_channels=256,
            drop_rate=0.0,
        ),
        track_aug=dict(
            drop_prob=0,
            fp_ratio=0.2,
        ),
        # used for training
        ema_drop=0.0,
        # used for inference
        class_spec = True,
        eval_det_only=eval_det_only,        
        velo_error=velocity_error, 
        assign_method='hungarian',
        det_thres=0.3,  # original 0.3
        new_born_thres=0.2,
        asso_thres=0.1,
        miss_thres=7,
    ),
    img_backbone=dict(
        type='VoVNetCP',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=(
            'stage4',
            'stage5',
        )),
    img_neck=dict(
        type='CPFPN', in_channels=[768, 1024], out_channels=256, num_outs=2),
    pts_bbox_head=dict(
        type='PETRHead',
        num_classes=10,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0)),
    pts_seg_head=dict(     # NOTE: not used in training and inference, only for exporting to pt2
        type='PETRHead_seg',
        num_classes=10,
        in_channels=256,
        num_lane=625,
        # blocks=[128,128,64],
        blocks=[256,256,128],
        LID=True,
        with_position=True,
        with_multiview=True,
        with_se=True,
        with_time=False,
        with_multi=False,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        normedlinear=False,
        transformer_lane=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            # type='NMSFreeClsCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_dri=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_lan=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.5,
            loss_weight=4.0),
        loss_veh=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.5,
            loss_weight=8.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0),
        loss_lane_mask=dict(type='mmdet.Sigmoid_ce_loss', loss_weight=1.0)
        ),
    bbox_coder=dict(
        type='DETRTrack3DCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range,
        max_num=100), 
    loss_cfg=dict(
        type='DualQueryMatcher',
        num_classes=10,
        class_dict=dict(
            all_classes=class_names,
            track_classes=track_names),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost.
            pc_range=point_cloud_range),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_asso=dict(
            type='mmdet.CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0),
        loss_me=dict(
            type="CategoricalCrossEntropyLoss",
            loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(
                    type='IoUCost', weight=0.0
                ),  # Fake cost. Just to be compatible with DETR head.
                pc_range=point_cloud_range))))

ida_aug_conf = {
        "resize_lim": (0.47, 0.625),
        "final_dim": (256, 704),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMapsFromFiles_flattenf200f3'),
    # dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, test_mode=False, sweep_range=[3,27]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='TrackletRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='Pack3DDetAndMapInputs',
        shape = ida_aug_conf['final_dim'],
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMapsFromFiles_flattenf200f3'),
    # dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, sweep_range=[3,27]),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='Pack3DDetAndMapInputs',
        shape = ida_aug_conf['final_dim'],
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ])
]

dataset_type = 'NuScenesDatasetPETRTrack'
data_root = 'data/nuscenes/'
info_root = 'data/infos/'
metainfo = dict(classes=class_names)
file_client_args = dict(backend='disk')
train_dataloader = dict(
    batch_size=1,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        # data_root=data_root,
        # ann_file='mmdet3d_nuscenes_30f_infos_train.pkl',
        num_frames_per_sample=track_frame,  # number of frames for each 
        pipeline=train_pipeline,
        # classes=class_names,
        # track_classes=track_names,
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=False,
        force_continuous=True, # force to use continuous frame
        modality=input_modality,
        use_valid_flag=True,))
test_dataloader = dict(
    batch_size=1,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        # data_root=data_root,
        # ann_file='mmdet3d_nuscenes_30f_infos_train.pkl',
        num_frames_per_sample=1,  # number of frames for each 
        pipeline=test_pipeline,
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        box_type_3d='LiDAR',
        metainfo=metainfo,
        # track_classes=track_names,
        test_mode=True,
        modality=input_modality,
        use_valid_flag=True))
val_dataloader = test_dataloader

val_evaluator = dict(
    type='NuScenesCustomMetric',
    data_root=data_root,
    ann_file=data_root + 'PETR_nuscenes_infos_val.pkl',
    metric='bbox',
    track_classes=track_names,
    eval_det_only=eval_det_only,
    backend_args=backend_args)
test_evaluator = val_evaluator


optim_wrapper = dict(
    # TODO Add Amp
    # type='AmpOptimWrapper',
    # loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
    }),
    clip_grad=dict(max_norm=35, norm_type=2))

num_epochs = 5

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        begin=0,
        end=500,
        by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        # TODO Figure out what T_max
        T_max=num_epochs,
        by_epoch=True,
    )
]

train_cfg = dict(max_epochs=num_epochs, val_interval=num_epochs)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5),
)
find_unused_parameters = True

# load_from = '/data1/private-data/zhaoqh/PETR_work_dirs/ckpts/petr_vovnet_gridmask_p4_800x320-e2191752.pth'
load_from = '/home/zhaoqh614/mmdetection3d/PETR_work_dirs/petr_vovnet_gridmask_p4_800x320_resume_segmap/epoch_26.pth'
load_from = '/home/zhaoqh614/mmdetection3d/ckpt/fcos3d_vovnet_imgbackbone-remapped.pth'
# work_dir = '/data1/private-data/zhaoqh/PETR_work_dirs/petr_track_wogridmask_finetune_from_det_only'
