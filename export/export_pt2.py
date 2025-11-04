import argparse
import os 

import torch
import torch.export._trace
# avoid `deprecated_api_warning` error during export
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
MultiheadAttention.forward = MultiheadAttention.forward.__wrapped__
FFN.forward = FFN.forward.__wrapped__
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.models.layers.transformer import inverse_sigmoid
from torch.export import Dim

_DIM_LIMIT = 4096

def forward_npu1(det_head, seg_head, x, coords3d, sin_embed, pos2posemb3d_ref,
                    map_pos2posemb2d_ref,
                    reference, map_reference):
    x = det_head.input_proj(x)
    # -------------------- subnpu2:[position_encoder + adapt_pos3d] -------------------------
    pos_embed = det_head.position_encoder(coords3d) # torch.Size([6, 256, 20, 50]
    sin_embed = det_head.adapt_pos3d(sin_embed) # [6, 256, 20, 50]
    pos_embed = pos_embed + sin_embed
    # -------------------- subnpu3:[query_embedding, det_transformer / map_transformer] -----
    # generate query_embeds
    query_embeds = det_head.query_embedding(pos2posemb3d_ref) # [900, 384] -> [900, 256]
    map_query_embeds = seg_head.query_embedding_lane(map_pos2posemb2d_ref) # [625, 384] -> [625, 256]  

    # det petr transformer
    bn, c, h, w = x.shape # [6, 256, 20, 50])
    bs = 1
    memory = x.permute(0, 2, 3, 1).reshape(-1, c).unsqueeze(dim=1) # [n*h*w, bs, c]
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, c).unsqueeze(dim=1)  # [n*h*w, bs, c]
    query_embeds = query_embeds.unsqueeze(1).repeat(
        1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
    mask = torch.zeros(1, pos_embed.shape[0], dtype=torch.bool)   # [bs, n*h*w] 

    # Since 408 has operator dim limit:4096, the pixels out of this limit 
    # will be dropped. Note that this limitation is only for detection task,
    memory = memory[:_DIM_LIMIT, ...]
    pos_embed = pos_embed[:_DIM_LIMIT, ...]
    mask = mask[:, :_DIM_LIMIT]

    target = torch.zeros_like(query_embeds)
    outs_dec = det_head.transformer.decoder(            
        query=target,
        key=memory,
        value=memory,
        key_pos=pos_embed,
        query_pos=query_embeds,
        key_padding_mask=mask,
        reg_branch=None,
    )
    outs_dec = outs_dec.transpose(1, 2) # torch.Size([6, 1, 900, 256])

    # map petr transformer
    map_query_embeds = map_query_embeds.unsqueeze(1).repeat(
        1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
    map_target = torch.zeros_like(map_query_embeds)
    map_outs_dec = seg_head.transformer_lane.decoder(            
        query=map_target,
        key=memory,
        value=memory,
        key_pos=pos_embed,
        query_pos=map_query_embeds,
        key_padding_mask=mask,
        reg_branch=None,
    ) 
    map_outs_dec = map_outs_dec.transpose(1, 2) # torch.Size([6, 1, 625, 256])

    # -------------------- subnpu4:[ det & map: cls+reg_branches]  ------------------------
    # 3D detection head
    outputs_classes = []
    outputs_coords = []

    for lvl in range(outs_dec.shape[0]):
        # reference = inverse_sigmoid(reference_points.clone())
        assert reference.shape[-1] == 3
        outputs_class = det_head.cls_branches[lvl](outs_dec[lvl]).to(
            torch.float32)
        tmp = det_head.reg_branches[lvl](outs_dec[lvl]).to(torch.float32) 
        tmp = tmp.clone()
        tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
        tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
        tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

        ref_points = torch.cat([tmp[...,0:2],tmp[...,4:5]], dim=-1)

        outputs_coord = tmp
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)

    last_query_feat = outs_dec[-1]
    # last_ref_points = inverse_sigmoid(ref_points)
    last_ref_points = ref_points   # NOTE: no inverse_sigmoid here, move inverse_sigmoid out of this graph

    all_cls_scores = torch.stack(outputs_classes)
    all_bbox_preds = torch.stack(outputs_coords)
    all_bbox_preds[..., 0:1] = (
        all_bbox_preds[..., 0:1] * (det_head.pc_range[3] - det_head.pc_range[0]) +
        det_head.pc_range[0])
    all_bbox_preds[..., 1:2] = (
        all_bbox_preds[..., 1:2] * (det_head.pc_range[4] - det_head.pc_range[1]) +
        det_head.pc_range[1])
    all_bbox_preds[..., 4:5] = (
        all_bbox_preds[..., 4:5] * (det_head.pc_range[5] - det_head.pc_range[2]) +
        det_head.pc_range[2])

    # map seg head
    outputs_lanes=[]
    for lvl in range(map_outs_dec.shape[0]):
        lane_queries_lvl=map_outs_dec[lvl].reshape(1,25,25,-1).permute(0,3,1,2).contiguous()
        outputs_dri=seg_head.lane_branches_dri[lvl](lane_queries_lvl)
        outputs_lan=seg_head.lane_branches_lan[lvl](lane_queries_lvl)
        outputs_vie=seg_head.lane_branches_vie[lvl](lane_queries_lvl)
        
        outputs_lane=torch.cat([outputs_dri,outputs_lan,outputs_vie],dim=1)
        # outputs_lane=outputs_lane.view(-1,3,200*200)
        
        outputs_lanes.append(outputs_lane)

    
    # all_lane_preds=torch.stack(outputs_lanes)  # [6, 1, 3, 200, 200]

    return all_cls_scores, all_bbox_preds, last_query_feat, last_ref_points, outputs_lane


def extract_img_feat_for_export(model, img):
    img_feats = model.img_backbone(img)
    if isinstance(img_feats, dict):
        img_feats = list(img_feats.values())
    img_feats = model.img_neck(img_feats)
    return img_feats[0]


class HeadWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x, coords3d, sin_embed, pos2posemb3d_ref, map_pos2posemb2d_ref, reference, map_reference):
        # extract img feats
        img_feats = extract_img_feat_for_export(self.model, x)
        # forward det and seg head
        all_cls_scores, all_bbox_preds, last_query_feat, last_ref_points, all_lane_preds = forward_npu1(self.model.pts_bbox_head, 
            self.model.pts_seg_head, img_feats, coords3d, sin_embed, pos2posemb3d_ref, map_pos2posemb2d_ref, reference, map_reference)
        return all_cls_scores, all_bbox_preds, last_query_feat, last_ref_points, all_lane_preds

class TrackAssocWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.detector_trans = model.detector_trans
        self.tracklet_trans = model.tracklet_trans
        self.query_inter = model.query_inter if hasattr(model, 'query_inter') else None
        self.rel_dist_embed = model.rel_dist_embed
        self.embed_trans = model.embed_trans if hasattr(model, 'embed_trans') else None

    def forward(
        self,
        det_output_embedding,  # [num_det, embed_dims]
        track_embedding,           # [num_track, embed_dims]
        rel_dist               # [num_det, num_track, 1]  由CPU预先计算
    ):
        # 外观/交互嵌入
        det_embedding = det_output_embedding
        track_embedding_nb = self.tracklet_trans(det_embedding)      # [num_det, embed_dims]
        obj_embedding = self.detector_trans(det_embedding)           # [num_det, embed_dims]
        if self.query_inter is not None:
            track_embedding_nb, obj_embedding = self.query_inter(track_embedding_nb, obj_embedding, None)
        # NOTE: track_embedding_nb 是new born tracklet的embedding，track_embedding是之前帧已有的tracklet的embedding
        # 运动嵌入（注意：这里不计算 sqrt，只是预计算的 rel_dist）
        motion_emb = self.rel_dist_embed(rel_dist)                   # [num_det, num_track, embed_dims]

        # 融合
        appear_emb = obj_embedding[:, None, :] * track_embedding[None, :, :]  # [num_det, num_track, embed_dims]
        fused = appear_emb + motion_emb

        # 生成匹配矩阵（不做 masked_fill，留到CPU后处理）
        if self.embed_trans is not None:
            det2track_mat = self.embed_trans(fused).sum(-1)          # [num_det, num_track]
        else:
            det2track_mat = fused.sum(-1) / (fused.shape[-1] ** 0.5)

        return det2track_mat, track_embedding_nb


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy PETRMultiTask with pt2')
    parser.add_argument("--config1", default="/home/zhaoqh614/mmdetection3d/projects/PETR/configs/petr_vovnet_gridmask_p4_800x320.py", help="det+seg config file path")
    parser.add_argument("--checkpoint1", default="/home/zhaoqh614/mmdetection3d/PETR_work_dirs/petr_vovnet_gridmask_p4_800x320/epoch_24.pth", help="det+seg checkpoint file")
    parser.add_argument("--config2", default="/home/zhaoqh614/mmdetection3d/projects/PETR/configs/petr_track.py", help="tracking config file path")
    parser.add_argument("--checkpoint2", default="/home/zhaoqh614/mmdetection3d/PETR_work_dirs/petr_track_wogridmask_finetune_from_det_seg/epoch_5.pth", help="tracking checkpoint file")
    parser.add_argument("--work_dir", default="/home/zhaoqh614/mmdetection3d/PETR_work_dirs/pt2_fp16", help="work dir to save file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    #----------- build model 1: det+seg head -----------
    cfg1 = Config.fromfile(args.config1)
    cfg1.load_from = args.checkpoint1
    cfg1.work_dir = args.work_dir
    runner1 = Runner.from_cfg(cfg1)

    runner1.model.eval() 
    runner1.load_or_resume()
    runner1.model.to('cpu')

    #----------- build model 2: tracking module -----------
    cfg2 = Config.fromfile(args.config2)
    cfg2.load_from = args.checkpoint2
    cfg2.work_dir = args.work_dir
    runner2 = Runner.from_cfg(cfg2)

    runner2.model.eval() 
    runner2.load_or_resume()
    runner2.model.to('cpu')

    # --------- export det and seg head ---------
    BN, C, H, W = 6, 3, 256, 704
    img = torch.zeros(BN, C, H, W).to('cpu')
    NUM_QUERY = 900
    NUM_LANE_QUERY = 625
    NUM_CAMS = 6
    H_FAET, W_FEAT = 16, 44
    COORDS3D_DIM = 192

    coords3d = torch.zeros(NUM_CAMS, COORDS3D_DIM, H_FAET, W_FEAT).to('cpu')
    sin_embed = torch.zeros(BN, 384, H_FAET, W_FEAT).to('cpu') # 384 = 2 * 192
    reference = torch.zeros(1, NUM_QUERY, 3).to('cpu')
    map_reference = torch.zeros(1, NUM_LANE_QUERY, 2).to('cpu')
    pos2posemb3d_ref = torch.zeros(NUM_QUERY, 384).to('cpu') 
    map_pos2posemb2d_ref = torch.zeros(NUM_LANE_QUERY, 256).to('cpu')


    head = HeadWrapper(runner1.model).eval().to('cpu')
    ep = torch.export.export(
        head,
        args=(img, coords3d, sin_embed, pos2posemb3d_ref, map_pos2posemb2d_ref, reference, map_reference)
    )
    torch.export.save(ep, "/home/zhaoqh614/mmdetection3d/PETR_work_dirs/pt2_fp16/pt2e/det_seg_head.pt2")   
    print("finished export det and seg head...")

    # ---------- export tracking assoc core ----------
    num_det = runner2.model.tracker_cfg.get('num_query', 300)
    num_track = runner2.model.tracker_cfg.num_track
    embed_dims = runner2.model.tracker_cfg.embed_dims

    N_DET = Dim("num_det", min=1, max=num_det)   # 这里为有效检测数量
    N_TRACK = Dim("num_track", min=1, max=num_track)  # 这里为有效tracklet数量
    det_output_embedding = torch.zeros(num_det, embed_dims, device='cpu')
    track_embedding = torch.zeros(num_track, embed_dims, device='cpu')
    rel_dist = torch.zeros(num_det, num_track, 1, device='cpu')  # CPU 预计算后的输入占位

    track_core = TrackAssocWrapper(runner2.model).eval().to('cpu')
    ep = torch.export.export(
        track_core,
        args=(det_output_embedding, track_embedding, rel_dist),
        dynamic_shapes={
            'det_output_embedding': { 0: N_DET},
            'track_embedding': { 0: N_TRACK},
            'rel_dist': { 0: N_DET, 1: N_TRACK}
        }
    )
    torch.export.save(ep, "/home/zhaoqh614/mmdetection3d/PETR_work_dirs/pt2_fp16/pt2e/track_core.pt2")
    print("finished export tracklet assoc core...")


if __name__ == "__main__":
    
    main()