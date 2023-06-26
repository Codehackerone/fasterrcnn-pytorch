# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import argparse
import sys
sys.path.append('.')

import nncore
import torch
import torch.nn.functional as F
from nncore.ops import cosine_similarity
import json
import time

from consnet.api import get_act_and_obj_name, get_act_name, get_obj_name, hoi_idx_to_act_idx, hoi_idx_to_obj_idx, load_anno, pair_iou
from consnet.models import build_dataloader, build_detector, build_embedder

# taken from consnet.api
OBJ_IDX_TO_COCO_ID = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90
]

def get_union_box(human_box, object_box):
    """
        box format - x1, y1, x2, y2
        human_box shape: (mx4)
        object_box shape: (mx4)
    """

    unionbox = human_box.clone()
    unionbox[:, 0] = torch.min(human_box[:, 0], object_box[:, 0])
    unionbox[:, 1] = torch.min(human_box[:, 1], object_box[:, 1])
    unionbox[:, 2] = torch.max(human_box[:, 2], object_box[:, 2])
    unionbox[:, 3] = torch.max(human_box[:, 3], object_box[:, 3])
    return unionbox

## Time required for the below function to run - 
# for train split -> 2523.1878sec or 0.7009hrs
# for test split -> 660.4048sec or 0.1834hrs
def detect_objects(model, data_loader, annos, split):
    start_time = time.time()
    anno, blob = annos[split], dict()

    print(f'detecting objects in *{split}* split')
    prog_bar = nncore.ProgressBar(num_tasks=len(data_loader))

    # looping over the batches (batch size is 1)
    # cnt = 0
    for data in data_loader:
        # cnt += 1
        # see the shape of the data
        # print(data.keys())

        img, img_metas = data['img'][0].data[0], data['img_metas'][0].data[0]
        # print(img.shape) # gives the transformed img shape
    
        data['proposals'], bboxes, img_ids, img_annos = [[]], [], [], []
        # img_annos = list of annotations corresponding to individual img_ids - list of lists
        # img_annos[i] - tensor of shape Mx10 - is a list of gt annos corr to img_id = img_ids[i]
        # bboxes[i] - shape 3Mx4 - is a list of gt bboxes - firstly M human boxes followed by M object boxes followed by M union boxes corr to img_id = img_ids[i]
        # data['proposals'][0][i] - same as boxes[i] but on GPU

        # print("A sample image id", img_metas[0]['filename'][-12:-4])
        # print("Number of samples in this mini-batch =", img.size(0))
        # print()

        for i in range(num_imgs := img.size(0)):
            img_ids.append(img_id := int(img_metas[i]['filename'][-12:-4]))
            img_annos.append(img_anno := anno[anno[:, 0] == img_id])
            union_boxes = get_union_box(img_anno[:, 2:6], img_anno[:, 6:])
            all_boxes = torch.cat(img_anno[:, 2:].split(4, dim=1)) # all human boxes, all obj boxes
            all_boxes = torch.cat((all_boxes, union_boxes), dim=0) # followed by all union boxes
            bboxes.append(all_boxes)
            # here I can append the union gt box to bboxes to extract the features
            # but how to do it for detected huma, object pairs? - maybe a function which calls model by passing in the union box of extracted human-object paur as gt input
            data['proposals'][0].append(bboxes[i].to(img.device))


        with torch.no_grad():
            # print(data)
            # I am also passing in the ground truth proposals to the model 
            gt_blobs, dt_blobs = model(return_loss=False, **data)
            # here for the ground truths, I will get the human box features, obj box features, union box features

            # print(len(dt_blobs)) - same as num_imgs in current batch
            # print(dt_blobs[0].shape) - (N, 1110) where N is the number of detected objects
            # print(gt_blobs[0].shape) - (2M, 1024) - where M is the number of gt hois in the batch img

        for i in range(num_imgs):
            num_anno = img_annos[i].size(0) # M
            obj_inds = hoi_idx_to_obj_idx(img_annos[i][:, 1].int().tolist())

            (h_conf := torch.zeros((num_anno, 80)))[:, 0] = 1
            (o_conf := torch.zeros((num_anno, 80)))[range(num_anno), obj_inds] = 1

            # the last h_conf is appended for union box as dummy to make the sizes compatible in line 92 in torch.cat
            conf = torch.cat((h_conf, o_conf, h_conf)) # shape is 3Mx80, 3M one hot vectors representing human and object label
            gt_blobs[i] = torch.cat((bboxes[i], conf, gt_blobs[i]), dim=1)

            blob[img_ids[i]] = dict(gt=gt_blobs[i], dt=dt_blobs[i])
            # basically each groundtruth blob has bbox, label and feature all concatenated together

        # if cnt == 1:
            # break

        prog_bar.update()

    finish_time = time.time() - start_time

    print()
    print('GT feature extraction for hou, object detection and DT feature extraction for ho complete.')
    print('Took {:.4f} seconds or {:.4f} hours to complete.'.format(finish_time, finish_time/3600))
    print()
    return blob


# extracts the human, object and the union box features for gts
## Time required for the below function to run - 
# for train split -> 926.9446sec or 0.2575hrs
# for test split -> 241.8338sec or 0.0672hrs
def extract_gt_features(annos, blobs, split, checkpoint_dataset='coco'):
    start_time = time.time()

    anno, blob = annos[split], blobs[split]
    img_list = anno[:, 0].unique().int().tolist()

    outputs = []

    print()
    print(f'Building json file for GT features of the *{split}* split.')
    prog_bar = nncore.ProgressBar(num_tasks=len(img_list))
    for img_id in img_list:
    # for img_id in range(1, 2):

        img_anno = anno[anno[:, 0] == img_id][:, 1:]
        # print(img_anno.shape) # Mx9 since its taken from index 1

        img_blob = blob[img_id]
        gt_blob = img_blob['gt']

        # print(gt_blob[:, 4:84])
        # print((gt_blob[:, 4:84]==1).nonzero()[:, 1])

        # print('gt_blob shape before modification', gt_blob.shape) # 3Mx1108


        gt_blob = gt_blob.split(int(gt_blob.size(0) / 3)) # since first onethird are humans and next onethird are objets and last one third are unions
        gt_blob = [b.split((4, 80, 1024), dim=1) for b in gt_blob]
        gt_blob = torch.cat(nncore.concat_list(zip(*gt_blob)), dim=1)

        # print((gt_blob[:, 4:84]==1).nonzero()[:, 1])

        # print('gt_blob shape after modification', gt_blob.shape) # Mx(1108*3)

        gt_img_id = torch.full((gt_blob.size(0), 1), img_id)
        # print(gt_img_id.shape) # M, 1

        gt_blob = torch.cat((gt_img_id, gt_blob), dim=1) # 1 + (1108*3) = 3325 = 1 + 4 + 4 + 4 + 80 + 80 + 80 + 1024 + 1024 + 1024

        # iterating over the hois present in the image
        assert(gt_blob.shape[0] == img_anno.shape[0])
        for i in range(gt_blob.shape[0]):
            hoi_anno = {}
            try:
                hoi_anno['img_id'] = img_id
                hoi_anno['hoi_idx'] = int(img_anno[i, 0].item())
                hoi_anno['human_bbox'] = gt_blob[i, 1:5].tolist()
                hoi_anno['object_bbox'] = gt_blob[i, 5:9].tolist()
                hoi_anno['union_bbox'] = gt_blob[i, 9:13].tolist()
                assert((gt_blob[i, 13:93]==1).nonzero().item() == 0)
                assert((gt_blob[i, 173:253]==1).nonzero().item() == 0)
                hoi_anno['object_idx'] = (gt_blob[i, 93:173]==1).nonzero().item()
                hoi_anno['action_idx'] = hoi_idx_to_act_idx(hoi_anno['hoi_idx'])
                assert(hoi_idx_to_obj_idx(hoi_anno['hoi_idx']) == hoi_anno['object_idx'])
                hoi_anno['human_feats'] = gt_blob[i, 253:1277].tolist()
                hoi_anno['object_feats'] = gt_blob[i, 1277:2301].tolist()
                hoi_anno['union_feats'] = gt_blob[i, 2301:3325].tolist()
                # print(hoi_anno)
                outputs.append(hoi_anno)
            except:
                print(img_anno)
                print(hoi_anno['img_id'])
                print(hoi_anno['hoi_idx'])
                print(hoi_anno['object_idx'])

        # print(gt_blob.shape) # 4,2817
        # print(gt_blob) 
        # print('img_anno', img_anno) # [(hoi_idx, human_bbox, object_bbox)]
        # print('gt_blob_img_ids', gt_blob[:, :1])
        # print('gt_blob_human_bboxes', gt_blob[:, 1:5])
        # print('gt_blob_object_bboxes', gt_blob[:, 5:9])
        # print('gt_blob_human_labels', (gt_blob[:, 9:89]==1).nonzero()[:, 1])
        # print('gt_blob_object_labels', (gt_blob[:, 89:169]==1).nonzero()[:, 1])
        # print('gt_blob_human_feats', gt_blob[:, 169:1193])
        # print('gt_blob_object_feats', gt_blob[:, 1193:2217])
        # print('gt_blob_hoi_labels', (gt_blob[:, 2217:]==1).nonzero()[:, 1]) # multiple hoi labels are coming for a singe human-obj anno pair bcoz of this line since iou >= 0.5 is used to construct gt_labels
        # print('gt_blob_hoi_labels', img_anno[:, 0])

        prog_bar.update()

    with open(f'./data/hico_det/{split}_FasterRCNN_{checkpoint_dataset}_GT_annos_features.json', 'w') as f:
        json.dump(outputs, f)

    finish_time = time.time() - start_time    

    print()
    print(f'Faster rcnn {checkpoint_dataset} detection outputs json file for {split} gts successfully created.')
    print('Took {:.4f} seconds or {:.4f} hours to complete.'.format(finish_time, finish_time/3600))
    print()


# find the union features for dt_blobs
def extract_and_save_dt_info(model, data_loader, annos, blobs, split, cfg, checkpoint_dataset='coco'):
    start_time = time.time()

    anno, blob = annos[split], blobs[split]
    cfg = cfg['test']

    print(f'Extracting union features for obtained HO detection *{split}* split')
    prog_bar = nncore.ProgressBar(num_tasks=len(data_loader))

    # looping over the batches (batch size is 1)
    outputs = []

    for data in data_loader:
        img, img_metas = data['img'][0].data[0], data['img_metas'][0].data[0]
        data['proposals'], bboxes, img_ids, img_annos, h_blobs_dt, o_blobs_dt, gt_blobs = [[]], [], [], [], [], [], []
        # img_annos[i] - tensor of shape Mx10 - is a list of gt annos corr to img_id = img_ids[i]

        for i in range(num_imgs := img.size(0)):
            assert(num_imgs == 1)
            img_ids.append(img_id := int(img_metas[i]['filename'][-12:-4]))
            img_anno = anno[anno[:, 0] == img_id][:, 1:]
            img_annos.append(img_anno)
            img_blob = blob[img_id]

            gt_blob = img_blob['gt']
            dt_blob = img_blob['dt']

            # print(dt_blob.shape) # shape is (N, 1110) where N is the number of detected objs for img

            gt_blob = gt_blob.split(int(gt_blob.size(0) / 3)) # since first half are humans and next half are objets
            gt_blob = [b.split((4, 80, 1024), dim=1) for b in gt_blob]
            gt_blob = torch.cat(nncore.concat_list(zip(*gt_blob)), dim=1)

            h_blob = dt_blob[dt_blob[:, -1] == 0][:, :-1] # humans are those detections for which label=0
            o_blob = dt_blob[dt_blob[:, -1] == 1][:, :-1]

            # print('h_blob.shape', h_blob.shape) # (Nh, 1109)
            # print('o_blob.shape', o_blob.shape) # (No, 1109)

            # inspecting the maximum number of humans which can be used as objects
            if (sep := cfg.max_h_as_o) > 0:
                h_inds = h_blob[:, -1].argsort(descending=True) # shape is (Nh,) - sorting the indices from 0 to Nh-1 in decreasing order of predicted confidence values
                o_blob = torch.cat((o_blob, h_blob[h_inds[:sep]])) # concatenate the top 3 human blobs to o_blob to be considered as objects
            else:
                o_blob = torch.cat((o_blob, h_blob)) # concatenate all the h_blob to o_blob

            h_blob = h_blob[h_blob[:, -1] > cfg.score_thr.hum] # filter out humans less than the threshold based on predicted score
            o_blob = o_blob[o_blob[:, -1] > cfg.score_thr.obj]

            h_inds = h_blob[:, -1].argsort(descending=True) # sort the h_inds in decreasing order of scores
            o_inds = o_blob[:, -1].argsort(descending=True)

            if (max_num := cfg.max_per_img.hum) > 0:
                h_inds = h_inds[:max_num] # take the top 10 humans only

            if (max_num := cfg.max_per_img.obj) > 0:
                o_inds = o_inds[:max_num] # take the top 20 objects only

            num_h = h_inds.size(0)
            num_o = o_inds.size(0)

            # TODO: think something about this line - dont remove the scores 
            # the scores are now removed from the h_blob and o_blob making the num_columns = 1108

            # h_blob = h_blob[h_inds][:, :-1]
            # o_blob = o_blob[o_inds][:, :-1]
            h_blob = h_blob[h_inds] # not removing the scores
            o_blob = o_blob[o_inds]

            # print()
            # print('h_blob.shape', h_blob.shape) # (Nh, 1108)
            # print('o_blob.shape', o_blob.shape) # (No, 1108)

            # if h_blob before = [h1, h2, h3], and num_o = 2, then h_blob later = [h1, h1, h2, h2, h3, h3] where hi is ith row
            h_blob = h_blob.repeat_interleave(num_o, dim=0)
            # if o_blob before = [o1, o2], and num_h = 3, then o_blob later = [o1, o2, o1, o2, o1, o2] where oi is ith row
            o_blob = o_blob.repeat(num_h, 1)

            union_boxes = get_union_box(h_blob[:, :4], o_blob[:, :4])
            # this will contain - [h1o1_ubox, h1o2_ubox, ..., h3o1_ubox, h3o2_ubox] - (6, 4)

            bboxes.append(union_boxes)
            h_blobs_dt.append(h_blob)
            o_blobs_dt.append(o_blob)
            gt_blobs.append(gt_blob)
            data['proposals'][0].append(bboxes[i].to(img.device))

        if bboxes[0].size(0) > 0:
            with torch.no_grad():
                u_feats, _ = model(return_loss=False, **data) # union features
        else:
            u_feats = [torch.empty(0, 1024)] # this is the case, when our faster-rcnn didnt produce any detections for this image and hence bboxes[0].size(0) == 0

        for i in range(num_imgs): # num_imgs=1 for current batch
            img_anno = img_annos[i]
            img_id = img_ids[i]
            gt_blob = gt_blobs[i]
            h_blob = h_blobs_dt[i]
            o_blob = o_blobs_dt[i]
            u_blob = h_blob.clone() # assigning human label to union blobs just for dummy purpose
            u_blob[:, :4] = bboxes[i]
            u_blob[:, 84:1108] = u_feats[i]
            u_blob[:, -1] = h_blob[:, -1] * o_blob[:, -1]

            h_blob = h_blob.split((4, 80, 1024, 1), dim=1) # breaking a tensor into tuple of 3 tensors
            o_blob = o_blob.split((4, 80, 1024, 1), dim=1)
            u_blob = u_blob.split((4, 80, 1024, 1), dim=1)
            dt_blob = [h_blob, o_blob, u_blob]
            dt_blob = torch.cat(nncore.concat_list(zip(*dt_blob)), dim=1)

            # now, dt_blob becomes [h1o1u11, h1o2u12, h2o1u21, h2o2u22, h3o1u31, h3o2u32] where hiojuij is a row in dt_blob obtained as hi_box(4), oj_box(4), uij_box(4), hi_conf(80), oj_conf(80), hi_conf(80-dummy), hi_feats(1024), oj_feats(1024), uij_feats(1024), hi_score(1), oj_score(1), uij_score(1)
            # print('dt_blob shape after modification', dt_blob.shape)

            if dt_blob.numel() > 0:
                # First, we are computing pair_iou - which will be of shape (X,Y) where X is the number of HO (i.e. rows) in dt_blob and Y is the number of HO (i.e. rows) in img_anno
                # pair_iou[i, j] = IoU(dt_blob[i], img_anno[j]) => This IoU is computed between 2 HOs, min(h_iou, o_iou)
                # then, we take max_iou for each row (i.e. i fixed, j varying) to get what img_anno maximally overlaps with the given dt_blob
                m_iou = pair_iou(dt_blob[:, :8], img_anno[:, 1:]).amax(dim=1) # shape is (X,)

                po_inds = (m_iou >= cfg.iou_thr.pos).nonzero()[:, 0] # positive detections as per pair IoU
                ne_inds = (m_iou < cfg.iou_thr.neg).nonzero()[:, 0] 
            else:
                po_inds = ne_inds = []

            gt_blob = torch.cat((gt_blob[:, :172], gt_blob[:, 252:]), dim=1) # = 4 + 4 + 4 + 80 + 80 + 80 (removed in this step) + 1024 + 1024 + 1024
            dt_blob = torch.cat((dt_blob[:, :172], dt_blob[:, 252:]), dim=1) # = 4 + 4 + 4 + 80 + 80 + 80 (removed in this step) + 1024 + 1024 + 1024 + 1 + 1 + 1

            po_blob = dt_blob[po_inds] # positive detections as per pair IoU
            ne_blob = dt_blob[ne_inds]

            gt_label = torch.zeros((gt_blob.size(0), 600))
            po_label = torch.zeros((po_blob.size(0), 600))
            ne_label = torch.zeros((ne_blob.size(0), 600))

            gt_iou = pair_iou(gt_blob[:, :8], img_anno[:, 1:])
            po_iou = pair_iou(po_blob[:, :8], img_anno[:, 1:])

            for i, iou in enumerate(gt_iou):
                gt_label[i][img_anno[:, 0][iou >= 0.5].long()] = 1
                # for each gt HO anno (i.e. row), find all HOI labels for it (depending on which img_anno HO it overlaps with >= 0.5)

            for i, iou in enumerate(po_iou):
                po_label[i][img_anno[:, 0][iou >= 0.5].long()] = 1
                # for each positive HO detection (i.e. row), find all HOI labels for it (depending on which img_anno HO it overlaps with >= 0.5)

            gt_img_id = torch.full((gt_blob.size(0), 1), img_id)
            po_img_id = torch.full((po_blob.size(0), 1), img_id)
            ne_img_id = torch.full((ne_blob.size(0), 1), img_id)

            gt_blob = torch.cat((gt_img_id, gt_blob, gt_label), dim=1) # = 1 + 4 + 4 + 4 + 80 + 80 + 80 + 1024 + 1024 + 1024 + 600

            po_blob = torch.cat((po_img_id, po_blob, po_label), dim=1) # = 1 + 4 + 4 + 4 + 80 + 80 + 1024 + 1024 + 1024 + 1 + 1 + 1 + 600
            ne_blob = torch.cat((ne_img_id, ne_blob, ne_label), dim=1)

            for i in range(po_blob.shape[0]):
                hoi_anno = {}
                try:
                    hoi_anno['img_id'] = img_id
                    # hoi_anno['hoi_idx'] = int(img_anno[i, 0].item())
                    hoi_anno['human_bbox'] = po_blob[i, 1:5].tolist()
                    hoi_anno['object_bbox'] = po_blob[i, 5:9].tolist()
                    hoi_anno['union_bbox'] = po_blob[i, 9:13].tolist()
                    hoi_anno['human_score'] = po_blob[i, -603].item()
                    assert(po_blob[i, -603] == po_blob[i, 13]) # matching the human score 
                    hoi_anno['all_object_scores'] = po_blob[i, 93:173].tolist()
                    # hoi_anno['action_idx'] = hoi_idx_to_act_idx(hoi_anno['hoi_idx'])
                    # assert(hoi_idx_to_obj_idx(hoi_anno['hoi_idx']) == hoi_anno['object_idx'])
                    hoi_anno['human_feats'] = po_blob[i, 173:1197].tolist()
                    hoi_anno['object_feats'] = po_blob[i, 1197:2221].tolist()
                    hoi_anno['union_feats'] = po_blob[i, 2221:3245].tolist()
                    # print(hoi_anno)
                    outputs.append(hoi_anno)
                except:
                    print()
                    # print(img_anno)
                    print(hoi_anno['img_id'])
                    print(po_blob[i, -603])
                    print(po_blob.shape)
                    print(po_blob)
                    # print(hoi_anno['hoi_idx'])
                    # print(hoi_anno['object_idx'])

            # # TODO: dont include gts in this, if to include, intelligently reduce file space, remove multiple HOI labels for each gt
            # if split == 'train':
            #     po_blob = torch.cat((po_blob, gt_blob)) # gts and pos dets
            #     if (fac := cfg.neg_pos_ub) > 0:
            #         ne_blob = ne_blob[:po_blob.size(0) * fac]
            #     nncore.dump(po_blob.numpy(), f, format='h5', dataset='pos')
            #     nncore.dump(ne_blob.numpy(), f, format='h5', dataset='neg')
            # else:
            #     po_blob = torch.cat((po_blob, ne_blob)) # all detections
            #     nncore.dump(gt_blob.numpy(), f, format='h5', dataset='gt')
            #     nncore.dump(po_blob.numpy(), f, format='h5', dataset='pos')

        prog_bar.update()

    with open(f'./data/hico_det/{split}_FasterRCNN_{checkpoint_dataset}_DT_info.json', 'w') as f:
        json.dump(outputs, f)

    finish_time = time.time() - start_time    

    print()
    print(f'Faster rcnn {checkpoint_dataset} detection outputs json file for {split} dts successfully created.')
    print('Took {:.4f} seconds or {:.4f} hours to complete.'.format(finish_time, finish_time/3600))
    print()


@nncore.open(mode='a', format='h5')
def build_dataset(annos, blobs, cfg, split, f):
    anno, blob, cfg = annos[split], blobs[split], cfg[split]
    img_list = anno[:, 0].unique().int().tolist()

    print(f'Building *{split}* split of the dataset')
    prog_bar = nncore.ProgressBar(num_tasks=len(img_list))
    # for img_id in img_list:
    for img_id in range(1, 2):

        img_anno = anno[anno[:, 0] == img_id][:, 1:]
        img_blob = blob[img_id]

        gt_blob = img_blob['gt']
        dt_blob = img_blob['dt']

        # print(dt_blob.shape) # shape is (N, 1110) where N is the number of detected objs for img

        # print(gt_blob[:, 4:84])
        # print((gt_blob[:, 4:84]==1).nonzero()[:, 1])

        # print('gt_blob shape before modification', gt_blob.shape)

        gt_blob = gt_blob.split(int(gt_blob.size(0) / 2)) # since first half are humans and next half are objets
        gt_blob = [b.split((4, 80, 1024), dim=1) for b in gt_blob]
        gt_blob = torch.cat(nncore.concat_list(zip(*gt_blob)), dim=1)

        # print((gt_blob[:, 4:84]==1).nonzero()[:, 1])

        # print('gt_blob shape after modification', gt_blob.shape) # 4, 2216

        h_blob = dt_blob[dt_blob[:, -1] == 0][:, :-1] # humans are those detections for which label=0
        o_blob = dt_blob[dt_blob[:, -1] == 1][:, :-1]

        # print('h_blob.shape', h_blob.shape) # (Nh, 1109)
        # print('o_blob.shape', o_blob.shape) # (No, 1109)

        # inspecting the maximum number of humans which can be used as objects
        if (sep := cfg.max_h_as_o) > 0:
            h_inds = h_blob[:, -1].argsort(descending=True) # shape is (Nh,) - sorting the indices from 0 to Nh-1 in decreasing order of predicted confidence values
            o_blob = torch.cat((o_blob, h_blob[h_inds[:sep]])) # concatenate the top 3 human blobs to o_blob to be considered as objects
        else:
            o_blob = torch.cat((o_blob, h_blob)) # concatenate all the h_blob to o_blob

        h_blob = h_blob[h_blob[:, -1] > cfg.score_thr.hum] # filter out humans less than the threshold based on predicted score
        o_blob = o_blob[o_blob[:, -1] > cfg.score_thr.obj]

        h_inds = h_blob[:, -1].argsort(descending=True) # sort the h_inds in decreasing order of scores
        o_inds = o_blob[:, -1].argsort(descending=True)

        if (max_num := cfg.max_per_img.hum) > 0:
            h_inds = h_inds[:max_num] # take the top 10 humans only

        if (max_num := cfg.max_per_img.obj) > 0:
            o_inds = o_inds[:max_num] # take the top 20 objects only

        num_h = h_inds.size(0)
        num_o = o_inds.size(0)

        # the scores are now removed from the h_blob and o_blob making the num_columns = 1108
        h_blob = h_blob[h_inds][:, :-1]
        o_blob = o_blob[o_inds][:, :-1]
        print()
        print('h_blob.shape', h_blob.shape) # (Nh, 1109)
        print('o_blob.shape', o_blob.shape) # (No, 1109)

        # if h_blob before = [h1, h2, h3], and num_o = 2, then h_blob later = [h1, h1, h2, h2, h3, h3] where hi is ith row
        h_blob = h_blob.repeat_interleave(num_o, dim=0)
        # if o_blob before = [o1, o2], and num_h = 3, then o_blob later = [o1, o2, o1, o2, o1, o2] where oi is ith row
        o_blob = o_blob.repeat(num_h, 1)

        h_blob = h_blob.split((4, 80, 1024), dim=1) # breaking a tensor into tuple of 3 tensors
        o_blob = o_blob.split((4, 80, 1024), dim=1)
        dt_blob = torch.cat(nncore.concat_list(zip(h_blob, o_blob)), dim=1)
        # now, dt_blob becomes [h1o1, h1o2, h2o1, h2o2, h3o1, h3o2] where hioj is a row in dt_blob obtained as hi_box(4), oj_box(4), hi_conf(80), oj_conf(80), hi_feats(1024), oj_feats(1024)
        # print('dt_blob shape after modification', dt_blob.shape)

        if dt_blob.numel() > 0:
            # print(dt_blob.shape) # (X, 2216)
            # print(img_anno.shape) # (Y, 9)
            # print(pair_iou(dt_blob[:, :8], img_anno[:, 1:]).shape) # (X, Y)

            # First, we are computing pair_iou - which will be of shape (X,Y) where X is the number of HO (i.e. rows) in dt_blob and Y is the number of HO (i.e. rows) in img_anno
            # pair_iou[i, j] = IoU(dt_blob[i], img_anno[j]) => This IoU is computed between 2 HOs, min(h_iou, o_iou)
            # then, we take max_iou for each row (i.e. i fixed, j varying) to get what img_anno maximally overlaps with the given dt_blob
            m_iou = pair_iou(dt_blob[:, :8], img_anno[:, 1:]).amax(dim=1) # shape is (X,)
            # print(m_iou.shape)

            po_inds = (m_iou >= cfg.iou_thr.pos).nonzero()[:, 0] # positive detections as per pair IoU
            ne_inds = (m_iou < cfg.iou_thr.neg).nonzero()[:, 0] 
            # print(po_inds)
            # print(ne_inds)
        else:
            po_inds = ne_inds = []

        po_blob = dt_blob[po_inds] # positive detections as per pair IoU
        ne_blob = dt_blob[ne_inds]
        print(po_blob.shape)
        print(ne_blob.shape)

        gt_label = torch.zeros((gt_blob.size(0), 600))
        po_label = torch.zeros((po_blob.size(0), 600))
        ne_label = torch.zeros((ne_blob.size(0), 600))

        gt_iou = pair_iou(gt_blob[:, :8], img_anno[:, 1:])
        po_iou = pair_iou(po_blob[:, :8], img_anno[:, 1:])
        print(po_iou.shape)
        # print('Shape of gt_iou:', gt_iou.shape)
        # print('gt_iou:', gt_iou)

        for i, iou in enumerate(gt_iou):
            gt_label[i][img_anno[:, 0][iou >= 0.5].long()] = 1
            # for each gt HO anno (i.e. row), find all HOI labels for it (depending on which img_anno HO it overlaps with >= 0.5)

        for i, iou in enumerate(po_iou):
            po_label[i][img_anno[:, 0][iou >= 0.5].long()] = 1
            # for each positive HO detection (i.e. row), find all HOI labels for it (depending on which img_anno HO it overlaps with >= 0.5)

        gt_img_id = torch.full((gt_blob.size(0), 1), img_id)
        po_img_id = torch.full((po_blob.size(0), 1), img_id)
        ne_img_id = torch.full((ne_blob.size(0), 1), img_id)
        # print(gt_img_id.shape) # 4, 1

        gt_blob = torch.cat((gt_img_id, gt_blob, gt_label), dim=1) # 1 + 2216 + 600 = 1 + 4 + 4 + 80 + 80 + 1024 + 1024 + 600
        po_blob = torch.cat((po_img_id, po_blob, po_label), dim=1)
        ne_blob = torch.cat((ne_img_id, ne_blob, ne_label), dim=1)

        # print(gt_blob.shape) # 4,2817
        # print(gt_blob) 
        # print('img_anno', img_anno) # [(hoi_idx, human_bbox, object_bbox)]
        # print('gt_blob_img_ids', gt_blob[:, :1])
        # print('gt_blob_human_bboxes', gt_blob[:, 1:5])
        # print('gt_blob_object_bboxes', gt_blob[:, 5:9])
        # print('gt_blob_human_labels', (gt_blob[:, 9:89]==1).nonzero()[:, 1])
        # print('gt_blob_object_labels', (gt_blob[:, 89:169]==1).nonzero()[:, 1])
        # print('gt_blob_human_feats', gt_blob[:, 169:1193])
        # print('gt_blob_object_feats', gt_blob[:, 1193:2217])
        # print('gt_blob_hoi_labels', (gt_blob[:, 2217:]==1).nonzero()[:, 1]) # multiple hoi labels are coming for a singe human-obj anno pair bcoz of this line since iou >= 0.5 is used to construct gt_labels
        # print('gt_blob_hoi_labels', img_anno[:, 0])

        # if split == 'train':
        #     po_blob = torch.cat((po_blob, gt_blob)) # gts and pos dets
        #     if (fac := cfg.neg_pos_ub) > 0:
        #         ne_blob = ne_blob[:po_blob.size(0) * fac]
        #     nncore.dump(gt_blob.numpy(), f, format='h5', dataset='gt') # I added this line
        #     nncore.dump(po_blob.numpy(), f, format='h5', dataset='pos')
        #     nncore.dump(ne_blob.numpy(), f, format='h5', dataset='neg')
        # else:
        #     po_blob = torch.cat((po_blob, ne_blob)) # all detections
        #     nncore.dump(gt_blob.numpy(), f, format='h5', dataset='gt')
        #     nncore.dump(po_blob.numpy(), f, format='h5', dataset='pos')

        prog_bar.update()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--anno',
        help='annotation file',
        default='data/hico_20160224_det/anno_bbox.mat')
    parser.add_argument(
        '--config', help='config file', default='configs/build.py')
    parser.add_argument(
        '--checkpoint',
        help='checkpoint file',
        default='checkpoints/faster_rcnn_r50_fpn_3x_coco-26df6f6b.pth')
    parser.add_argument(
        '--checkpoint_dataset',
        help='The dataset on which checkpoint pretrained or finetuned',
        default='coco'
    )
    parser.add_argument(
        '--out', help='output directory', default='data/hico_det')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = nncore.Config.from_file(args.config) # load the modified faster-RCNN config file

    # load pretrained detector
    args.checkpoint = 'checkpoints/faster_rcnn_r50_fpn_20e_hico_det-77b91312.pth'
    model = build_detector(cfg.model, args.checkpoint)
    print(f"Model checkpoint {args.checkpoint} loaded ... ")
    # print(model)

    ## we can use the following set of line from mmdet tutorial instead of above line
    # import mmcv
    # from mmcv.runner import load_checkpoint

    # from mmdet.apis import inference_detector, show_result_pyplot
    # from mmdet.models import build_detector
    # model = build_detector(config.model)
    # checkpoint = load_checkpoint(model, checkpoint, map_location=device)
    # model.cfg = config
    # model.to(device)
    # model.eval()

    annos, blobs = dict(), dict()
    for split in ['test', 'train']:
        annos[split] = load_anno(args.anno, split)
        # annos['train'] will be a Nx10 tensor -> (img_id, hoi_idx, human_bbox, object_bbox)

        data_loader = build_dataloader(
            cfg.data[split],
            num_samples=cfg.data.samples_per_gpu,
            num_workers=cfg.data.workers_per_gpu)

        blobs[split] = detect_objects(model, data_loader, annos, split)
        extract_gt_features(annos, blobs, split, 'hico')
        extract_and_save_dt_info(model, data_loader, annos, blobs, split, cfg.build, checkpoint_dataset='hico')

        # print(blobs[split][1]['gt'][:, :84])
        # print(blobs[split][1]['dt'][:, :84])
        # print(blobs[split][1]['gt'].shape) # the gt blobs for image with img_id=1; shape = (8, 1108) bbox, conf, feat
        # print(blobs[split][1]['dt'].shape) # shape = (3, 1110) # features are from 84 to 1108 i.e. a[84:1108]; bbox, conf, feat, score, label(0-human/1-object)-obtained based on conf
        # print(blobs[split][1]['dt'][:, 4:84])
        # print(blobs[split][1]['dt'][:, 1108:])

        # file = nncore.join(args.out, f'hico_det_{split}_hou_gt_dtpos_neg_{args.checkpoint_dataset}.hdf5')
        # extract_gt_modified_and_dt_union_features_and_build_hdf5_dataset(model, data_loader, annos, blobs, split, cfg.build, file=file)
        # build_dataset(annos, blobs, cfg.build, split, file=file)
    # BLOBS are object-detection features

if __name__ == '__main__':
    main()

# ## some parts in this function maybe important for extracting frcnn features - visual app features
# def build_graph(annos, blobs, cfg):

#     # setting the edge weights as 1
#     def _link(graph, a, b):
#         graph[a, b] = graph[b, a] = 1

#     print('building consistency graph...')
#     size = dict(act=117, obj=80, tri=600) # len(size.items() == 3)
#     feat = {k: [[] for _ in range(v)] for k, v in size.items()}

#     for split in ['train', 'test']:
#         anno = annos[split]
#         img_list = anno[:, 0].unique().int().tolist()

#         for img_id in img_list:
#             # hois in that image
#             hoi_list = anno[anno[:, 0] == img_id][:, 1].int().tolist()

#             # extracting image features from the object detections
#             img_feat = blobs[split][img_id]['gt'][:, 84:]
#             base_idx = int(len(hoi_list) / 2)

#             for i, hoi_idx in enumerate(hoi_list):
#                 # find the verb and object index
#                 act_idx = hoi_idx_to_act_idx(hoi_idx)
#                 obj_idx = hoi_idx_to_obj_idx(hoi_idx)

#                 # extract the verb and object feature from the img_feature
#                 a_feat = img_feat[i]
#                 o_feat = img_feat[base_idx + i]

#                 # collecting all features for a particular action id
#                 feat['act'][act_idx].append(a_feat)
#                 feat['obj'][obj_idx].append(o_feat)
#                 feat['tri'][hoi_idx].append((a_feat + o_feat) / 2)

#     # 3 keys - act, obj, tri
#     # avergaing out the features per key
#     for key in feat:
#         # class features - taking average per class
#         cls_feat = [torch.stack(f).mean(dim=0) for f in feat[key]]
#         feat[key] = torch.stack(cls_feat) # stacking the class features together

#     # gives the Elmo embedding
#     embedder = build_embedder(cfg.embedder)

#     emb = dict(hum=[embedder.embed(['human'])[0]], act=[], obj=[], tri=[])
#     u_w, t_w, e_w = cfg.uni_weight, cfg.tri_weight, cfg.emb_weight

#     # find out the elmo embedding of all the verbs and store them in emb['act']
#     for idx in range(117):
#         tokens = get_act_name(idx).split('_')
#         out = embedder.embed(tokens)
#         out = out[0] * u_w.act + out[-1] * (1 - u_w.act)
#         emb['act'].append(out)

#     # find out the elmo embedding of all the verbs and store them in emb['act']
#     for idx in range(80):
#         tokens = get_obj_name(idx).split('_')
#         out = embedder.embed(tokens)
#         out = out[0] * u_w.obj + out[-1] * (1 - u_w.obj)
#         emb['obj'].append(out)

#     # find out the elmo embedding of all the interactions and store them in emb['tri']
#     for idx in range(600):
#         a_name, o_name = get_act_and_obj_name(idx)
#         a_t = a_name.split('_')
#         o_t = o_name.split('_')
#         tokens = ['human'] + a_t + o_t # (human, verb, object)
#         out = embedder.embed(tokens)

#         a_emb, o_emb = out[1:].split((len(a_t), len(o_t)))
#         a_emb = a_emb[0] * u_w.act + a_emb[-1] * (1 - u_w.act)
#         o_emb = o_emb[0] * u_w.obj + o_emb[-1] * (1 - u_w.obj)

#         t_emb = out[0] * t_w.hum + a_emb * t_w.act + o_emb * t_w.obj
#         emb['tri'].append(t_emb)

#     emb = {k: torch.stack(v) for k, v in emb.items()}
#     # contains elmo featurs for all actions, objects and interaction triplets

#     # mix the visual embedding coming from pretrained object detector and 
#     # semantic embedding coming from elmo
#     mix_emb = dict()
#     for key in feat:
#         vis_emb = F.normalize(feat[key]) * e_w.vis
#         sem_emb = F.normalize(emb[key]) * e_w.sem
#         mix_emb[key] = torch.cat((vis_emb, sem_emb), dim=1)


#     edges = dict() # graph edges
#     # keys are ['act', 'obj', 'tri']
#     for key, m_emb in mix_emb.items():
#         # m_emb consists of mixed embeddings of all classes for key
#         # compute the pairwise cosine similarities between the classes of a key
#         similarity = cosine_similarity(m_emb, m_emb)
#         inds = similarity.argsort(descending=True)

#         # consider only top 10 (for eg) inter-class edges based on the decreasing order of similarity
#         # edges[key] stores the indices of the classes for whom edges will be made
#         edges[key] = inds[:, :cfg.num_edges[key] + 1].tolist()

#     # not able to understand from where did the below numbers came ??? - 
#     # interaction nodes from 0 to 599
#     # node 600 is a dummy node it seems
#     # act nodes from 601 to 717
#     # obj nodes from 718 to 797
#     base_idx = dict(act=601, obj=718, tri=0)
#     graph = torch.zeros(798, 798)

#     # making the graph edges of weight 1
#     for idx in range(600):
#         _link(graph, idx, 600) # edge from every interaction node (0-599) to node 600
#         _link(graph, idx, hoi_idx_to_act_idx(idx) + base_idx['act']) # edge from interaction idx to action idx
#         _link(graph, idx, hoi_idx_to_obj_idx(idx) + base_idx['obj']) # edge from interaction idx to object idx

#     # constructing edges based on the embedding similarities between
#     # action-action
#     # object-object
#     # interaction-interacion
#     for key, idx in base_idx.items():
#         for i, edge in enumerate(edges[key]):
#             for j in edge:
#                 _link(graph, i + idx, j + idx)

#     # initialising node featurs with elmo embeddings
#     # constructing edges using mix embedding
#     nodes = torch.cat((emb['tri'], emb['hum'], emb['act'], emb['obj']))
#     return dict(nodes=nodes, graph=graph)
