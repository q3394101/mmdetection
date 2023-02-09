# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import HOOKS

from .base_label_assignment_vis_hook import BaseLabelAssignmentVisHook


@HOOKS.register_module()
class SimOTAVisualizeHook(BaseLabelAssignmentVisHook):

    def __init__(self, **kwargs):
        super(SimOTAVisualizeHook, self).__init__(**kwargs)

    def _get_assign_results(self, runner):
        model = runner.model.module
        model.eval()
        bbox_head = model.bbox_head

        all_images_assign_matrices = []
        all_images_strides = []
        all_images_num_priors_per_level = []
        all_images_featmap_sizes = []

        with torch.no_grad():
            for (image, gt_bboxes, gt_labels,
                 img_metas) in zip(self.image_list, self.gt_bboxes_list,
                                   self.gt_label_list, self.img_metas_list):
                backbone_feat = model.extract_feat(image)
                outs = bbox_head(backbone_feat)
                if len(outs) == 3:
                    mlvl_cls_score, mlvl_bbox_pred, mlvl_lqe_pred = outs
                elif len(outs) == 2:
                    mlvl_cls_score, mlvl_bbox_pred = outs
                    mlvl_lqe_pred = None
                else:
                    raise NotImplementedError
                num_imgs = 1
                featmap_sizes = [
                    cls_score.shape[2:] for cls_score in mlvl_cls_score
                ]
                mlvl_priors = bbox_head.prior_generator.grid_priors(
                    featmap_sizes,
                    dtype=mlvl_cls_score[0].dtype,
                    device=mlvl_cls_score[0].device,
                    with_stride=True)
                num_priors_per_level = [
                    single_level_priors.shape[0]
                    for single_level_priors in mlvl_priors
                ]
                strides = bbox_head.prior_generator.strides
                all_images_num_priors_per_level.append(num_priors_per_level)
                all_images_strides.append(
                    torch.tensor([stride for stride in strides]))
                all_images_featmap_sizes.append(featmap_sizes)

                flatten_cls_preds = [
                    cls_pred.permute(0, 2, 3,
                                     1).reshape(num_imgs, -1,
                                                bbox_head.cls_out_channels)
                    for cls_pred in mlvl_cls_score
                ]
                flatten_bbox_preds = [
                    bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
                    for bbox_pred in mlvl_bbox_pred
                ]
                if mlvl_lqe_pred is not None:
                    flatten_lqe_pred = [
                        lqe_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                        for lqe_pred in mlvl_lqe_pred
                    ]

                flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
                flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
                if mlvl_lqe_pred is not None:
                    flatten_lqe_pred = torch.cat(flatten_lqe_pred, dim=1)
                flatten_priors = torch.cat(mlvl_priors)
                # since simOTA requires mlvl priors with stride, so we have to
                # explicitly make a :func: _bbox_decode to decode bbox.
                # normal bbox_coder will raise assertion error
                flatten_bboxes = bbox_head._bbox_decode(
                    flatten_priors, flatten_bbox_preds)

                flatten_cls_preds = flatten_cls_preds.squeeze(0)
                flatten_bboxes = flatten_bboxes.squeeze(0)
                flatten_priors = flatten_priors.squeeze(0)
                if mlvl_lqe_pred is not None:
                    flatten_lqe_pred = flatten_lqe_pred.squeeze(0)

                num_priors = flatten_priors.size(0)
                num_gts = gt_labels.size(0)
                gt_bboxes = gt_bboxes.to(flatten_bboxes.dtype)
                # No target
                if num_gts == 0:
                    cls_target = flatten_cls_preds.new_zeros(
                        (0, self.num_classes))
                    bbox_target = flatten_cls_preds.new_zeros((0, 4))
                    l1_target = flatten_cls_preds.new_zeros((0, 4))
                    obj_target = flatten_cls_preds.new_zeros((num_priors, 1))
                    foreground_mask = flatten_cls_preds.new_zeros(
                        num_priors).bool()
                    return (foreground_mask, cls_target, obj_target,
                            bbox_target, l1_target, 0)

                # # YOLOX uses center priors with 0.5 offset to assign targets,
                # # but use center priors without offset to regress bboxes.
                # offset_priors = torch.cat(
                #     [flatten_priors[:, :2] + flatten_priors[:, 2:] * 0.5,
                #     flatten_priors[:, 2:]], dim=-1)

                cls_score = flatten_cls_preds.sigmoid()
                if mlvl_lqe_pred is not None:
                    cls_score = cls_score * flatten_lqe_pred.unsqueeze(
                        1).sigmoid()

                assign_result = bbox_head.assigner.assign(
                    cls_score, flatten_priors, flatten_bboxes, gt_bboxes,
                    gt_labels)

                sampling_result = bbox_head.sampler.sample(
                    assign_result, flatten_priors, gt_bboxes)
                pos_inds = sampling_result.pos_inds
                pos_labels = sampling_result.pos_gt_labels

                assign_matrix = torch.zeros_like(flatten_lqe_pred).long()
                assign_matrix[pos_inds] = pos_labels
                all_images_assign_matrices.append(assign_matrix)

        return (all_images_assign_matrices, all_images_strides,
                all_images_num_priors_per_level, all_images_featmap_sizes)
