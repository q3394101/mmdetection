# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch.nn as nn
import mmcv
import torch

from ..builder import LOSSES
from .utils import weighted_loss
pdist = nn.PairwiseDistance(p=2)


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def bound_loss(pred, target, linear=False, mode='linear',eps=1e-16):
    """Bbox loss.

        Computing the Bbox loss between a set of predicted bboxes and target bboxes.
        The loss is calculated as negative log of Bbox.

        Args:
            pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
            linear (bool, optional): If True, use linear scale of loss instead of
                log scale. Default: False.
            eps (float): Eps to avoid log(0).

        Return:
            torch.Tensor: Loss tensor.
        """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn('DeprecationWarning: Setting "linear=True" in '
                      'iou_loss is deprecated, please use "mode=`linear`" '
                      'instead.')
    loss = cla_line_point(pred, target).clamp(min=eps)

    if mode == "linear":
        loss = loss
    elif mode == 'square':
        loss = 1 - loss ** 2
    elif mode == 'log':
        loss = -loss.log()
    else:
        raise NotImplementedError
    return loss


def cla_line_point(bboxes1, bboxes2):
    det_point_y = bboxes1[..., 1] + ((bboxes1[..., 3] - bboxes1[..., 1]) / 2)
    det_point_x = bboxes1[..., 2]
    det_point = (det_point_x, det_point_y)

    gt_point_y = bboxes2[..., 1] + ((bboxes2[..., 3] - bboxes2[..., 1]) / 2)
    gt_point_x = bboxes2[..., 0]
    gt_point = (gt_point_x, gt_point_y)

    point_val_result = (((gt_point_x - det_point_x) ** 2) + ((gt_point_y - det_point_y) ** 2)).sqrt()

    # point_val_result = pdist(gt_point, det_point)

    line_val_result = bboxes2[..., 2] - bboxes1[..., 2]

    result = (torch.abs(point_val_result) + torch.abs(line_val_result)) / 32

    return result

@LOSSES.register_module()
class BoundLoss(nn.Module):
    """BBoxLoss.

        Computing the BBox loss between a set of predicted bboxes and target bboxes.

        Args:
            linear (bool): If True, use linear scale of loss else determined
                by mode. Default: False.
            eps (float): Eps to avoid log(0).
            reduction (str): Options are "none", "mean" and "sum".
            loss_weight (float): Weight of loss.
            mode (str): Loss scaling mode, including "linear", "square", and "log".
                Default: 'log'
        """
    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='log'):
        super(BoundLoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * bound_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss