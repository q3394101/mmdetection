# Copyright (c) OpenMMLab. All rights reserved.
import copy
from turtle import forward
import cv2
import imageio
import torch
from mmengine.dist import get_world_size
from mmengine.logging import print_log

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig
from typing import Any, List, Mapping, Tuple, Union


@MODELS.register_module()
class RTMDet_Dual(SingleStageDetector):
    """Implementation of RTMDet.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of ATSS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of ATSS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
        use_syncbn (bool): Whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        # remove backbone and create dula stream backbones
        del self.backbone
        self.backbone1 = MODELS.build(backbone)
        self.backbone2 = MODELS.build(backbone)
        
        # TODO: Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Find backbone parameters
        copy_ori = False
        ori_backbone_params = []
        ori_backbone_key = []
        for k, v in state_dict.items():
            if k.startswith("backbone")  and "backbone1" not in k and "backbone2" not in k:
                # Pretrained on original model
                ori_backbone_params += [v]
                ori_backbone_key += [k]
                copy_ori = True
                
        if copy_ori:
            for k, v in zip(ori_backbone_key, ori_backbone_params):
                state_dict[k.replace("backbone", "backbone1")] = v
                state_dict[k.replace("backbone", "backbone2")] = copy.deepcopy(v)
                del state_dict[k]
            # Force set the strict to "False"
            strict = False
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def loss(self, batch_inputs: Tensor, batch_inputs2: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs, batch_inputs2)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_inputs2: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs, batch_inputs2)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_inputs2: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """

        
        x = self.extract_feat(batch_inputs, batch_inputs2)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor, batch_inputs2: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """

        x = self.backbone1(batch_inputs)
        y = self.backbone2(batch_inputs2)
        
        ## Concat ##
        z = [.5 * i + .5 * j for i, j in zip(x, y)]
        
        if self.with_neck:
            z = self.neck(z)
        return z
    
    def forward(self,
                inputs: torch.Tensor,
                inputs2: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        # Debug
        # print("*" * 80)
        # cv2.imwrite("debug1.jpg", inputs.permute(0, 2,3,1)[0].cpu().numpy()[:,:,::-1] * 255)
        # cv2.imwrite("debug2.jpg", inputs2.permute(0, 2,3,1)[0].cpu().numpy()[:,:,::-1] * 255)
        # from mmdet.visualization.local_visualizer import DetLocalVisualizer
        # dv = DetLocalVisualizer()
        # image = inputs2.permute(0, 2,3,1)[0].cpu().numpy()[:,:,::-1] * 255
        # image2 = inputs.permute(0, 2,3,1)[0].cpu().numpy()[:,:,::-1] * 255
        # dv.add_datasample('image', image, data_samples[0], draw_gt=True, show=True)
        # dv.add_datasample('image2', image2, data_samples[0], draw_gt=True, show=True)
        
        if mode == 'loss':
            return self.loss(inputs, inputs2, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, inputs2, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, inputs2, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')