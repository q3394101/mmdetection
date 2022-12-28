from typing import Union
import os.path as osp
import os
import cv2
import warnings

import torch

from mmcv.fileio import FileClient
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.parallel import collate, scatter

from mmcv.runner import EpochBasedRunner, CheckpointHook
from mmcv.parallel import MMDataParallel, scatter, MMDistributedDataParallel


class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

unorm = UnNormalize(mean=[0, 0, 0], std=[1, 1, 1])

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()


@HOOKS.register_module()
class BaseLabelAssignmentVisHook(Hook):
    def __init__(self,
                 sample_idxs: Union[int, list]=0,
                 num_images=None,
                 rate='epoch'):
        self.sample_idxs = [sample_idxs] if isinstance(sample_idxs, int) else sample_idxs
        if num_images is not None and isinstance(sample_idxs, int):
            self.sample_idxs = [_ for _ in range(num_images)]
            warnings.warn(f"parameter 'sample_idxs' must be a list when 'num_images' is not None, "
                          f"setting 'sample_idxs' to {self.sample_idxs}")
        assert rate in ['epoch', 'iter'], "'rate' must be 'epoch' or 'iter'"
        self.rate = rate

        # parameter to check if this Hook has executed before_train_epoch
        # this will make sure before_train_epoch only executes once
        # we set this explicit in before_train_epoch instead of using before_run hook because
        # before_run doesn't contain dataset or dataloader
        self.sampled = False

    def before_run(self, runner):
        self.out_dir = runner.work_dir
        self.file_client = FileClient.infer_client(None,
                                                   self.out_dir)
        self.out_dir = self.file_client.join_path(self.out_dir, "LabelAssignmentVis")
        runner.logger.info(f'Label Assignment Visualization will be saved to {self.out_dir} by '
                           f'{self.file_client.name}.')
        os.makedirs(self.out_dir, exist_ok=True)

    def before_train_epoch(self, runner):
        if self.sampled is False:
            model = runner.model.module
            device = next(model.parameters()).device  # get model device
            dataset = runner.data_loader.dataset
            self.image_list = []
            self.gt_bboxes_list = []
            self.gt_label_list = []
            self.img_metas_list = []
            for i in self.sample_idxs:
                sample_dict = dataset[i]
                # just get the actual data from DataContainer
                sample_dict['img_metas'] = [sample_dict['img_metas'].data]
                sample_dict['img']       = [sample_dict['img'].data]
                sample_dict['gt_bboxes'] = [sample_dict['gt_bboxes'].data]
                sample_dict['gt_labels'] = [sample_dict['gt_labels'].data]
                # scatter sample to specific gpus
                if device.type != 'cpu':
                    sample_dict = scatter(sample_dict, [device])[0]
                image_tensor = sample_dict['img'][0][None] # expand image dim
                gt_bboxes_tensor = sample_dict['gt_bboxes'][0]
                gt_labels_tensor = sample_dict['gt_labels'][0]
                img_metas = sample_dict['img_metas'][0]
                self.image_list.append(image_tensor)
                self.gt_bboxes_list.append(gt_bboxes_tensor)
                self.gt_label_list.append(gt_labels_tensor)
                self.img_metas_list.append(img_metas)
            self.sampled = True

    def after_train_iter(self, runner):
        if self.rate == 'iter':
            runner.logger.info("Performing Label Assignment Visualization...")
            assign_matrices, strides, priors_per_level, featmap_sizes =\
                self._get_assign_results(runner)
            self._plot_results(assign_matrices,
                               strides,
                               priors_per_level,
                               featmap_sizes,
                               runner)

    def after_train_epoch(self, runner):
        if self.rate == 'epoch':
            runner.logger.info("Performing Label Assignment Visualization...")
            assign_matrices, strides, priors_per_level, featmap_sizes =\
                self._get_assign_results(runner)
            self._plot_results(assign_matrices,
                               strides,
                               priors_per_level,
                               featmap_sizes,
                               runner)


    def _get_assign_results(self, runner):
        """ This will execute label assignment from the start for only the images
        in self.image_list. Since every model has its own implementation of label assignment,
        every specific label assignment strat will inherit from this Base and execute its own
        label assignment
        This function must execute these things in order:
        1. Grab model from runner
        2. Put model in eval mode (avoid model updating gradients)
        3. Perform forward pass, get outputs of bbox_head
        4. Perform label assignment to get `assign_result` and `sampling_result`
        5. Return `assign_matrices`, `strides` and `multi_priors_per_level`
        """
        pass

    def _plot_results(self,
                      assign_matrices,
                      strides,
                      multi_priors_per_level,
                      multi_featmap_sizes,
                      runner):
        counter = runner._epoch if self.rate == 'epoch' else runner._iter
        for (image, image_metas, gt_bboxes, gt_label, assign_matrix, stride, priors_per_level, featmap_sizes) in \
                zip(
                    self.image_list,
                    self.img_metas_list,
                    self.gt_bboxes_list,
                    self.gt_label_list,
                    assign_matrices,
                    strides,
                    multi_priors_per_level,
                    multi_featmap_sizes):
            assert len(stride) == len(priors_per_level), "Number of level must equal to number of strides"
            results = []
            image_name = osp.splitext(image_metas['ori_filename'])[0]
            # loop through each scale level to reshape 1D assign matrix
            # to 2D assign matrix of each scale
            num_priors_from_prev_levels = 0
            for i in range(len(priors_per_level)):
                stride_level_i = stride[i]
                featmap_size_level_i = featmap_sizes[i]
                num_priors_level_i = priors_per_level[i]
                matrix_level_i = assign_matrix[num_priors_from_prev_levels:
                                               (num_priors_from_prev_levels + num_priors_level_i)]
                num_priors_from_prev_levels += num_priors_level_i
                matrix_level_i = matrix_level_i.view((featmap_size_level_i[0],
                                                      featmap_size_level_i[1]))
                # this will return a list of 2D position of where the label is non-zero on the matrix
                pos_location_level_i = torch.nonzero(matrix_level_i)
                if pos_location_level_i.numel() > 0:
                    for location in pos_location_level_i:
                        category_id = matrix_level_i[location[0], location[1]]
                        location = (location + 0.5) * stride_level_i[0]
                        results.append(
                            [location.int().cpu().numpy(),
                             category_id.int().cpu().numpy(),
                             stride_level_i[0].numpy()]
                        )

            # unorm_img = unorm(image[0].clone())
            unorm_img = image[0].clone()
            np_image = unorm_img.cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
            # draw positive anchors as circles
            for result in results:
                coord = result[0]
                category_id = result[1]
                scale = result[2]
                np_image = cv2.circle(np_image.copy(),
                                      (coord[1], coord[0]),
                                      int(scale / 2),
                                      colors(category_id),
                                      thickness=-1)
            # draw gt bbox
            for i, gt_bbox in enumerate(gt_bboxes):
                gt_bbox = gt_bbox.int().cpu().numpy()
                gt_id = gt_label[i].int().cpu().numpy()
                np_image = cv2.rectangle(np_image.copy(),
                                         gt_bbox[:2],
                                         gt_bbox[2:],
                                         colors(gt_id))
            # cv2.imwrite(osp.join(self.out_dir, image_name + str(counter) + ".jpg"), np_image)
            cv2.imwrite(osp.join(self.out_dir, image_name  + ".jpg"), np_image)