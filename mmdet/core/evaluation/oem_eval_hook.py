# Copyright (c) OpenMMLab. All rights reserved.
"""Custom evaluation hook for OEM Building semantic segmentation.

Runs SegRefinerSemantic inference on the val set every N iterations,
compares refined masks with GT labels, and logs mIoU + mBA metrics.
"""
import os
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import Hook, HOOKS


@HOOKS.register_module()
class OEMBuildingEvalHook(Hook):
    """Periodic evaluation hook for OEM Building refinement.

    Runs semantic inference on the val dataloader, computes mIoU and mBA
    by comparing refined masks against GT labels on disk.

    Args:
        dataloader: Val dataloader (test_mode=True, loads pseudo-labels).
        data_root (str): Root directory of the dataset.
        interval (int): Evaluation interval in iterations.
        label_dir (str): Subdirectory containing GT label masks.
        pseudo_dir (str): Subdirectory containing pseudo-label masks.
    """

    def __init__(self, dataloader, data_root, interval=5000,
                 label_dir='labels', pseudo_dir='pseudolabels',
                 save_best=True):
        self.dataloader = dataloader
        self.data_root = data_root
        self.interval = interval
        self.label_dir = osp.join(data_root, label_dir)
        self.pseudo_dir = osp.join(data_root, pseudo_dir)
        self.save_best = save_best
        self.best_miou = 0.0

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        self._do_evaluate(runner)

    def _do_evaluate(self, runner):
        runner.logger.info(
            f'\n--- Val evaluation at iter {runner.iter + 1} ---')
        model = runner.model
        model.eval()

        # Accumulators
        total_refined_i, total_refined_u = 0, 0
        total_pseudo_i, total_pseudo_u = 0, 0
        total_refined_mba, total_pseudo_mba = 0.0, 0.0
        total_num = 0

        prog_bar = mmcv.ProgressBar(len(self.dataloader.dataset))

        for data in self.dataloader:
            with torch.no_grad():
                results = model(return_loss=False, rescale=True, **data)

            for mask_arr, output_fname in results:
                # basename is like 'aachen_1'
                basename = osp.splitext(osp.basename(output_fname))[0]
                gt_fname = basename + '.tif'

                gt_path = osp.join(self.label_dir, gt_fname)
                pseudo_path = osp.join(self.pseudo_dir, gt_fname)

                gt = self._load_mask(gt_path)
                pseudo = self._load_mask(pseudo_path)
                refined = (mask_arr >= 0.5).astype(np.uint8)

                # Ensure all masks match GT shape for metrics
                if refined.shape != gt.shape:
                    refined = cv2.resize(refined, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                if pseudo.shape != gt.shape:
                    pseudo = cv2.resize(pseudo, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

                # IoU building (class 1)
                ri, ru = self._compute_iou(refined, gt)
                pi, pu = self._compute_iou(pseudo, gt)
                total_refined_i += ri
                total_refined_u += ru
                total_pseudo_i += pi
                total_pseudo_u += pu

                # IoU background (class 0)
                ri_bg, ru_bg = self._compute_iou(1 - refined, 1 - gt)
                pi_bg, pu_bg = self._compute_iou(1 - pseudo, 1 - gt)
                
                # Check for existence of bg_iou accumulators, init if needed
                if not hasattr(self, '_accums_init'):
                    self.total_refined_i_bg = 0
                    self.total_refined_u_bg = 0
                    self.total_pseudo_i_bg = 0
                    self.total_pseudo_u_bg = 0
                    self._accums_init = True
                
                self.total_refined_i_bg += ri_bg
                self.total_refined_u_bg += ru_bg
                self.total_pseudo_i_bg += pi_bg
                self.total_pseudo_u_bg += pu_bg

                # mBA
                p_mba, r_mba = self._compute_mba(gt, pseudo, refined)
                total_pseudo_mba += p_mba
                total_refined_mba += r_mba
                total_num += 1

                prog_bar.update()

        # Compute aggregated metrics
        refined_iou_building = total_refined_i / max(total_refined_u, 1)
        refined_iou_bg = self.total_refined_i_bg / max(self.total_refined_u_bg, 1)
        refined_miou = (refined_iou_building + refined_iou_bg) / 2

        pseudo_iou_building = total_pseudo_i / max(total_pseudo_u, 1)
        pseudo_iou_bg = self.total_pseudo_i_bg / max(self.total_pseudo_u_bg, 1)
        pseudo_miou = (pseudo_iou_building + pseudo_iou_bg) / 2

        refined_mba = total_refined_mba / max(total_num, 1)
        pseudo_mba = total_pseudo_mba / max(total_num, 1)

        # Cleanup accumulators for next run
        del self._accums_init

        # Log to runner
        runner.log_buffer.output['val/mIoU'] = refined_miou
        runner.log_buffer.output['val/IoU.building'] = refined_iou_building
        runner.log_buffer.output['val/IoU.background'] = refined_iou_bg
        runner.log_buffer.output['val/mBA'] = refined_mba
        runner.log_buffer.ready = True

        from terminaltables import AsciiTable
        table_data = [
            ['Class', 'Refined IoU', 'Pseudo IoU'],
            ['background', f'{refined_iou_bg*100:.2f}', f'{pseudo_iou_bg*100:.2f}'],
            ['building', f'{refined_iou_building*100:.2f}', f'{pseudo_iou_building*100:.2f}'],
            ['Summary', f'mIoU: {refined_miou*100:.2f}', f'mIoU: {pseudo_miou*100:.2f}']
        ]
        table = AsciiTable(table_data)
        
        runner.logger.info(
            f'\n{table.table}\n'
            f'Val mBA: {refined_mba:.4f} (pseudo={pseudo_mba:.4f}, Δ={refined_mba - pseudo_mba:+.4f})\n'
            f'Images evaluated: {total_num}')

        # Save best checkpoint
        if self.save_best and refined_miou > self.best_miou:
            prev_best = self.best_miou
            self.best_miou = refined_miou
            save_path = osp.join(runner.work_dir, 'best_model.pth')
            runner.save_checkpoint(runner.work_dir, filename_tmpl='best_model.pth', save_optimizer=False)
            runner.logger.info(
                f'★ New best mIoU: {refined_miou*100:.2f}% '
                f'(prev: {prev_best*100:.2f}%, Δ=+{(refined_miou-prev_best)*100:.2f}%) '
                f'→ saved to {save_path}')

        model.train()

    @staticmethod
    def _load_mask(path):
        """Load binary mask, handling both 0/1 and 0/255 formats."""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Could not load mask: {path}')
        return (mask > 0).astype(np.uint8)

    @staticmethod
    def _compute_iou(pred, gt):
        """Return (intersection, union) counts for binary masks."""
        intersection = np.count_nonzero(pred & gt)
        union = np.count_nonzero(pred | gt)
        return intersection, union

    @staticmethod
    def _compute_mba(gt, seg, mask):
        """Compute mean Boundary Accuracy at multiple radii.

        Returns:
            tuple: (seg_mba, mask_mba) — mBA for coarse and refined masks.
        """
        gt = gt.astype(np.uint8)
        seg = seg.astype(np.uint8)
        mask = mask.astype(np.uint8)

        h, w = gt.shape
        min_radius = 1
        max_radius = max(1, (w + h) // 300)
        num_steps = 5

        seg_acc = [0.0] * num_steps
        mask_acc = [0.0] * num_steps

        for i in range(num_steps):
            curr_radius = min_radius + int(
                (max_radius - min_radius) / num_steps * i)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (curr_radius * 2 + 1, curr_radius * 2 + 1))
            boundary_region = cv2.morphologyEx(
                gt, cv2.MORPH_GRADIENT, kernel) > 0

            num_edge_pixels = boundary_region.sum()
            if num_edge_pixels == 0:
                seg_acc[i] = 1.0
                mask_acc[i] = 1.0
                continue

            gt_in_bound = gt[boundary_region]
            seg_in_bound = seg[boundary_region]
            mask_in_bound = mask[boundary_region]

            num_seg_gd_pix = (
                (gt_in_bound) * (seg_in_bound) +
                (1 - gt_in_bound) * (1 - seg_in_bound)
            ).sum()
            num_mask_gd_pix = (
                (gt_in_bound) * (mask_in_bound) +
                (1 - gt_in_bound) * (1 - mask_in_bound)
            ).sum()

            seg_acc[i] = num_seg_gd_pix / num_edge_pixels
            mask_acc[i] = num_mask_gd_pix / num_edge_pixels

        return sum(seg_acc) / num_steps, sum(mask_acc) / num_steps
