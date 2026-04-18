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

        # Accumulators for single-step
        s_ri, s_ru = 0, 0  # building
        s_ri_bg, s_ru_bg = 0, 0  # background
        s_mba_sum = 0.0
        # Accumulators for iterative
        it_ri, it_ru = 0, 0
        it_ri_bg, it_ru_bg = 0, 0
        it_mba_sum = 0.0
        # Accumulators for pseudo
        p_ri, p_ru = 0, 0
        p_ri_bg, p_ru_bg = 0, 0
        p_mba_sum = 0.0
        total_num = 0

        prog_bar = mmcv.ProgressBar(len(self.dataloader.dataset))

        for data in self.dataloader:
            with torch.no_grad():
                results = model(return_loss=False, rescale=True, **data)

            # results is a list of tuples: (mask_arr, output_fname, mode)
            # Group by mode
            single_results = [(m, f) for m, f, mode in results if mode == 'single']
            iter_results = [(m, f) for m, f, mode in results if mode == 'iterative']
            has_iter = len(iter_results) > 0

            for idx, (s_mask, output_fname) in enumerate(single_results):
                basename = osp.splitext(osp.basename(output_fname))[0]
                gt_fname = basename + '.tif'

                gt_path = osp.join(self.label_dir, gt_fname)
                pseudo_path = osp.join(self.pseudo_dir, gt_fname)

                gt = self._load_mask(gt_path)
                pseudo = self._load_mask(pseudo_path)
                refined_s = (s_mask >= 0.5).astype(np.uint8)

                # Resize if needed
                if refined_s.shape != gt.shape:
                    refined_s = cv2.resize(refined_s, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                if pseudo.shape != gt.shape:
                    pseudo = cv2.resize(pseudo, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

                # IoU building (class 1)
                i, u = self._compute_iou(refined_s, gt)
                s_ri += i; s_ru += u
                i, u = self._compute_iou(pseudo, gt)
                p_ri += i; p_ru += u

                # IoU background (class 0)
                i, u = self._compute_iou(1 - refined_s, 1 - gt)
                s_ri_bg += i; s_ru_bg += u
                i, u = self._compute_iou(1 - pseudo, 1 - gt)
                p_ri_bg += i; p_ru_bg += u

                # Iterative metrics (only if available)
                if has_iter:
                    it_mask, _ = iter_results[idx]
                    refined_it = (it_mask >= 0.5).astype(np.uint8)
                    if refined_it.shape != gt.shape:
                        refined_it = cv2.resize(refined_it, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                    i, u = self._compute_iou(refined_it, gt)
                    it_ri += i; it_ru += u
                    i, u = self._compute_iou(1 - refined_it, 1 - gt)
                    it_ri_bg += i; it_ru_bg += u
                    _, it_ba = self._compute_mba(gt, pseudo, refined_it)
                    it_mba_sum += it_ba

                # mBA
                p_ba, s_ba = self._compute_mba(gt, pseudo, refined_s)
                s_mba_sum += s_ba
                p_mba_sum += p_ba
                total_num += 1

                prog_bar.update()

        # Compute aggregated metrics
        s_iou_b = s_ri / max(s_ru, 1)
        s_iou_bg = s_ri_bg / max(s_ru_bg, 1)
        s_miou = (s_iou_b + s_iou_bg) / 2

        p_iou_b = p_ri / max(p_ru, 1)
        p_iou_bg = p_ri_bg / max(p_ru_bg, 1)
        p_miou = (p_iou_b + p_iou_bg) / 2

        s_mba = s_mba_sum / max(total_num, 1)
        p_mba = p_mba_sum / max(total_num, 1)

        # Log to runner
        runner.log_buffer.output['val/mIoU_single'] = s_miou
        runner.log_buffer.output['val/IoU_single.building'] = s_iou_b
        runner.log_buffer.output['val/IoU_single.background'] = s_iou_bg
        runner.log_buffer.output['val/mBA_single'] = s_mba

        from terminaltables import AsciiTable

        if has_iter:
            it_iou_b = it_ri / max(it_ru, 1)
            it_iou_bg = it_ri_bg / max(it_ru_bg, 1)
            it_miou = (it_iou_b + it_iou_bg) / 2
            it_mba = it_mba_sum / max(total_num, 1)

            runner.log_buffer.output['val/mIoU_iter'] = it_miou
            runner.log_buffer.output['val/IoU_iter.building'] = it_iou_b
            runner.log_buffer.output['val/IoU_iter.background'] = it_iou_bg
            runner.log_buffer.output['val/mBA_iter'] = it_mba

            table_data = [
                ['Class', 'Single IoU', 'Iter IoU', 'Pseudo IoU'],
                ['background', f'{s_iou_bg*100:.2f}', f'{it_iou_bg*100:.2f}', f'{p_iou_bg*100:.2f}'],
                ['building', f'{s_iou_b*100:.2f}', f'{it_iou_b*100:.2f}', f'{p_iou_b*100:.2f}'],
                ['Summary',
                 f'mIoU: {s_miou*100:.2f}',
                 f'mIoU: {it_miou*100:.2f}',
                 f'mIoU: {p_miou*100:.2f}']
            ]
            mba_str = f'mBA: single={s_mba:.4f}, iter={it_mba:.4f}, pseudo={p_mba:.4f}'
        else:
            table_data = [
                ['Class', 'Refined IoU', 'Pseudo IoU'],
                ['background', f'{s_iou_bg*100:.2f}', f'{p_iou_bg*100:.2f}'],
                ['building', f'{s_iou_b*100:.2f}', f'{p_iou_b*100:.2f}'],
                ['Summary', f'mIoU: {s_miou*100:.2f}', f'mIoU: {p_miou*100:.2f}']
            ]
            mba_str = f'mBA: refined={s_mba:.4f}, pseudo={p_mba:.4f}'

        runner.log_buffer.ready = True
        table = AsciiTable(table_data)
        
        runner.logger.info(
            f'\n{table.table}\n'
            f'{mba_str}\n'
            f'Images evaluated: {total_num}')

        # Save best checkpoint (based on single-step mIoU)
        if self.save_best and s_miou > self.best_miou:
            prev_best = self.best_miou
            self.best_miou = s_miou
            save_path = osp.join(runner.work_dir, 'best_model.pth')
            runner.save_checkpoint(runner.work_dir, filename_tmpl='best_model.pth', save_optimizer=False)
            runner.logger.info(
                f'★ New best mIoU: {s_miou*100:.2f}% '
                f'(prev: {prev_best*100:.2f}%, Δ=+{(s_miou-prev_best)*100:.2f}%) '
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
