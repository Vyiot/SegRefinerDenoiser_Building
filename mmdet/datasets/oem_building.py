# Copyright (c) OpenMMLab. All rights reserved.
# OEM Building binary segmentation dataset for SegRefiner pipeline.
import os
import os.path as osp

import cv2
import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmdet.core import BitmapMasks
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class OEMBuildingDataset(Dataset):
    """Dataset for OEM Building binary mask refinement.

    Data layout:
        data_root/
        ├── images/          (RGB .tif, flat)
        ├── labels/          (GT binary masks .tif, 0/1, flat)
        ├── pseudolabels/    (CISC-R binary masks .tif, 0/1, flat)
        ├── train.txt
        ├── val.txt
        └── test.txt

    Train mode:
        - Loads RGB images + GT binary masks from labels/
        - Pipeline: LoadImageFromFile → (skip LoadAnnotations) → LoadCoarseMasks
          → LoadObjectData → ...
        - GT mask is loaded by the dataset and injected as BitmapMasks into
          results['gt_masks'] so modify_boundary() in LoadCoarseMasks works.

    Test mode:
        - Loads RGB images + pseudo-labels from pseudolabels/ as coarse masks
        - Pipeline: LoadImageFromFile → LoadCoarseMasks(test_mode=True) → ...

    Note:
        LoadAnnotations._load_masks divides by 255 and thresholds at 0.5,
        which destroys binary 0/1 masks. We bypass it by loading masks
        ourselves in __getitem__ and injecting directly into the results dict.
    """

    CLASSES = ('background', 'building')

    def __init__(self,
                 data_root,
                 img_dir='images',
                 label_dir='labels',
                 pseudo_dir='pseudolabels',
                 split_file=None,
                 pipeline=[],
                 test_mode=False,
                 use_x12=False,
                 use_t5_only=False,
                 use_all_t=False,
                 use_pseudo_direct=False,
                 file_client_args=dict(backend='disk')):
        self.data_root = data_root
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.pseudo_dir = pseudo_dir
        self.test_mode = test_mode
        self.use_x12 = use_x12
        self.use_t5_only = use_t5_only
        self.use_all_t = use_all_t
        self.use_pseudo_direct = use_pseudo_direct
        self.file_client = mmcv.FileClient(**file_client_args)

        # Load split file
        self.data_infos = self.load_data(split_file)

        # Processing pipeline
        self.pipeline = Compose(pipeline)

        if not test_mode:
            self._set_group_flag()

    def load_data(self, split_file):
        """Load file list from split file (.txt).

        Each line in split_file is a filename like 'aachen_1.tif'.
        """
        data_infos = []

        if split_file is not None:
            split_path = osp.join(self.data_root, split_file)
            assert osp.exists(split_path), f'Split file not found: {split_path}'
            with open(split_path, 'r') as f:
                filenames = [line.strip() for line in f if line.strip()]
        else:
            # If no split file, use all files in images dir
            img_full_dir = osp.join(self.data_root, self.img_dir)
            filenames = sorted([
                f for f in os.listdir(img_full_dir)
                if f.endswith('.tif')
            ])

        for fname in filenames:
            img_path = osp.join(self.img_dir, fname)
            label_path = osp.join(self.label_dir, fname)
            pseudo_path = osp.join(self.pseudo_dir, fname)

            if not self.test_mode and self.use_x12:
                # 12x augmentation: 6 noise_gen + 6 mix
                for t in range(6):
                    # T-1 target: mix/(t-1)/ when t>0, else GT (final step learns GT)
                    if t > 0:
                        prev_path = osp.join('noise_mask', 'mix', str(t - 1), fname)
                    else:
                        prev_path = label_path  # T=0 → target is GT

                    # noise_gen version
                    data_infos.append(dict(
                        filename=img_path,
                        label_path=label_path,
                        pseudo_path=pseudo_path,
                        basename=fname,
                        noise_mask_path=osp.join('noise_mask', 'noise_gen', str(t), fname),
                        noise_mask_prev_path=prev_path,
                        timestep=t
                    ))
                    # mix version
                    data_infos.append(dict(
                        filename=img_path,
                        label_path=label_path,
                        pseudo_path=pseudo_path,
                        basename=fname,
                        noise_mask_path=osp.join('noise_mask', 'mix', str(t), fname),
                        noise_mask_prev_path=prev_path,
                        timestep=t
                    ))
            elif not self.test_mode and self.use_t5_only:
                # Only t=5 noise masks (noise_gen + mix), predict GT directly
                for source in ['noise_gen', 'mix']:
                    data_infos.append(dict(
                        filename=img_path,
                        label_path=label_path,
                        pseudo_path=pseudo_path,
                        basename=fname,
                        noise_mask_path=osp.join('noise_mask', source, '5', fname),
                        noise_mask_prev_path=label_path,
                        timestep=5
                    ))
            elif not self.test_mode and self.use_all_t:
                # All timesteps t∈{0..5} × 2 sources, always predict GT (x0)
                for t in range(6):
                    for source in ['noise_gen', 'mix']:
                        data_infos.append(dict(
                            filename=img_path,
                            label_path=label_path,
                            pseudo_path=pseudo_path,
                            basename=fname,
                            noise_mask_path=osp.join('noise_mask', source, str(t), fname),
                            noise_mask_prev_path=label_path,
                            timestep=t
                        ))
            elif not self.test_mode and self.use_pseudo_direct:
                # Use pseudolabel directly as noisy input, fixed t=0 (no diffusion), target=GT
                data_infos.append(dict(
                    filename=img_path,
                    label_path=label_path,
                    pseudo_path=pseudo_path,
                    basename=fname,
                    noise_mask_path=pseudo_path,  # pseudolabel IS the noisy mask
                    noise_mask_prev_path=label_path,
                    timestep=0
                ))
            else:
                data_info = dict(
                    filename=img_path,
                    label_path=label_path,
                    pseudo_path=pseudo_path,
                    basename=fname,
                )
                data_infos.append(data_info)

        from mmdet.utils import get_root_logger
        logger = get_root_logger()
        logger.info(f'Loaded {len(data_infos)} images '
                    f'(split={split_file}, test_mode={self.test_mode})')
        return data_infos

    def _set_group_flag(self):
        """Set flag for GroupSampler (all same group for simplicity)."""
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        return len(self.data_infos)

    def _load_binary_mask(self, mask_path):
        """Load a binary .tif mask (values 0/1) as uint8 array.

        Unlike LoadAnnotations._load_masks which divides by 255 (destroying
        our 0/1 valued masks), this method handles binary masks correctly.
        """
        full_path = osp.join(self.data_root, mask_path)
        mask = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f'Failed to load mask: {full_path}'
        # Ensure binary: values should be 0 or 1
        mask = (mask > 0).astype(np.uint8)
        return mask

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Train mode:
            - Loads GT mask and injects as gt_masks (BitmapMasks)
            - LoadCoarseMasks (test_mode=False) will apply modify_boundary()
              on gt_masks to generate coarse_masks
            - Skips samples with empty masks (no building pixels)

        Test mode:
            - Sets pseudo-label path in coarse_info for LoadCoarseMasks
              (test_mode=True) to load directly
        """
        if not self.test_mode:
            return self._getitem_train(idx)
        else:
            return self._getitem_test(idx)

    def _getitem_train(self, idx):
        """Get training data. Retries with random sample on empty masks."""
        for _ in range(10):  # max retries to avoid infinite loop
            data_info = self.data_infos[idx]

            # Load GT binary mask directly (bypass LoadAnnotations)
            gt_mask = self._load_binary_mask(data_info['label_path'])
            h, w = gt_mask.shape

            # Skip empty masks (no building pixels) — _mask2bbox would crash
            if gt_mask.sum() == 0:
                idx = np.random.randint(0, len(self))
                continue

            img_info = dict(
                filename=data_info['filename'],
                height=h,
                width=w,
            )
            # ann_info needed for LoadObjectData cleanup (del results['ann_info'])
            ann_info = dict()

            results = dict(img_info=img_info, ann_info=ann_info)
            if self.use_x12 or self.use_t5_only or self.use_all_t or self.use_pseudo_direct:
                results['noise_mask_path'] = osp.join(self.data_root, data_info['noise_mask_path'])
                results['noise_mask_prev_path'] = osp.join(self.data_root, data_info['noise_mask_prev_path'])
                results['timestep'] = np.array([data_info['timestep']], dtype=np.int64)
            
            self.pre_pipeline(results)

            # Inject GT mask as BitmapMasks — this is what LoadAnnotations
            # would produce, but correctly handling 0/1 valued masks.
            gt_masks = BitmapMasks([gt_mask], h, w)
            results['gt_masks'] = gt_masks
            results['mask_fields'].append('gt_masks')

            data = self.pipeline(results)
            if data is not None:
                return data
            # Pipeline returned None (e.g., all-background crop), retry
            idx = np.random.randint(0, len(self))

        # Fallback: should rarely reach here
        raise RuntimeError(f'Failed to load valid sample after retries, last idx={idx}')

    def _getitem_test(self, idx):
        """Get test data."""
        data_info = self.data_infos[idx]

        img_info = dict(
            filename=data_info['filename'],
        )
        # coarse_info with path to pseudo-label for LoadCoarseMasks
        coarse_info = dict(masks=data_info['pseudo_path'])

        results = dict(img_info=img_info, coarse_info=coarse_info)
        self.pre_pipeline(results)

        return self.pipeline(results)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.data_root
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Evaluation results.
        """
        from terminaltables import AsciiTable
        from mmcv.utils import print_log

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        # results is a list of [(mask, output_file)] for each sample
        num_classes = len(self.CLASSES)
        total_intersection = np.zeros(num_classes, dtype=np.float64)
        total_union = np.zeros(num_classes, dtype=np.float64)
        total_acc = np.zeros(num_classes, dtype=np.float64)
        total_pixels = np.zeros(num_classes, dtype=np.float64)

        print_log(f'Evaluating {len(results)} samples...', logger=logger)
        
        for i in range(len(self)):
            # Handle results format [(mask_arr, output_fname)]
            pred_mask = results[i][0][0]
            gt_mask = self._load_binary_mask(self.data_infos[i]['label_path'])
            
            for cls_idx in range(num_classes):
                pred_c = (pred_mask == cls_idx)
                gt_c = (gt_mask == cls_idx)
                
                intersection = np.count_nonzero(pred_c & gt_c)
                union = np.count_nonzero(pred_c | gt_c)
                total_intersection[cls_idx] += intersection
                total_union[cls_idx] += union
                
                total_acc[cls_idx] += intersection
                total_pixels[cls_idx] += np.count_nonzero(gt_c)

        ious = total_intersection / np.maximum(total_union, 1e-6)
        accs = total_acc / np.maximum(total_pixels, 1e-6)
        
        # Format per-class results
        header = ['Class', 'IoU', 'Acc']
        table_data = [header]
        for i, cls_name in enumerate(self.CLASSES):
            table_data.append([
                cls_name, f'{ious[i]*100:.2f}', f'{accs[i]*100:.2f}'
            ])
            eval_results[f'IoU.{cls_name}'] = ious[i]
            eval_results[f'Acc.{cls_name}'] = accs[i]

        table = AsciiTable(table_data)
        print_log('\n' + table.table, logger=logger)

        # Summary
        mIoU = np.mean(ious)
        mAcc = np.mean(accs)
        aAcc = np.sum(total_acc) / np.maximum(np.sum(total_pixels), 1e-6)
        
        eval_results['mIoU'] = mIoU
        eval_results['mAcc'] = mAcc
        eval_results['aAcc'] = aAcc
        
        summary_header = ['aAcc', 'mIoU', 'mAcc']
        summary_data = [
            summary_header,
            [f'{aAcc*100:.2f}', f'{mIoU*100:.2f}', f'{mAcc*100:.2f}']
        ]
        summary_table = AsciiTable(summary_data)
        print_log('\nSummary:\n' + summary_table.table, logger=logger)

        return eval_results
