# -----------------------------------------------------------------------------------
# Pipeline transforms for training the Learned Q-Sample model.
#
# Generates the ground-truth noisy mask y_t from:
#   - x^{GT}:  object_gt_masks
#   - x^{P}:   object_pseudo_masks  (pre-computed offline)
#   - t:       sampled timestep in {1, ..., T}
#
# Formulation:
#   M^{e}   = (x^{GT} != x^{P})            — binary error mask
#   M^{e}_t = Erosion(M^{e}, T - t)         — eroded error mask at step t
#   y_t     = x^{GT} * (1 - M^{e}_t)        — where error is eroded away, keep GT
#           + x^{P}  * M^{e}_t              — where error remains, keep pseudo
# -----------------------------------------------------------------------------------

import numpy as np
import cv2
import torch
from mmdet.datasets.builder import PIPELINES
from mmdet.core.mask import BitmapMasks


@PIPELINES.register_module()
class LoadPseudoMasks:
    """Load pre-computed pseudo masks from disk."""

    def __init__(self, pseudo_mask_root, filename_key='ann_id', file_ext='.png'):
        self.pseudo_mask_root = pseudo_mask_root
        self.filename_key = filename_key
        self.file_ext = file_ext

    def __call__(self, results):
        ann_id = results['ann_info'].get(self.filename_key, None)
        if ann_id is None:
            raise KeyError(
                f"Key '{self.filename_key}' not found in results['ann_info']. "
                f"Available keys: {list(results['ann_info'].keys())}"
            )

        pseudo_mask_path = f"{self.pseudo_mask_root}/{ann_id}{self.file_ext}"
        mask = cv2.imread(pseudo_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(
                f"Pseudo mask not found at: {pseudo_mask_path}"
            )

        mask = (mask >= 128).astype(np.uint8)
        h, w = results['img_shape'][:2]

        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        results['pseudo_masks'] = BitmapMasks([mask], h, w)
        results['mask_fields'].append('pseudo_masks')
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'pseudo_mask_root={self.pseudo_mask_root}, '
                f'filename_key={self.filename_key})')


@PIPELINES.register_module()
class LoadObjectDataWithPseudo:
    """Crop object region from image, GT mask, and pseudo mask simultaneously."""

    def __init__(self, pad_size=40):
        self.pad_size = pad_size

    def _mask2bbox(self, mask):
        x_any = mask.any(axis=0)
        y_any = mask.any(axis=1)
        x = np.where(x_any)[0]
        y = np.where(y_any)[0]
        x_1, x_2 = x[0], x[-1] + 1
        y_1, y_2 = y[0], y[-1] + 1
        return x_1, y_1, x_2, y_2

    def _get_object_crop_coor(self, x_1, x_2, w, object_size):
        x_start = int(max(object_size / 2, x_2 - object_size / 2))
        x_end = int(min(x_1 + object_size / 2, w - object_size / 2))
        x_c = np.random.randint(x_start, x_end + 1)
        x_1_ob = max(int(x_c - object_size / 2), 0)
        x_2_ob = min(int(x_c + object_size / 2), w)
        return x_1_ob, x_2_ob

    def __call__(self, results):
        h, w = results['img_shape'][:2]
        x_1, y_1, x_2, y_2 = self._mask2bbox(results['gt_masks'].masks[0])
        object_h = min(y_2 - y_1 + self.pad_size, h)
        object_w = min(x_2 - x_1 + self.pad_size, w)
        x_1_ob, x_2_ob = self._get_object_crop_coor(x_1, x_2, w, object_w)
        y_1_ob, y_2_ob = self._get_object_crop_coor(y_1, y_2, h, object_h)

        # Crop image
        results['object_img'] = results['img'][y_1_ob:y_2_ob, x_1_ob:x_2_ob, :]

        # Crop GT masks
        gt_crop = results['gt_masks'].masks[:, y_1_ob:y_2_ob, x_1_ob:x_2_ob]
        results['object_gt_masks'] = BitmapMasks(gt_crop, gt_crop.shape[-2], gt_crop.shape[-1])

        # Crop pseudo masks
        pseudo_crop = results['pseudo_masks'].masks[:, y_1_ob:y_2_ob, x_1_ob:x_2_ob]
        results['object_pseudo_masks'] = BitmapMasks(pseudo_crop, pseudo_crop.shape[-2], pseudo_crop.shape[-1])

        # Clean up and update fields
        del results['ann_info']
        del results['mask_fields']
        results['mask_fields'] = ['object_gt_masks', 'object_pseudo_masks']
        del results['img']
        results['img_shape'] = results['object_img'].shape
        results['ori_shape'] = results['object_img'].shape
        del results['img_fields']
        results['img_fields'] = ['object_img']
        del results['gt_masks']
        del results['pseudo_masks']
        return results


@PIPELINES.register_module()
class GenerateNoisyMask:
    """Generate the ground-truth noisy mask y_t for training the learned q-sample."""

    def __init__(self, num_timesteps=6, kernel_size=3, kernel_shape='cross'):
        self.num_timesteps = num_timesteps
        self.kernel_size = kernel_size

        if kernel_shape == 'cross':
            self.kernel = cv2.getStructuringElement(
                cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elif kernel_shape == 'rect':
            self.kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (kernel_size, kernel_size))
        elif kernel_shape == 'ellipse':
            self.kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        else:
            raise ValueError(f"Unsupported kernel_shape: {kernel_shape}. "
                             f"Choose from 'cross', 'rect', 'ellipse'.")

    def _erode_mask(self, mask, num_iters):
        """Apply morphological erosion to a binary mask."""
        if num_iters <= 0:
            return mask.copy()
        eroded = cv2.erode(mask, self.kernel, iterations=num_iters)
        return eroded

    def __call__(self, results):
        """
        Args:
            results (dict): Must contain:
                - 'object_gt_masks': BitmapMasks with GT mask
                - 'object_pseudo_masks': BitmapMasks with pseudo mask

        Returns:
            dict: Updated results with added keys:
                - 'timestep': (np.ndarray) sampled timestep, shape (1,)
                - 'object_noisy_masks': BitmapMasks with generated noisy mask
                - 'error_mask': BitmapMasks with M^{e} (for debugging/monitoring)
                - 'error_mask_t': BitmapMasks with M^{e}_t (for debugging/monitoring)
        """
        # Extract masks as numpy arrays
        gt_mask = results['object_gt_masks'].masks[0].astype(np.uint8)
        pseudo_mask = results['object_pseudo_masks'].masks[0].astype(np.uint8)
        h, w = gt_mask.shape

        # Sample timestep t from {1, ..., T} if not already provided
        # This allows deterministic validation if 'timestep' is pre-set.
        t_input = results.get('timestep', None)
        if t_input is not None:
            if isinstance(t_input, (np.ndarray, torch.Tensor)):
                t = int(t_input.item())
            else:
                t = int(t_input)
        else:
            t = np.random.randint(1, self.num_timesteps + 1)

        # Compute error mask: M^{e} = (x^{GT} != x^{P})
        error_mask = (gt_mask != pseudo_mask).astype(np.uint8)

        # Compute eroded error mask: M^{e}_t = Erosion(M^{e}, T - t)
        num_erosion_iters = self.num_timesteps - t
        error_mask_t = self._erode_mask(error_mask, num_erosion_iters)

        # Compute noisy mask: y_t = x^{GT} * (1 - M^{e}_t) + x^{P} * M^{e}_t
        noisy_mask = gt_mask * (1 - error_mask_t) + pseudo_mask * error_mask_t
        noisy_mask = noisy_mask.astype(np.uint8)

        # Store results
        results['timestep'] = np.array([t], dtype=np.int64)
        results['object_noisy_masks'] = BitmapMasks([noisy_mask], h, w)
        results['error_mask'] = BitmapMasks([error_mask], h, w)
        results['error_mask_t'] = BitmapMasks([error_mask_t], h, w)

        # Register mask fields for downstream transforms
        results['mask_fields'].append('object_noisy_masks')
        results['mask_fields'].append('error_mask')
        results['mask_fields'].append('error_mask_t')

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_timesteps={self.num_timesteps}, '
                f'kernel_size={self.kernel_size})')
