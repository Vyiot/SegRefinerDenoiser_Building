"""Evaluation script for OEM Building refinement results.

Computes IoU and mean Boundary Accuracy (mBA) comparing:
  - Pseudo-labels (coarse) vs GT
  - Refined predictions vs GT

Usage:
    python scripts/eval_oem_building.py \
        --data_root data/OEM_v2_Building \
        --split val \
        --refined_dir results/oem_building_refined
"""
import argparse
import os
import os.path as osp

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_boundary_acc(gt, seg, mask):
    """Compute mean Boundary Accuracy at multiple radii.

    Adapted from CascadePSP evaluation.

    Args:
        gt: Ground truth binary mask (H, W), uint8 0/1.
        seg: Coarse/pseudo-label binary mask (H, W), uint8 0/1.
        mask: Refined prediction binary mask (H, W), uint8 0/1.

    Returns:
        seg_acc: mBA of coarse mask.
        mask_acc: mBA of refined mask.
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
        curr_radius = min_radius + int((max_radius - min_radius) / num_steps * i)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (curr_radius * 2 + 1, curr_radius * 2 + 1))
        boundary_region = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0

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


def compute_iou(pred, gt):
    """Compute intersection and union for binary masks."""
    intersection = np.count_nonzero(pred & gt)
    union = np.count_nonzero(pred | gt)
    return intersection, union


def load_binary_mask(filepath):
    """Load a binary mask from .tif or .png file.

    Handles both 0/1 valued masks and 0/255 valued masks.
    """
    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Could not load mask: {filepath}')
    # Threshold to handle both 0/1 and 0/255 formats
    mask = (mask > 0).astype(np.uint8)
    return mask


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate OEM Building refinement results')
    parser.add_argument('--data_root', required=True,
                        help='Root directory of OEM_v2_Building dataset')
    parser.add_argument('--split', default='val', choices=['val', 'test'],
                        help='Which split to evaluate (default: val)')
    parser.add_argument('--refined_dir', required=True,
                        help='Directory containing refined prediction masks')
    parser.add_argument('--refined_suffix', default='.png',
                        help='File extension for refined masks (default: .png)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data_root = args.data_root
    split = args.split
    refined_dir = args.refined_dir
    refined_suffix = args.refined_suffix

    # Read split file
    split_file = osp.join(data_root, f'{split}.txt')
    assert osp.exists(split_file), f'Split file not found: {split_file}'
    with open(split_file, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]

    gt_dir = osp.join(data_root, 'labels')
    pseudo_dir = osp.join(data_root, 'pseudolabels')

    # Accumulators
    total_refined_i = 0
    total_refined_u = 0
    total_pseudo_i = 0
    total_pseudo_u = 0
    total_num_images = 0
    total_pseudo_mba = 0.0
    total_refined_mba = 0.0

    good_cases = []
    bad_cases = []
    skipped = []

    print(f'Evaluating {len(filenames)} images from {split} split...')
    print(f'  GT dir:      {gt_dir}')
    print(f'  Pseudo dir:  {pseudo_dir}')
    print(f'  Refined dir: {refined_dir}')
    print()

    with tqdm(total=len(filenames)) as pbar:
        for fname in filenames:
            gt_path = osp.join(gt_dir, fname)
            pseudo_path = osp.join(pseudo_dir, fname)
            # Try both the original .tif extension and the refined suffix
            basename = osp.splitext(fname)[0]
            refined_path = osp.join(refined_dir, basename + refined_suffix)

            # Fallback: try .tif if .png not found
            if not osp.exists(refined_path):
                refined_path_tif = osp.join(refined_dir, fname)
                if osp.exists(refined_path_tif):
                    refined_path = refined_path_tif
                else:
                    skipped.append(fname)
                    pbar.update()
                    continue

            gt = load_binary_mask(gt_path)
            pseudo = load_binary_mask(pseudo_path)
            refined = load_binary_mask(refined_path)

            # Compute IoU
            pseudo_i, pseudo_u = compute_iou(pseudo, gt)
            refined_i, refined_u = compute_iou(refined, gt)

            total_pseudo_i += pseudo_i
            total_pseudo_u += pseudo_u
            total_refined_i += refined_i
            total_refined_u += refined_u

            # Compute boundary accuracy
            pseudo_mba, refined_mba = compute_boundary_acc(gt, pseudo, refined)
            total_pseudo_mba += pseudo_mba
            total_refined_mba += refined_mba
            total_num_images += 1

            # Track good/bad cases
            pseudo_iou = pseudo_i / max(pseudo_u, 1)
            refined_iou = refined_i / max(refined_u, 1)
            case = dict(
                filename=fname,
                pseudo_iou=pseudo_iou,
                refined_iou=refined_iou,
                pseudo_mba=pseudo_mba,
                refined_mba=refined_mba,
            )
            if refined_iou >= pseudo_iou:
                good_cases.append(case)
            else:
                bad_cases.append(case)

            pbar.update()

    if skipped:
        print(f'\nWARNING: {len(skipped)} files skipped (refined mask not found)')
        if len(skipped) <= 10:
            for s in skipped:
                print(f'  - {s}')

    if total_num_images == 0:
        print('ERROR: No images evaluated!')
        exit(1)

    # Compute aggregated metrics
    pseudo_iou = total_pseudo_i / max(total_pseudo_u, 1)
    refined_iou = total_refined_i / max(total_refined_u, 1)
    pseudo_mba = total_pseudo_mba / total_num_images
    refined_mba = total_refined_mba / total_num_images

    print(f'\n{"=" * 50}')
    print(f'OEM Building Refinement Evaluation ({split} split)')
    print(f'{"=" * 50}')
    print(f'Images evaluated: {total_num_images}')
    print()
    print(f'{"Metric":<15} {"Pseudo-label":>14} {"Refined":>14} {"Delta":>14}')
    print(f'{"-" * 57}')
    print(f'{"IoU":<15} {pseudo_iou:>14.4f} {refined_iou:>14.4f} '
          f'{refined_iou - pseudo_iou:>+14.4f}')
    print(f'{"mBA":<15} {pseudo_mba:>14.4f} {refined_mba:>14.4f} '
          f'{refined_mba - pseudo_mba:>+14.4f}')
    print()
    print(f'Improved cases: {len(good_cases)} / {total_num_images} '
          f'({100.0 * len(good_cases) / total_num_images:.1f}%)')
    print(f'Degraded cases: {len(bad_cases)} / {total_num_images} '
          f'({100.0 * len(bad_cases) / total_num_images:.1f}%)')
