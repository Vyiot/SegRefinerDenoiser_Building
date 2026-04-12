import os
import torch
import torch.nn.functional as F
import numpy as np
from .segrefiner_base import SegRefiner
from ..builder import DETECTORS, build_head, build_loss
from mmcv.ops import nms

def mask2bbox(mask):
    x_any = mask.any(axis=0)
    y_any = mask.any(axis=1)
    x = np.where(x_any)[0]
    y = np.where(y_any)[0]
    x_1, x_2 = x[0], x[-1] + 1
    y_1, y_2 = y[0], y[-1] + 1
    return x_1, y_1, x_2, y_2

@DETECTORS.register_module()
class SegRefinerSemantic(SegRefiner):
    
    def get_output_filename(self, img_metas):
        ori_filename = img_metas[0]['ori_filename']
        # Handle OEM Building format: images/filename.tif → filename.png
        if ori_filename.endswith('.tif'):
            basename = os.path.basename(ori_filename)
            return basename.replace('.tif', '.png')
        if 'dis' in ori_filename:
            ori_filename = ori_filename.split('/')
            testset, finename = ori_filename[1], ori_filename[-1]
            output_flie = os.path.join(testset, finename.replace('.jpg', '.png'))
        else:
            output_flie = ori_filename.replace('im.jpg', 'refine.png')
        return output_flie
    
    def simple_test_semantic(self, img_metas, img, coarse_masks, **kwargs):

        output_file = self.get_output_filename(img_metas)

        if coarse_masks[0].masks.sum() <= 128:
            return [(np.zeros_like(coarse_masks[0].masks[0]), output_file)]
        
        current_device = img.device
        ori_shape = img_metas[0]['ori_shape'][:2]

        # Direct 1-step xT → x0 (t=5 is the noise level)
        indices = [5]

        # global_step only (no local refinement)
        global_img, global_mask = self._get_global_input(img, coarse_masks, ori_shape, current_device)

        # Backbone feature extraction (only when backbone exists)
        global_features = None
        if self.backbone is not None:
            if hasattr(self.backbone, 'in_channels') and self.backbone.in_channels == 4:
                global_features = self.backbone(torch.cat((global_img, global_mask), dim=1))
            else:
                global_features = self.backbone(global_img)

        model_size_mask, _ = self.p_sample_loop(
            [(global_mask, global_img, None)],
            indices, current_device, use_last_step=True,
            features=global_features)
        
        ori_size_mask = F.interpolate(model_size_mask, size=ori_shape)
        ori_size_mask = (ori_size_mask >= 0.5).float()

        mask = ori_size_mask.squeeze(0).squeeze(0)
        return [(mask.cpu().numpy(), output_file)]
    
    def _get_global_input(self, img, coarse_masks, ori_shape, current_device):
        model_size = self.test_cfg.get('model_size', 256)
        coarse_mask = coarse_masks[0].masks[0]
        global_img = F.interpolate(img, size=(model_size, model_size))
        global_mask = torch.tensor(coarse_mask, dtype=torch.float32, device=current_device)
        global_mask = F.interpolate(global_mask.unsqueeze(0).unsqueeze(0), size=(model_size, model_size))
        global_mask = (global_mask >= 0.5).float()
        return global_img, global_mask    
        
    def get_local_input(self, img, ori_size_mask, fine_probs, ori_shape):
        img_h, img_w = ori_shape
        ori_size_fine_probs = F.interpolate(fine_probs, ori_shape)
        fine_prob_thr = self.test_cfg.get('fine_prob_thr', 0.9)
        fine_prob_thr = fine_probs.max().item() * fine_prob_thr
        model_size = self.test_cfg.get('model_size', 0)
        low_cofidence_points = fine_probs < fine_prob_thr
        scores = fine_probs[low_cofidence_points]
        y_c, x_c = torch.where(low_cofidence_points.squeeze(0).squeeze(0))
        scale_factor_y, scale_factor_x = img_h / model_size, img_w / model_size
        y_c, x_c = (y_c * scale_factor_y).int(), (x_c * scale_factor_x).int()        
        scores = 1 - scores
        patch_coors = self._get_patch_coors(x_c, y_c, 0, 0, img_w, img_h, model_size, scores)
        return self.crop_patch(img, ori_size_mask, ori_size_fine_probs, patch_coors)
    
    def _get_patch_coors(self, x_c, y_c, X_1, Y_1, X_2, Y_2, patch_size, scores):
        y_1, y_2 = y_c - patch_size/2, y_c + patch_size/2 
        x_1, x_2 = x_c - patch_size/2, x_c + patch_size/2
        invalid_y = y_1 < Y_1
        y_1[invalid_y] = Y_1
        y_2[invalid_y] = Y_1 + patch_size
        invalid_y = y_2 > Y_2
        y_1[invalid_y] = Y_2 - patch_size
        y_2[invalid_y] = Y_2
        invalid_x = x_1 < X_1
        x_1[invalid_x] = X_1
        x_2[invalid_x] = X_1 + patch_size
        invalid_x = x_2 > X_2
        x_1[invalid_x] = X_2 - patch_size
        x_2[invalid_x] = X_2
        proposals = torch.stack((x_1, y_1, x_2, y_2), dim=-1)
        patch_coors, _ = nms(proposals, scores, iou_threshold=self.test_cfg.get('iou_thr', 0.2))
        return patch_coors.int()
    
    def crop_patch(self, img, mask, fine_probs, patch_coors):
        patch_imgs, patch_masks, patch_fine_probs, new_patch_coors = [], [], [], []
        for coor in patch_coors:
            patch_mask = mask[:, :, coor[1]:coor[3], coor[0]:coor[2]]
            if (patch_mask.any()) and (not patch_mask.all()):
                patch_imgs.append(img[:, :, coor[1]:coor[3], coor[0]:coor[2]])
                patch_fine_probs.append(fine_probs[:, :, coor[1]:coor[3], coor[0]:coor[2]])
                patch_masks.append(patch_mask)
                new_patch_coors.append(coor)
        if len(patch_imgs) == 0:
            return None, None, None, None
        
        patch_imgs = torch.cat(patch_imgs, dim=0)
        patch_masks = torch.cat(patch_masks, dim=0)
        patch_fine_probs = torch.cat(patch_fine_probs, dim=0)

        patch_masks = (patch_masks >= 0.5).float()
        return patch_imgs, patch_masks, patch_fine_probs, new_patch_coors
    
    def paste_local_patch(self, local_masks, mask, patch_coors):
        mask = mask.squeeze(0).squeeze(0)
        refined_mask = torch.zeros_like(mask)
        weight = torch.zeros_like(mask)
        local_masks = local_masks.squeeze(1)
        for local_mask, coor in zip(local_masks, patch_coors):
            refined_mask[coor[1]:coor[3], coor[0]:coor[2]] += local_mask
            weight[coor[1]:coor[3], coor[0]:coor[2]] += 1
        refined_area = (weight > 0).float()
        weight[weight == 0] = 1
        refined_mask = refined_mask / weight
        refined_mask = (refined_mask >= 0.5).float()
        return refined_area * refined_mask + (1 - refined_area) * mask

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
    
    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        raise NotImplementedError



