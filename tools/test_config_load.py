import torch
from mmcv import Config
from mmdet.models import build_detector

def test_load():
    cfg_path = 'configs/segrefiner/segrefiner_oem_building.py'
    cfg = Config.fromfile(cfg_path)
    
    print("Building SegRefiner with Learned Noise Generator...")
    model = build_detector(cfg.model)
    
    if hasattr(model, 'qsample_model') and model.qsample_model is not None:
        print("✓ Noise Generator found in model.")
        print(f"✓ Noise Generator grad status: {all(not p.requires_grad for p in model.qsample_model.parameters())}")
    else:
        print("✗ Noise Generator NOT found in model configuration!")

if __name__ == '__main__':
    test_load()
