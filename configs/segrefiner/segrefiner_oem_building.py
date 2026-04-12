_base_ = [
    '../_base_/default_runtime.py'
]

object_size = 256
task = 'semantic'

model = dict(
    type='SegRefinerSemantic',
    task=task,
    step=6,
    backbone=None,
    denoise_model=dict(
        type='DenoiseUNet',
        in_channels=4,
        out_channels=1,
        model_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_strides=(16, 32),
        learn_time_embd=True,
        channel_mult=(1, 1, 2, 2, 4, 4),
        dropout=0.1,
        backbone_channels=None),
    diffusion_cfg=dict(
        betas=dict(
            type='sigmoid',
            start=0.8,
            stop=0,
            num_timesteps=6),
        diff_iter=False),
    loss_texture=dict(type='TextureL1Loss', loss_weight=2.0),
    loss_dice=dict(type='DiceLoss', use_sigmoid=True, loss_weight=2.0),
    qsample_cfg=None,
    qsample_checkpoint=None,
    test_cfg=dict(
        model_size=object_size,
        fine_prob_thr=0.9,
        iou_thr=0.3,
        batch_max=32))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Training pipeline: RGB + pre-computed noisy mask at t=5, predict GT directly
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPrecomputedNoisyMask'),
    dict(type='Resize', img_scale=(object_size, object_size), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['object_img', 'object_gt_masks', 'object_noisy_masks',
                               'timestep'])]

# Validation pipeline: pseudo-labels as input, predict refined mask
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCoarseMasks', test_mode=True),
    dict(type='Resize', img_scale=(object_size, object_size), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'coarse_masks'])]

dataset_type = 'OEMBuildingDataset'
data_root = '/home/ubuntu/vy/Denoiser/OEM_v2_Building'
train_dataloader = dict(
    samples_per_gpu=8,
    workers_per_gpu=16,
    shuffle=True)
data = dict(
    train=dict(
        pipeline=train_pipeline,
        type=dataset_type,
        data_root=data_root,
        use_t5_only=True,
        split_file='train.txt'),
    train_dataloader=train_dataloader,
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split_file='val.txt',
        pipeline=test_pipeline,
        test_mode=True),
    val_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=1),
    test=dict())

val_interval = 550
evaluation = dict(interval=val_interval, metric='mIoU', data_root=data_root)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999))
optimizer_config = dict(
    grad_clip=dict(max_norm=0.5, norm_type=2))

max_iters = 55000
runner = dict(type='IterBasedRunner', max_iters=max_iters)

lr_config = dict(
    policy='step',
    gamma=0.5,
    by_epoch=False,
    step=[15000, 20000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=0.001,
    warmup_iters=1000)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False)
    ])
workflow = [('train', 1)]
checkpoint_config = dict(
    by_epoch=False, interval=5000, save_last=False, max_keep_ckpts=1)

