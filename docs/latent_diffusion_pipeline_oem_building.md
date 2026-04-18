# Pipeline: Latent Diffusion Model cho Pseudo-label Denoising (OEM Building)

> **Mục tiêu**: Dùng Latent Diffusion Model (LDM) để **làm sạch pseudo-label** cho building segmentation. Model nhìn vào ảnh RGB (condition) và pseudo-label bị lỗi → sinh ra mask sạch hơn.

---

## 1. Tổng quan: Cái gì vào, cái gì ra?

```
Input:  1) RGB satellite image (ảnh vệ tinh)
        2) Pseudo-label (mask building bị lỗi, từ CISC-R)
        
Output: Refined mask (mask building đã được làm sạch, gần GT hơn)
```

**Ý tưởng cốt lõi**: Thay vì làm diffusion trực tiếp trên ảnh mask (pixel space), ta **nén mask nhỏ lại** (latent space) rồi mới làm diffusion → nhanh hơn, chất lượng tốt hơn.

---

## 2. Pipeline gồm 2 Stage — Chi tiết cụ thể

### Stage 1: Train Mask Autoencoder (nén mask)

#### Autoencoder là gì?

Autoencoder = 2 phần: **Encoder** (nén) + **Decoder** (giải nén):

```
Encoder: mask 256×256 (lớn) → nén thành "latent" 64×64×3 (nhỏ 16 lần)
Decoder: latent 64×64×3 (nhỏ) → phục hồi lại mask 256×256 (lớn)
```

#### Train autoencoder như thế nào?

```
┌──────────────── MỘT BƯỚC TRAINING CỦA AUTOENCODER ────────────────┐
│                                                                     │
│  Bước 1: Lấy 1 GT mask từ folder labels/                           │
│          Ví dụ: labels/aachen_1.tif  →  mask kích thước 256×256     │
│          (giá trị 0 = background, 1 = building)                     │
│                                                                     │
│  Bước 2: Đưa mask qua Encoder                                      │
│          mask (1×256×256) → Encoder → latent z (3×64×64)            │
│          (nén nhỏ lại 4 lần mỗi chiều)                              │
│                                                                     │
│  Bước 3: Đưa latent z qua Decoder                                  │
│          latent z (3×64×64) → Decoder → mask' (1×256×256)           │
│          (phục hồi lại kích thước gốc)                              │
│                                                                     │
│  Bước 4: So sánh mask' với mask gốc, tính Loss                     │
│          Loss = |mask' - mask|  (L1 loss, càng nhỏ càng tốt)       │
│          Mục tiêu: mask' ≈ mask (reconstruct chính xác)             │
│                                                                     │
│  Bước 5: Backprop, cập nhật weights Encoder + Decoder               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Tóm lại Stage 1**: Model học cách nén/giải nén mask. Sau khi train xong:
- Encoder biết cách nén mask → latent nhỏ
- Decoder biết cách giải nén latent → mask lại

> [!NOTE]
> **Data dùng cho Stage 1**: Chỉ dùng **GT masks** (labels/), KHÔNG dùng pseudo-labels. Vì autoencoder cần học nén mask "sạch". Không cần ảnh RGB ở stage này.

#### Val (đánh giá) autoencoder như thế nào?

```
┌──────────────── ĐÁNH GIÁ AUTOENCODER ──────────────────────────────┐
│                                                                     │
│  Với mỗi GT mask trong val.txt (218 ảnh):                          │
│                                                                     │
│    1. mask_val = đọc GT mask từ labels/                             │
│    2. latent   = Encoder(mask_val)                                  │
│    3. mask_rec = Decoder(latent)                                    │
│    4. mask_bin = (mask_rec > 0.5) ? 1 : 0    # threshold về binary │
│    5. IoU = tính IoU giữa mask_bin và mask_val                      │
│                                                                     │
│  Kết quả kỳ vọng: IoU ≥ 0.98                                       │
│  (Autoencoder phải reconstruct gần như hoàn hảo)                    │
│                                                                     │
│  Nếu IoU < 0.95 → autoencoder chưa đủ tốt → train thêm            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Stage 2: Train Latent Diffusion (học khử nhiễu trong latent space)

#### Yêu cầu trước: Cần có sẵn
- ✅ Mask Autoencoder đã train xong (Stage 1) → **freeze** (đóng băng, không update nữa)
- ✅ RGB Autoencoder pretrained (download sẵn từ LDM Model Zoo) → **freeze**

#### Train diffusion model như thế nào? (từng bước cụ thể)

```
┌──────────────── MỘT BƯỚC TRAINING CỦA DIFFUSION ──────────────────┐
│                                                                     │
│  Bước 1: Lấy cặp (RGB image, GT mask) từ training set              │
│          RGB:  images/aachen_1.tif    → (3×256×256)                 │
│          Mask: labels/aachen_1.tif    → (1×256×256)                 │
│                                                                     │
│  Bước 2: Encode cả hai vào latent space (dùng AE frozen)           │
│          z_mask = MaskEncoder(GT_mask)     → (3×64×64) latent mask  │
│          c_rgb  = RGBEncoder(RGB_image)    → (3×64×64) latent RGB   │
│                                                                     │
│  Bước 3: Thêm noise Gaussian ngẫu nhiên vào z_mask                 │
│          - Random chọn timestep t (ví dụ t=300 trong 1→1000)       │
│          - Random noise ε ~ N(0, I)   (noise Gaussian chuẩn)       │
│          - z_t = √(ᾱ_t) × z_mask + √(1 - ᾱ_t) × ε                │
│            (ᾱ_t càng nhỏ → noise càng nhiều)                       │
│            t=1:   z_t ≈ z_mask            (gần như không noise)     │
│            t=500: z_t = 50% z_mask + 50% noise (lẫn lộn)          │
│            t=1000:z_t ≈ ε                 (gần như toàn noise)     │
│                                                                     │
│  Bước 4: Ghép z_t với c_rgb làm input cho UNet                     │
│          input = concat(z_t, c_rgb)        → (6×64×64)              │
│          (UNet nhìn thấy: mask bị noise + ảnh RGB gốc)              │
│                                                                     │
│  Bước 5: UNet dự đoán noise                                        │
│          ε_predicted = UNet(input, t)      → (3×64×64)              │
│          (UNet cố đoán: đâu là noise đã thêm vào ở Bước 3)        │
│                                                                     │
│  Bước 6: Tính Loss                                                  │
│          Loss = MSE(ε_predicted, ε)                                 │
│          (So sánh noise dự đoán với noise thật)                     │
│          Càng đoán đúng noise → càng "khử" được noise → mask sạch  │
│                                                                     │
│  Bước 7: Backprop, chỉ cập nhật weights của UNet                   │
│          (2 autoencoder vẫn frozen, không thay đổi)                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Tại sao cần RGB condition?**
- Không có RGB → UNet chỉ thấy mask bị noise → khó biết building ở đâu
- Có RGB → UNet thấy ảnh vệ tinh gốc → biết chính xác building nằm chỗ nào → khử noise chính xác hơn

**Tại sao train với GT mask mà không phải pseudo-label?**
- Training cần cặp (clean, noisy): ta tự thêm noise vào GT mask → tạo ra cặp train
- Model học cách: "nhìn ảnh RGB + mask bị noise → tìm ra noise → loại bỏ noise"
- Sau khi học xong → áp dụng cho pseudo-label (cũng là dạng mask bị "noise")

#### Val (đánh giá) diffusion model như thế nào?

```
┌──────────────── ĐÁNH GIÁ DIFFUSION MODEL ─────────────────────────┐
│                                                                     │
│  CÓ 2 CÁCH ĐÁNH GIÁ:                                              │
│                                                                     │
│  ═══ Cách 1: Val loss (tự động, chạy mỗi N iter) ═══              │
│                                                                     │
│  Giống training nhưng KHÔNG backprop:                               │
│    1. Lấy cặp (RGB, GT mask) từ val set (val.txt, 218 ảnh)        │
│    2. Encode → z_mask, c_rgb                                        │
│    3. Random t, random ε, tạo z_t                                   │
│    4. UNet dự đoán ε_pred                                           │
│    5. val_loss = MSE(ε_pred, ε)                                     │
│                                                                     │
│  Theo dõi: val_loss giảm dần → model đang học tốt                  │
│            val_loss tăng lại → overfitting → stop training          │
│                                                                     │
│  ═══ Cách 2: Denoise pseudo-label → SO SÁNH mIoU (quan trọng) ═══ │
│                                                                     │
│  Đây là cách đánh giá CHÍNH: model có thật sự cải thiện            │
│  pseudo-label hay không?                                            │
│                                                                     │
│  Chạy trên TOÀN BỘ val set (218 ảnh), tính mIoU:                  │
│                                                                     │
│    ┌─ mIoU 1: Pseudo vs GT (baseline) ──────────────────────┐      │
│    │  Với mỗi ảnh: IoU(pseudo_label, GT_mask)                │      │
│    │  pseudo_mIoU = trung bình IoU trên 218 ảnh              │      │
│    └─────────────────────────────────────────────────────────┘      │
│                                                                     │
│    ┌─ mIoU 2: Refined vs GT (model output) ─────────────────┐      │
│    │  Với mỗi ảnh:                                           │      │
│    │    1. z_pseudo = MaskEncoder(pseudo_label)               │      │
│    │    2. c_rgb = RGBEncoder(RGB_image)                      │      │
│    │    3. z_noisy = add_noise(z_pseudo, t=500)               │      │
│    │    4. DDIM 50 bước: z_500 → ... → z_0                   │      │
│    │       (mỗi bước UNet nhìn RGB → đoán noise → bỏ bớt)   │      │
│    │    5. refined_mask = MaskDecoder(z_0)                    │      │
│    │    6. final_mask = (refined_mask > 0.5) ? 1 : 0         │      │
│    │  refined_mIoU = trung bình IoU trên 218 ảnh             │      │
│    └─────────────────────────────────────────────────────────┘      │
│                                                                     │
│  So sánh:                                                           │
│    ┌─────────────────────────────────────────────────────────┐      │
│    │  pseudo_mIoU   = 0.72   (baseline)                      │      │
│    │  refined_mIoU  = 0.82   (model output)                  │      │
│    │  Δ mIoU        = +0.10  ✅ Model cải thiện!             │      │
│    │                                                         │      │
│    │  PASS nếu: refined_mIoU > pseudo_mIoU                   │      │
│    │  TARGET:   Δ mIoU ≥ +0.05 (tăng ít nhất 5 điểm)       │      │
│    └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tóm tắt toàn bộ Flow

```
╔══════════════════════════════════════════════════════════════════════╗
║                        TOÀN BỘ PIPELINE                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─── STAGE 1: Train Autoencoder ───────────────────────────────┐   ║
║  │  Input:  GT masks (labels/)                                   │   ║
║  │  Học gì: Nén mask ↔ giải nén mask (không liên quan RGB)      │   ║
║  │  Output: Trained Encoder + Decoder                            │   ║
║  │  Val:    Reconstruct mask val, đo IoU ≥ 0.98                  │   ║
║  └───────────────────────────────────────────────────────────────┘   ║
║                              ↓ freeze                                ║
║  ┌─── STAGE 2: Train Diffusion UNet ────────────────────────────┐   ║
║  │  Input:  (RGB images + GT masks) từ train set                 │   ║
║  │  Học gì: Nhìn RGB + mask bị noise → đoán noise → loại noise  │   ║
║  │  Output: Trained UNet                                         │   ║
║  │  Val:    1) val_loss giảm? 2) Denoise pseudo → IoU tăng?     │   ║
║  └───────────────────────────────────────────────────────────────┘   ║
║                              ↓                                       ║
║  ┌─── INFERENCE: Denoise Pseudo-labels ─────────────────────────┐   ║
║  │  Input:  RGB image + Pseudo-label (lỗi)                       │   ║
║  │  Process:                                                     │   ║
║  │    1. Encode pseudo-label → latent                            │   ║
║  │    2. Thêm noise 1 phần (t=500)                              │   ║
║  │    3. DDIM: khử noise dần dần (50 bước),                     │   ║
║  │       mỗi bước UNet nhìn RGB condition để biết building ở đâu│   ║
║  │    4. Decode latent → refined mask                            │   ║
║  │  Output: Refined mask (sạch hơn pseudo-label)                 │   ║
║  └───────────────────────────────────────────────────────────────┘   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 4. Data: Cái gì dùng ở đâu?

| Data | Folder | Stage 1 (AE) | Stage 2 Train | Stage 2 Val (loss) | Inference |
|------|--------|:---:|:---:|:---:|:---:|
| **GT masks** | `labels/` | ✅ Input+Target | ✅ Target (để thêm noise) | ✅ Target | ❌ (không có lúc test thật) |
| **RGB images** | `images/` | ❌ | ✅ Condition | ✅ Condition | ✅ Condition |
| **Pseudo-labels** | `pseudolabels/` | ❌ | ❌ | ✅ (cách 2: test thực tế) | ✅ Input cần denoise |

---

## 5. Kiến trúc chi tiết & Config

### 5.1 Mask Autoencoder (KL-VAE f=4)

```yaml
model:
  target: ldm.models.autoencoder.AutoencoderKL
  params:
    embed_dim: 3                    # latent channels
    ddconfig:
      double_z: true
      z_channels: 3
      resolution: 256               # input 256×256
      in_channels: 1                # binary mask = 1 channel
      out_ch: 1                     # output 1 channel
      ch: 128
      ch_mult: [1, 2, 4]            # f=4 (downscale 4 lần)
      num_res_blocks: 2
      attn_resolutions: []
      dropout: 0.0
    lossconfig:                      # L1 + KL (thay LPIPS vì mask binary)
      target: ldm.modules.losses.LPIPSWithDiscriminator
      params:
        disc_start: 50001
        kl_weight: 0.000001
        disc_weight: 0.0             # bỏ discriminator cho mask
```

### 5.2 Latent Diffusion Model

```yaml
model:
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    timesteps: 1000
    first_stage_key: "mask"          # batch["mask"] = GT mask
    cond_stage_key: "image"          # batch["image"] = RGB image
    image_size: 64                   # latent size (256/4)
    channels: 3
    conditioning_key: "concat"       # ghép RGB latent + mask latent

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 6               # 3 (mask latent) + 3 (RGB latent)
        out_channels: 3              # predict noise (3 channels)
        model_channels: 192
        attention_resolutions: [8, 4, 2]
        num_res_blocks: 2
        channel_mult: [1, 2, 3, 5]
        num_head_channels: 32

    first_stage_config:              # Mask AE (frozen, từ Stage 1)
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        ckpt_path: "checkpoints/mask_autoencoder.ckpt"
        # ... (config giống Stage 1)

    cond_stage_config:               # RGB AE (pretrained, frozen)
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        ckpt_path: "checkpoints/kl-f4/model.ckpt"  # download sẵn
        # ... (config cho RGB 3 channels)
```

---

## 6. Hyperparameters

### Stage 1 (Autoencoder)

| Parameter | Value | Giải thích |
|-----------|-------|-----------|
| Input | 256×256, 1 channel | GT mask crop + resize |
| Latent | 64×64×3 | Nhỏ 16 lần so với input |
| LR | 4.5e-6 | Theo LDM paper |
| Batch size | 4–8 | Tùy GPU |
| Epochs | ~100–200 | Đến khi rec_loss ≤ 0.01 |

### Stage 2 (Diffusion)

| Parameter | Value | Giải thích |
|-----------|-------|-----------|
| Timesteps | 1000 | Số bước noise (standard) |
| LR | 1e-4 | Theo LDM paper |
| Batch size | 8–16 | Latent nhỏ → fit nhiều |
| Training iters | 100K–200K | Đến convergence |
| DDIM steps (inference) | 50 | Số bước khử noise khi inference |
| t_start (inference) | 500 | Bắt đầu từ noise level 50% |

---

## 7. Cấu trúc thư mục

```
~/vy/Denoiser/LatentDiffusionDenoiser_Building/
├── configs/
│   ├── autoencoder/
│   │   └── mask_autoencoder_kl_f4.yaml
│   └── latent-diffusion/
│       └── ldm_mask_denoiser_building.yaml
├── data/
│   └── OEM_v2_Building -> ../../OEM_v2_Building
├── ldm/                                    # Từ CompVis repo
├── scripts/
│   ├── inference_denoise.py                # Denoise pseudo-labels
│   └── eval_building.py
├── checkpoints/
│   ├── kl-f4/model.ckpt                   # Pretrained RGB AE
│   └── mask_autoencoder.ckpt              # Trained mask AE
└── main.py
```

---

## 8. Training Commands

```bash
# Stage 1: Train mask autoencoder
CUDA_VISIBLE_DEVICES=0 python main.py \
    --base configs/autoencoder/mask_autoencoder_kl_f4.yaml \
    -t --gpus 0,

# Stage 2: Train latent diffusion
CUDA_VISIBLE_DEVICES=0 python main.py \
    --base configs/latent-diffusion/ldm_mask_denoiser_building.yaml \
    -t --gpus 0,

# Inference: Denoise pseudo-labels trên val set
python scripts/inference_denoise.py \
    --ldm_ckpt logs/<run_name>/checkpoints/last.ckpt \
    --config configs/latent-diffusion/ldm_mask_denoiser_building.yaml \
    --data_root data/OEM_v2_Building \
    --split val \
    --t_start 500 \
    --ddim_steps 50 \
    --output_dir results/ldm_refined

# Evaluate: So sánh refined vs pseudo vs GT
python scripts/eval_building.py \
    --data_root data/OEM_v2_Building \
    --split val \
    --refined_dir results/ldm_refined
```

---

## 9. Implementation Plan

| Phase | Thời gian | Nội dung |
|-------|----------|----------|
| **Phase 1** | 1–2 ngày | Clone repo, setup env, train mask autoencoder, validate reconstruct IoU ≥ 0.98 |
| **Phase 2** | 2–3 ngày | Tạo config, dataset class, train LDM, monitor val_loss |
| **Phase 3** | 1 ngày | Inference denoise pseudo-labels, evaluate IoU, ablation t_start |

---

## 10. Rủi ro & Giải pháp

| Rủi ro | Giải pháp |
|--------|----------|
| Autoencoder reconstruct mất boundary | Tăng model size, thay loss bằng BCE+Dice |
| UNet ignore RGB condition | Dùng cross-attention thay concat |
| Inference chậm | Giảm DDIM steps xuống 20–30 |
| Mask output không binary (giá trị trung gian) | Threshold ở 0.5, optional CRF |
