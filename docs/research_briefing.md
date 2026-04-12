# Research Briefing: Pseudo-label Noise Characterization for Satellite Image Segmentation

## Document Purpose

Tài liệu này tổng hợp toàn bộ phân tích thống kê về noise trong pseudo-label dataset, phục vụ nghiên cứu thiết kế bộ **denoiser cho semantic segmentation maps** trong ảnh viễn thám. Mục tiêu: cung cấp đầy đủ data insights để AI/researcher đề xuất kiến trúc và phương pháp denoising phù hợp — **không bias vào bất kỳ model cụ thể nào**.

---

## 1. Problem Statement

**Bài toán cốt lõi: Pseudo-label → Ground Truth Refinement**

Cho một bộ pseudo-label maps (output của model segmentation CISC-R) cho ảnh vệ tinh, **cùng với ground truth (GT) tương ứng cho training**, cần thiết kế một model học cách **chuyển pseudo-label (noisy) thành label map sát ground truth nhất có thể**.

Đây là bài toán **label refinement / label denoising** — không phải image denoising. Input và output đều là **discrete semantic segmentation maps** (integer class IDs), không phải continuous pixel values.

**Task formulation:**
- **Training:** Có paired data `(pseudo_label, ground_truth)` cho 2,189 images
- **Inference:** Nhận pseudo-label map mới (chưa có GT) → output refined label map
- **Goal:** Maximize chất lượng output so với GT thật (mIoU, boundary accuracy, per-class recall)

**Input/Output format:**
- **Input chính:** Label map H×W (integer, 8 classes) — pseudo-label bị noise ~30%
- **Conditioning (tùy chọn):** RGB satellite image H×W×3 — có thể cung cấp thêm visual context
- **Output (target):** Label map H×W (integer, 8 classes) — ground truth quality
- **Lưu ý:** Đây là **discrete-to-discrete mapping**, không phải continuous denoising

**Ứng dụng:** Land-use/land-cover (LULC) segmentation từ ảnh viễn thám đa phổ.

**Tại sao cần denoiser thay vì train lại segmentation model?**
- Vì dataset OpenEarthMap không phải ảnh RGB nào cũng có label
- teacher model (CISC-R) sinh ra pseudo label, nhung noise nhiều, cần khử noise để student model học
---

## 2. Dataset Overview

| Property | Value |
|----------|-------|
| Dataset name | OEM_v2_aDanh (OpenEarthMap variant) |
| Total images | 2,189 |
| Train / Val / Test | 1,751 / 218 / 220 |
| Image size | 1024 × 1024 pixels |
| Number of classes | 8 |
| Label format | Single-channel integer map (0–7) |
| Ignore index | 255 (nodata regions) |
| Geographic regions | 72 distinct cities/regions worldwide |
| Source | Multispectral satellite imagery |

**Classes:**

| ID | Class Name | GT Pixel Ratio | Description |
|----|-----------|---------------|-------------|
| 0 | Bareland | 1.92% | Exposed soil, sand, rock |
| 1 | Rangeland | 21.18% | Grassland, shrub, sparse vegetation |
| 2 | Developed | 18.29% | Urban/built-up impervious surfaces |
| 3 | Road | 6.64% | Paved/unpaved roads |
| 4 | Tree | 18.75% | Forest, dense canopy |
| 5 | Water | 3.26% | Rivers, lakes, ocean |
| 6 | Agriculture | 13.28% | Cropland, farmland |
| 7 | Building | 16.67% | Individual building footprints |

---

## 3. Overall Noise Statistics

| Metric | Value |
|--------|-------|
| Overall pixel accuracy | 70.11% |
| Overall noise ratio | **29.89%** |
| Mean IoU (mIoU) | 0.4907 |
| Mean precision | 0.7230 |
| Mean recall | 0.6137 |
| Mean F1 | 0.6521 |

→ Gần **1/3 pixels** bị gán sai class bởi CISC-R pseudo-labeler.

---

## 4. Noise Characterization — 10 Key Dimensions

### 4.1 Boundary Error Ratio

| Location | Error Rate | Pixels |
|----------|-----------|--------|
| **Edge (class boundary)** | **54.35%** | 130M / 240M |
| **Interior (non-boundary)** | **26.78%** | 503M / 1,878M |
| **Edge/Interior ratio** | **2.03×** | — |

**Per-class boundary error:**

| Class | Edge Error Rate | Interpretation |
|-------|----------------|----------------|
| Bareland | 92.13% | Gần như mọi biên đều sai |
| Water | 77.47% | Ranh giới nước bị confuse nặng |
| Rangeland | 65.94% | Biên vùng cỏ bị nhầm với Tree/Developed |
| Agriculture | 63.11% | Biên nông nghiệp nhầm với Rangeland |
| Road | 61.25% | Biên đường bị Developed nuốt |
| Tree | 49.21% | |
| Developed | 48.57% | |
| Building | 43.57% | Ít boundary error nhất |

**→ Key insight:** Noise tập trung **chủ yếu ở boundary**, không phải interior. Hơn 50% pixels tại biên bị sai. Bất kỳ denoiser nào cũng phải xử lý boundary error hiệu quả.

### 4.2 Spatial Noise Clustering (Error Morphology)

| Property | Value |
|----------|-------|
| Total noise clusters | 1,479,507 |
| Mean clusters/image | 675.9 ± 454.5 |
| **Median cluster size** | **8 pixels** |
| Mean cluster size | 428 pixels |
| Max cluster | 1,021,252 pixels |

**Cluster size distribution:**

| Size Range | Count | % | Cumulative % |
|-----------|-------|---|-------------|
| 1–4 px | 627,464 | 42.4% | 42.4% |
| 5–9 px | 143,206 | 9.7% | 52.1% |
| 10–49 px | 325,609 | 22.0% | 74.1% |
| 50–99 px | 121,042 | 8.2% | 82.3% |
| 100–499 px | 172,295 | 11.6% | 93.9% |
| 500–999 px | 36,174 | 2.4% | 96.4% |
| 1k–5k px | 38,833 | 2.6% | 99.0% |
| 5k–50k px | 12,935 | 0.9% | 99.9% |
| >50k px | 1,949 | 0.1% | 100% |

**→ Key insight:** **74% noise clusters là dưới 50 pixels** — chủ yếu salt-and-pepper / boundary jitter. Tuy nhiên 0.1% clusters lớn (>50k px) chiếm phần lớn noisy pixels → có cả region-level confusion. Noise có **bimodal distribution**: rất nhỏ (boundary) hoặc rất lớn (class confusion vùng).

### 4.3 Per-Region Noise (Geographic Variation)

**72 regions** với noise rate từ **14.06%** đến **57.12%**:

| Top Noisiest | Noise Rate | Top Cleanest | Noise Rate |
|-------------|------------|-------------|------------|
| Western Sahara | 57.12% | Zachodniopomorskie (PL) | 14.06% |
| Paris (FR) | 48.88% | Swietokrzyskie (PL) | 17.65% |
| Shanghai (CN) | 40.63% | Wielkopolskie (PL) | 18.49% |
| Ulaanbaatar (MN) | 40.42% | Baybay (PH) | 19.20% |
| Khartoum (SD) | 40.26% | Malopolskie (PL) | 20.53% |

- **Std of noise rate across regions:** σ = 0.0742
- **Coefficient of variation:** 0.2493 (25%)

**→ Key insight:** Poland regions consistently clean (~15–20%), arid/dense-urban regions (Paris, Shanghai, Khartoum) much noisier (~40–57%). CISC-R performance varies by **4× across geography**. Noise patterns are geographically non-stationary.

### 4.4 Class-conditional Transition Matrix

Ma trận **P(pseudo=j | GT=i)** — xác suất class i bị CISC-R gán thành class j:

| GT \ Pseudo | Bareland | Rangeland | Developed | Road | Tree | Water | Agriculture | Building |
|---|---|---|---|---|---|---|---|---|
| **Bareland** | **0.082** | 0.110 | **0.683** | 0.010 | 0.015 | 0.048 | 0.048 | 0.003 |
| **Rangeland** | 0.005 | **0.578** | 0.146 | 0.009 | **0.165** | 0.013 | 0.065 | 0.019 |
| **Developed** | 0.006 | 0.056 | **0.716** | 0.068 | 0.023 | 0.004 | 0.012 | **0.115** |
| **Road** | 0.001 | 0.028 | **0.313** | **0.614** | 0.017 | 0.003 | 0.009 | 0.016 |
| **Tree** | 0.001 | **0.132** | 0.043 | 0.004 | **0.790** | 0.010 | 0.013 | 0.007 |
| **Water** | 0.003 | **0.119** | **0.146** | 0.017 | 0.027 | **0.625** | 0.051 | 0.012 |
| **Agriculture** | 0.004 | **0.142** | 0.049 | 0.004 | 0.031 | 0.010 | **0.756** | 0.004 |
| **Building** | 0.000 | 0.009 | **0.158** | 0.003 | 0.007 | 0.001 | 0.004 | **0.818** |

**Class retention rate (diagonal):**
```
Bareland     :  8.2% ███
Rangeland    : 57.8% ███████████████████████
Developed    : 71.6% ████████████████████████████
Road         : 61.4% ████████████████████████
Tree         : 79.0% ███████████████████████████████
Water        : 62.5% █████████████████████████
Agriculture  : 75.6% ██████████████████████████████
Building     : 81.8% ████████████████████████████████
```

**Top confusion patterns:**

| Rank | GT → Pseudo | Pixels | % of GT |
|------|------------|--------|---------|
| 1 | Bareland → Developed | 27.8M | **68.3%** |
| 2 | Rangeland → Tree | 74.2M | 16.5% |
| 3 | Rangeland → Developed | 65.6M | 14.6% |
| 4 | Building → Developed | 55.7M | 15.8% |
| 5 | Road → Developed | 44.1M | 31.3% |
| 6 | Tree → Rangeland | 52.6M | 13.3% |

**→ Key insights:**
- **Bareland is catastrophically destroyed**: chỉ giữ 8.2%, 68% → Developed
- **Developed là "noise sink"**: hấp thụ lỗi từ 5+ class khác
- **Noise KHÔNG symmetric**: P(Rangeland→Tree) ≠ P(Tree→Rangeland) → transition matrix là asymmetric
- Ma trận này **có thể dùng làm learned noise prior** cho bất kỳ denoising framework nào

### 4.5 Noise Spatial Heatmap (Position in Image)

Error rate theo vùng 4×4 grid (mỗi ô = 256×256 pixels):

```
[0.298  0.305  0.303  0.308]
[0.298  0.296  0.296  0.301]
[0.295  0.294  0.290  0.299]
[0.298  0.299  0.301  0.305]
```

| Zone | Error Rate |
|------|-----------|
| Corner zones | 30.21% |
| Center zones | 29.39% |
| Border zones | 30.07% |
| Inner zones | 29.39% |

**→ Key insight:** Noise phân bố **gần như đồng đều** (max-min = 1.75%). Không có context-window bias. Noise là **class-dependent và boundary-dependent**, không phải position-dependent.

### 4.6 Small Object Error Rate

| Region Size | Num Regions | Accuracy | Gap from Huge |
|-------------|------------|----------|---------------|
| **Micro (<50px)** | 392,779 | **21.17%** | -51.75% |
| Tiny (50–100px) | 140,078 | 30.83% | -42.09% |
| Small (100–500px) | 323,822 | 48.55% | -23.37% |
| Medium (500–2k) | 189,449 | 64.00% | -8.92% |
| Large (2k–10k) | 88,022 | 68.49% | -4.43% |
| **Huge (>10k px)** | 31,154 | **72.92%** | baseline |

**→ Key insight:** Object size là predictor mạnh nhất cho pseudo-label quality. Object nhỏ (<50px) chỉ đúng **21%** — gần như random cho 8 classes. Denoiser cần xử lý tốt small objects hoặc accept chúng là unrecoverable.

### 4.7 Label Smoothness / Entropy Analysis

| Metric | Value |
|--------|-------|
| Mean local entropy (pseudo) | 0.9504 ± 0.3683 |
| Mean local entropy (GT) | 1.0031 ± 0.4008 |
| Entropy diff (pseudo − GT) | **−0.0527** ± 0.1461 |
| Images where pseudo > GT entropy | 701/2189 (32%) |

**→ Key insight:** Pseudo-labels **smoother hơn GT** (entropy thấp hơn). CISC-R có xu hướng **over-smooth** — merge các class nhỏ/ít vào class lớn/phổ biến. Pseudo "mất" chi tiết hơn là "thêm" noise random. Điều này consistent với quan sát Bareland bị nuốt vào Developed.

### 4.8 Confidence Map Analysis

**KHÔNG CÓ DỮ LIỆU.** Dataset chỉ chứa hard pseudo-labels (argmax output), không có softmax probabilities hay confidence scores. Nếu re-run CISC-R với `--save-probabilities`, có thể dùng confidence làm quality weight.

### 4.9 Geographic Bias — Per-Region Per-Class Breakdown

Noise rate per class trong 5 regions noisiest vs 5 regions cleanest:

| Region | Overall | Bareland | Rangeland | Developed | Road | Tree | Water | Agriculture | Building |
|--------|---------|----------|-----------|-----------|------|------|-------|-------------|----------|
| Western Sahara | 57.1% | 86.4% | 63.0% | 15.8% | 68.6% | 15.8% | 87.7% | 66.2% | 16.9% |
| Paris | 48.9% | 100% | 57.6% | 58.0% | 29.7% | 65.2% | 80.1% | — | 19.8% |
| Shanghai | 40.6% | 100% | 51.4% | 41.0% | 52.4% | 54.0% | 69.7% | 46.1% | 27.8% |
| Zachodniopomorskie | 14.1% | — | 18.7% | 27.3% | 25.3% | 7.9% | 3.1% | 26.1% | 14.5% |
| Swietokrzyskie | 17.7% | — | 16.3% | 46.2% | 36.1% | 8.4% | 1.2% | 28.1% | 10.3% |

**→ Key insight:** Noise patterns khác nhau hoàn toàn theo geography:
- Arid regions: Bareland/Water noise cực cao
- Dense urban: Developed/Building confusion
- European agriculture: khá clean (Poland ~15%)
- **Same class, different noise rate by 10×** across regions → noise là non-stationary

### 4.10 Class Imbalance Impact

| Class | GT Frequency | Noise Rate |
|-------|-------------|------------|
| Bareland | 1.92% (rarest) | **91.81%** (highest) |
| Water | 3.26% | 37.45% |
| Road | 6.64% | 38.63% |
| Agriculture | 13.28% | 24.41% |
| Building | 16.67% | **18.19%** (lowest) |
| Tree | 18.75% | 21.04% |
| Developed | 18.29% | 28.37% |
| Rangeland | 21.18% | 42.23% |

- **Pearson correlation (freq vs noise):** r = **−0.634** (p = 0.091)
- **Spearman correlation:** ρ = −0.357

**→ Key insight:** Moderate negative correlation — **minority classes bị noise nhiều hơn**. Bareland (2% of data) bị 92% noise, Building (17%) chỉ 18% noise. Tuy nhiên Rangeland (21%) vẫn bị 42% noise → inter-class visual similarity cũng là factor lớn, không chỉ frequency.

---

## 5. Tổng hợp: 10 Data Facts Quan Trọng Nhất

1. **~30% pixels bị sai** — pixel accuracy 70.1%, mIoU 0.49
2. **Boundary noise cao gấp 2× interior** — 54.4% vs 26.8%
3. **74% noise clusters dưới 50px** — chủ yếu boundary jitter, nhưng có 0.1% vùng noise >50k px
4. **Noise phân bố đều trong ảnh** — không có spatial/position bias
5. **Bareland gần như bị xóa**: chỉ 8.2% pixels đúng, 68% bị gán thành Developed
6. **Developed là noise sink**: hấp thụ lỗi từ Bareland (68%), Road (31%), Building (16%), Water (15%)
7. **Pseudo-labels smoother hơn GT**: entropy thấp hơn, CISC-R over-smooth thay vì random noise
8. **Noise variance cao theo geography**: σ = 7.4%, range 14%–57% — CISC-R performance rất non-stationary
9. **Small objects rất unreliable**: <50px accuracy = 21%, >10k px accuracy = 73%
10. **Minority classes bị ảnh hưởng nhiều hơn**: Pearson r = −0.63

---

## 6. Noise Characterization Summary (for Model Design)

### Noise không phải random — nó có structure rõ ràng:

| Noise Property | Finding | Implication |
|---------------|---------|-------------|
| **Type** | Structured, class-dependent | Không phải Gaussian/uniform; cần class-conditional modeling |
| **Location** | Boundary-concentrated (2× interior) | Cần boundary-aware processing |
| **Morphology** | Bimodal: nhiều cluster nhỏ + ít cluster rất lớn | Cần xử lý cả local (pixel) và regional (patch) errors |
| **Symmetry** | Asymmetric transitions | P(A→B) ≠ P(B→A); transition matrix là non-symmetric |
| **Spatial** | Uniform across image position | Không cần position-aware correction |
| **Scale** | Strong size-dependency | Small objects nearly unrecoverable from pseudo-labels alone |
| **Geographic** | Highly non-stationary | Noise patterns vary 4× across regions |
| **Smoothness** | Over-smoothed (entropy lower than GT) | Pseudo merges details, doesn't add random noise |
| **Class bias** | Minority classes severely affected | Class-balanced approach needed |

### Transition matrix (ma trận noise) dạng 8×8:

```
Available at: analysis_output/advanced/transition_matrix.npy
Format: numpy float64 array, shape (8, 8)
Row i, col j = P(pseudo=j | GT=i)
Rows sum to 1.0
```

---

## 7. Câu Hỏi Nghiên Cứu Mở
-> Literature Review  
-> Research Gap
-> Gợi ý các phương án thiết kế model để giải quyết

---

*Generated from 2,189 images across 72 geographic regions. From OpenEarthMap dataset, Pseudo label from CISC-R model. Full analysis scripts available in `tools/analyze_dataset.py`, `tools/analyze_advanced.py`, `tools/analyze_10metrics.py`.*
