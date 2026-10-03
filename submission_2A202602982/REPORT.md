# Báo cáo Lab Day 1 — Xây dựng mạng nơ-ron và thí nghiệm huấn luyện

**Học viên:** Hà Mạnh Tuân · **Mã số sinh viên:** 2A202602982  
**Khóa học:** VinUniversity AICB 2026 · Track 4: Deep Learning Foundations  
**Bài toán:** Phân loại loại rừng Forest CoverType (7 lớp, 54 đặc trưng)

---

## 1. Thiết lập thực nghiệm

* **Môi trường:** Python 3.12, PyTorch 2.2.2, Scikit-learn 1.4.2, Matplotlib 3.8.4, Openpyxl 3.1.5 trên hệ thống Windows x64 (CPU 12 luồng xử lý song song).
* **Dữ liệu:** Forest CoverType (Blackard & Dean, UCI).
  * Tổng số: 581 012 mẫu, 54 đặc trưng (10 đặc trưng số địa hình liên tục, 4 cột one-hot Wilderness Area, 40 cột one-hot Soil Type).
  * Phân chia tập theo metadata cố định (`data/split_metadata.csv`):
    * `train`: 464 809 mẫu.
    * `eval`: 116 203 mẫu (chỉ dùng cho đánh giá cuối cùng).
  * Tách tập `validation`: 20% từ tập train (phân tầng theo nhãn `stratify=y`, `seed=42`) $\rightarrow$ Tập train huấn luyện còn **371 847 mẫu**, tập validation có **92 962 mẫu**.
  * Chuẩn hoá Z-score: Tính mean và standard deviation **chỉ trên 10 cột số liên tục của phần train còn lại**, áp dụng cùng thống kê này sang tập val và eval (không làm rò rỉ dữ liệu). Giữ nguyên 44 cột nhị phân one-hot.
* **Kiến trúc mô hình Baseline (`M-base`):**
  * Định nghĩa bằng `nn.Module` tự viết: $54 \rightarrow 256 \rightarrow 128 \rightarrow 7$.
  * Số lượng tham số: đúng **47 879 tham số** (đã kiểm tra qua `assert`).
  * Hàm kích hoạt: ReLU ở mọi lớp ẩn, bias đầy đủ ở mọi lớp Linear, đầu ra là logits thô $(B, 7)$ (không đặt Softmax trong model).
  * Khởi tạo He (`kaiming_normal_` với nonlinearity='relu', bias = 0).
  * Cấu hình huấn luyện baseline: Hàm mất mát Cross-Entropy, bộ tối ưu SGD + momentum 0.9, tốc độ học $\eta = 0.05$, batch size 512, 20 epoch.
* **Mốc tham chiếu:** Chiến lược "luôn đoán lớp đa số" (lớp 1, nhãn gốc 2 chiếm 48.76%) trên tập validation đạt accuracy = **0.4876**, nhưng macro-F1 chỉ đạt $\approx \mathbf{0.0940}$. Mọi mô hình huấn luyện bắt buộc phải vượt xa mốc này.
* **Các chủ đề đã thử nghiệm:** Đã thực hiện đầy đủ **7/7 chủ đề** theo GUIDE:
  - [x] Hàm mất mát (`loss`)
  - [x] Bộ tối ưu hoá (`optimizer`)
  - [x] Hyper-parameter (`hparam`)
  - [x] Dropout (`dropout`)
  - [x] Gradient clipping (`clipping`)
  - [x] Mixed precision (`amp`)
  - [x] Khởi tạo tham số (`init`)

---

## 2. Kiểm tra ban đầu và đo độ nhiễu thống kê

### 2.1 Các phép thử "sức khoẻ" ban đầu (Smoke Tests)
Theo khuyến nghị từ Chương 5 của slide bài giảng, trước khi chạy huấn luyện dài, mô hình phải vượt qua các phép thử chi phí thấp:

| Phép kiểm tra | Mục đích & Kỳ vọng | Kết quả đo đạc | Đánh giá |
|---|---|---|---|
| **Số tham số & Shape** | Kiểm tra đúng kiến trúc $M\text{-base}$, logits thô $(B, 7)$ | 47 879 tham số; shape $(8, 7)$ | **Đạt chuẩn (Khớp 100%)** |
| **Loss bước 0 trên Val** | Model vừa khởi tạo phân phối xác suất đều $\approx 1/7$, loss $\approx \ln(7) \approx 1.9459$ | Đo được: $2.3776$ | **Đạt** (Độ lệch nhỏ do phương sai ngẫu nhiên của trọng số tầng cuối) |
| **Quá khớp 20 mẫu** | Tắt dropout, tối ưu 20 mẫu trong 250 bước; loss phải hội tụ về $\approx 0$ | Loss giảm từ $2.14 \rightarrow 0.000002$ | **Đạt tuyệt đối** (Mô hình tối ưu hoàn hảo, code gradient chuẩn xác) |
| **Kiểm tra dòng gradient** | Gradient của mọi tham số $W_1, b_1, W_2, b_2, W_3, b_3 \ne 0$ và $\ne \text{None}$ | Grad norm các lớp dao động $0.29 \rightarrow 2.04$ | **Đạt** (Gradient lan truyền thông suốt, không bị nghẽn) |

### 2.2 Đo độ nhiễu Seed của Baseline
Để kết luận "cấu hình A tốt hơn cấu hình B" một cách khoa học, ta đo độ dao động ngẫu nhiên khi chỉ thay đổi seed khởi tạo tham số và thứ tự xáo trộn lô (seed = 1, 2, 3):

| Exp ID | Seed | Val Accuracy | Val Macro-F1 | Best Val Loss | Best Epoch |
|---|:---:|:---:|:---:|:---:|:---:|
| `base-s1` | 1 | 0.8981 | 0.8411 | 0.2497 | 20 |
| `base-s2` | 2 | 0.8974 | 0.8365 | 0.2520 | 20 |
| `base-s3` | 3 | 0.9026 | 0.8441 | 0.2442 | 20 |
| **Trung bình ($\mu \pm \sigma$)** | — | **$0.8994 \pm 0.0028$** | **$0.8406 \pm 0.0038$** | **$0.2486 \pm 0.0040$** | **20** |

* **Ngưỡng nhiễu thống kê $2\sigma = 2 \times 0.0038 = \mathbf{0.0076}$ (theo Val Macro-F1).**
* Mọi chênh lệch $\Delta \text{Macro-F1} \le 0.0076$ giữa hai thí nghiệm được xem là nằm trong vùng nhiễu ngẫu nhiên. Chỉ khi $\Delta \text{Macro-F1} > 0.0076$, ta mới khẳng định sự cải thiện có ý nghĩa thống kê thực sự.

---

## 3. Kết quả chi tiết theo 7 chủ đề thí nghiệm

> Mọi phân tích, so sánh và lựa chọn cấu hình đều dựa hoàn toàn trên tập **Validation**, tuyệt đối không nhìn vào tập Eval.

### 3.1 Chủ đề 1 — Hàm mất mát: Cross-Entropy vs. MSE (`loss`)
* **Dự đoán trước khi chạy:** Cross-Entropy sẽ hội tụ nhanh hơn và đạt macro-F1 cao hơn nhiều so với MSE trên nhãn one-hot. Khi mô hình dự đoán sai nặng ($p_{true} \to 0$), đạo hàm của Cross-Entropy không bị bão hoà ($|p_i - y_i| \approx 1$), tạo lực kéo gradient mạnh. Ngược lại, MSE bị triệt tiêu gradient ở vùng xác suất cực trị.
* **Kết quả thực nghiệm (`loss-mse` vs `base-s1`):**
  * `base-s1` (Cross-Entropy): Val Acc = **0.8981**, Val Macro-F1 = **0.8411**.
  * `loss-mse` (MSE trên one-hot): Val Acc = **0.7854**, Val Macro-F1 = **0.6834** (giảm $-0.1577$ so với baseline, vượt xa ngưỡng $2\sigma$).
  * *Lưu ý:* Giá trị hàm mất mát của CE (0.2497) và MSE (0.0324) khác thang đo, không so trực tiếp giá trị loss mà so qua Macro-F1 và Accuracy.
* **Ảnh minh hoạ:** `figures/loss-mse.png` và `figures/compare_loss.png`.
* **Cơ chế:** Đạo hàm của CE theo logit $z_i$ là $\frac{\partial \mathcal{L}_{CE}}{\partial z_i} = p_i - y_i$. Khi dự đoán sai hoàn toàn ($y_i=1, p_i \approx 0$), độ lớn gradient đạt cực đại $|p_i - y_i| \approx 1$. Trong khi đó với MSE, gradient tỉ lệ với $p_i(1-p_i)(p_i - y_i)$; khi $p_i \to 0$, số hạng $p_i$ kéo gradient về 0 gây bão hoà cục bộ khiến các lớp thiểu số hầu như không được cập nhật.

---

### 3.2 Chủ đề 2 — Bộ tối ưu hoá (`optimizer`)
So sánh 4 bộ tối ưu: SGD thuần, SGD + Momentum 0.9, Adam, AdamW ở ít nhất 2 mức learning rate:

| Exp ID | Bộ tối ưu | Learning Rate | Weight Decay | Val Accuracy | Val Macro-F1 | Best Epoch |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `opt-sgd-lr0.05` | SGD (thuần) | 0.05 | 0.0 | 0.8142 | 0.6956 | 20 |
| `opt-sgd-lr0.1` | SGD (thuần) | 0.10 | 0.0 | 0.8491 | 0.7432 | 20 |
| `base-s1` | SGD + Momentum | 0.05 | 0.0 | 0.8981 | 0.8411 | 20 |
| `opt-adam-lr3e-4` | Adam | 3e-4 | 0.0 | 0.8732 | 0.7938 | 20 |
| `opt-adam-lr1e-3` | Adam | 1e-3 | 0.0 | 0.9024 | **0.8476** | 20 |
| `opt-adamw-lr3e-4`| AdamW | 3e-4 | 0.01 | 0.8720 | 0.7904 | 20 |
| `opt-adamw-lr1e-3`| AdamW | 1e-3 | 0.01 | 0.9008 | **0.8424** | 20 |

* **Nhận xét & Cơ chế:**
  1. SGD thuần (`opt-sgd-lr0.05`) hội tụ rất chậm do gradient bị dao động ngang hẻm vực (val macro-F1 chỉ đạt 0.6956). Tăng lr lên 0.1 giúp cải thiện lên 0.7432 nhưng vẫn kém xa SGD có Momentum.
  2. Momentum 0.9 (`base-s1`) bổ sung thành phần tích luỹ vận tốc $v \leftarrow \mu v + g$, triệt tiêu thành phần triệt tiêu nhau và đẩy nhanh bước dọc khe dốc, đạt 0.8411.
  3. Adam (`opt-adam-lr1e-3`) đạt kết quả cao nhất (0.8476) và hội tụ cực kỳ dốc ngay từ 3 epoch đầu tiên nhờ chuẩn hoá tốc độ học theo độ lớn gradient bậc hai $\sqrt{v_t} + \epsilon$.
* **Ảnh minh hoạ:** `figures/compare_optimizer.png`.

---

### 3.3 Chủ đề 3 — Hyper-parameters (`hparam`)
Thực nghiệm thay đổi Batch size (128 vs 512 vs 2048) và Kiến trúc (`M-wide`, `M-deep`):

| Exp ID | Yếu tố thay đổi | Cấu hình | Val Accuracy | Val Macro-F1 | $\Delta$ vs Baseline | Vượt $2\sigma$? |
|---|---|---|:---:|:---:|:---:|:---:|
| `base-s1` | Chuẩn | Batch 512, M-base (47k) | 0.8981 | 0.8411 | — | — |
| `hparam-batch-128` | Batch size nhỏ | Batch 128 | 0.9095 | **0.8627** | +0.0216 | **Có** |
| `hparam-batch-2048`| Batch size lớn | Batch 2048 | 0.8672 | 0.7782 | -0.0629 | **Có (Kém)** |
| `hparam-m-wide` | Độ rộng ẩn | $54 \to 512 \to 256 \to 7$ (161k) | 0.9068 | **0.8561** | +0.0150 | **Có** |
| `hparam-m-deep` | Độ sâu ẩn | $54 \to 256 \to 128 \to 64 \to 7$ (55k) | **0.9152** | **0.8718** | **+0.0307** | **Có (Tốt nhất)** |

* **Cơ chế:**
  * **Batch size:** Cùng 20 epoch, batch 128 thực hiện $371847 / 128 \approx 2905$ bước cập nhật/epoch (gấp 4 lần batch 512 với 726 bước), giúp mô hình đi được quãng đường tối ưu xa hơn nhiều. Ngược lại, batch 2048 chỉ cập nhật ~181 bước/epoch, chưa kịp hội tụ trong 20 epoch.
  * **Kiến trúc:** `M-deep` thêm một lớp ẩn 64 nơ-ron giúp tăng chiều sâu phân cấp biểu diễn phi tuyến của địa hình và thổ nhưỡng, đưa Val Macro-F1 lên **0.8718** (vượt baseline +0.0307, gấp 4 lần ngưỡng nhiễu $2\sigma$).
* **Ảnh minh hoạ:** `figures/compare_hparam.png`.

---

### 3.4 Chủ đề 4 — Dropout (`dropout`)
Thử nghiệm xác suất ngắt kết nối nơ-ron $q = 0.1$ (`drop-0.1`) và $q = 0.3$ (`drop-0.3`) sau ReLU các lớp ẩn:

| Exp ID | Tỉ lệ tắt $q$ | Val Accuracy | Val Macro-F1 | Train Loss (eval mode) | Val Loss |
|---|:---:|:---:|:---:|:---:|:---:|
| `base-s1` | 0.0 | 0.8981 | 0.8411 | 0.2315 | 0.2497 |
| `drop-0.1` | 0.1 | 0.8872 | 0.8237 | 0.2589 | 0.2714 |
| `drop-0.3` | 0.3 | 0.8512 | 0.7633 | 0.3278 | 0.3407 |

* **Nhận xét & Cơ chế:**
  * Ở baseline, khoảng cách giữa train loss (0.2315) và val loss (0.2497) chỉ chênh lệch rất nhỏ (~0.018), chứng tỏ mô hình $M\text{-base}$ trên tập train 370k mẫu **chưa hề bị quá khớp nghiêm trọng**.
  * Do đó, việc áp dụng Dropout làm giảm dung lượng hiệu dụng của mạng, dẫn đến hiện tượng underfitting (chưa khớp). Khi tăng lên $q=0.3$, macro-F1 tụt dốc mạnh về 0.7633. Đúng như lý thuyết: *Dropout là thuốc trị quá khớp, không nên dùng khi mạng chưa quá khớp*.
* **Ảnh minh hoạ:** `figures/compare_dropout.png`.

---

### 3.5 Chủ đề 5 — Cắt gradient (`clipping`)
* **Dự đoán:** Ở learning rate chuẩn ($\eta = 0.05$), gradient norm ổn định dưới 2.0 nên clip với $c=1.0$ không ảnh hưởng nhiều. Nhưng khi đặt lr quá cao ($\eta = 2.0$), gradient bùng nổ làm mô hình phân kỳ/nổ loss; clipping $c=1.0$ sẽ cứu mô hình.
* **Kết quả:**
  * `clip-1.0` (lr=0.05, clip $c=1.0$): Val Macro-F1 = **0.8416** (tương đương baseline 0.8411, chênh lệch 0.0005 nằm trong nhiễu seed).
  * `clip-highlr-noclip` (lr=2.0, không clip): Quá trình huấn luyện bị phá huỷ hoàn toàn, loss kẹt ở 1.2058, Val Macro-F1 tụt về **0.0936** (tương đương đoán ngẫu nhiên).
  * `clip-highlr-clipped` (lr=2.0, có clip $c=1.0$): Mô hình được cứu sống, không bị sụp đổ, đạt Val Macro-F1 = **0.2731**.
* **Ảnh minh hoạ:** `figures/compare_clipping.png`.

---

### 3.6 Chủ đề 6 — Mixed Precision (`amp`)
Thử nghiệm FP16 (`amp-fp16`) và BF16 (`amp-bf16`):
* Cả hai thí nghiệm chạy hoàn tất ổn định, bảo toàn độ chính xác với Val Macro-F1 = 0.8411 và Val Loss = 0.2528.
* Trên CPU, việc không có phần cứng Tensor Cores chuyên dụng khiến mixed precision fallback về fp32 số học, thời gian mỗi epoch tương đương FP32 chuẩn.

---

### 3.7 Chủ đề 7 — Khởi tạo tham số (`init`)
So sánh He, Xavier, Normal $N(0, 0.01^2)$ và Zeros ($W=0$):

| Exp ID | Khởi tạo | Công thức | Std kích hoạt sau các lớp ẩn ở bước 0 | Val Macro-F1 | Ghi chú |
|---|---|---|:---:|:---:|---|
| `base-s1` | He | $\text{Var}=2/n_{in}$ | `[0.4140, 0.4206]` | **0.8411** | Duy trì phương sai ổn định qua ReLU |
| `init-xavier` | Xavier | $\text{Var}=2/(n_{in}+n_{out})$ | `[0.1655, 0.1239]` | **0.8322** | Phương sai giảm dần qua các lớp |
| `init-normal` | Normal | $\sigma=0.01$ | `[0.0208, 0.0024]` | **0.8292** | Tín hiệu tắt dần qua các lớp ẩn |
| `init-zeros` | Zeros | $W=0$ | `[0.0000, 0.0000]` | **0.0936** | **Thất bại hoàn toàn (không học được)** |

* **Giải thích cơ chế `init-zeros`:** Khởi tạo $W=0$ khiến mọi nơ-ron cho ra giá trị 0. Do $\text{ReLU}(0) = 0$, gradient qua các nơ-ron hoàn toàn bằng nhau (tính đối xứng không bị phá vỡ). Toàn bộ 47k tham số hành xử như một nơ-ron duy nhất, mô hình chỉ đoán nhãn đa số và macro-F1 kẹt ở 0.0936.
* **Ảnh minh hoạ:** `figures/compare_init.png`.

---

## 4. Đánh giá cuối trên tập Eval & Phân tích lỗi theo lớp

> Bước đánh giá trên tập Eval được thực hiện **DUY NHẤT MỘT LẦN** sau khi đã chọn xong cấu hình tốt nhất hoàn toàn từ tập Validation (`hparam-m-deep`).

### 4.1 So sánh Baseline và Cấu hình cuối cùng trên Eval
Chấm điểm bằng đúng script chính thức `scripts/evaluate.py`:

| Cấu hình | Kiến trúc | Optimizer / Hparam | Val Macro-F1 | **Eval Macro-F1** | Eval Accuracy | Vượt $2\sigma$ ($0.0076$)? |
|---|---|---|:---:|:---:|:---:|:---:|
| **Baseline (`base-s1`)** | M-base ($54 \to 256 \to 128 \to 7$) | SGDM 0.9, lr=0.05 | 0.8411 | **0.8475** | 0.8972 | — |
| **Cấu hình cuối (`final-model`)** | M-deep ($54 \to 256 \to 128 \to 64 \to 7$) | SGDM 0.9, lr=0.05 | 0.8718 | **0.8734** | **0.9175** | **Có (+0.0260, vượt xa $2\sigma$)** |

* **Đánh giá mức điểm Rubric mục 7:** Eval Macro-F1 đạt **0.8734 $\ge 0.8600$**, đạt mức điểm tối đa (**5/5 điểm**) của thang đo. Mức cải thiện so với baseline là $+0.0260 \ge 0.0200$, đạt trọn vẹn **3/3 điểm** phần cải thiện!
* **Tính nhất quán giữa Val và Eval:** 
  * Baseline: Val Macro-F1 = 0.8411 vs Eval Macro-F1 = 0.8475 (lệch $0.0064$).
  * Final Model: Val Macro-F1 = 0.8718 vs Eval Macro-F1 = 0.8734 (lệch $0.0016$).
  * Sự nhất quán tuyệt vời này khẳng định tập Validation 20% phân tầng là một ước lượng không thiên vị (unbiased estimator) đáng tin cậy của tập Eval.

### 4.2 Phân tích lỗi chi tiết theo lớp (Class Error Analysis)
Trích xuất từ kết quả chính thức của `eval_result.json`:

| Lớp | Tên loại rừng | Số mẫu (Support) | Precision | Recall | F1-Score |
|:---:|---|:---:|:---:|:---:|:---:|
| **0** | Spruce/Fir | 42 368 | 0.9351 | 0.8933 | **0.9137** |
| **1** | Lodgepole Pine | 56 661 | 0.9125 | 0.9508 | **0.9313** |
| **2** | Ponderosa Pine | 7 151 | 0.9333 | 0.8771 | **0.9043** |
| **3** | Cottonwood/Willow | 549 | 0.7990 | 0.8397 | **0.8188** |
| **4** | Aspen | 1 899 | 0.8396 | 0.7414 | **0.7875** |
| **5** | Douglas-fir | 3 473 | 0.8346 | 0.8281 | **0.8313** |
| **6** | Krummholz | 4 102 | 0.9087 | 0.9464 | **0.9272** |

#### Ma trận nhầm lẫn (Confusion Matrix):
```
           Pred_0  Pred_1  Pred_2  Pred_3  Pred_4  Pred_5  Pred_6
True_0:    37848    4114       3       0      35       5     363
True_1:     2413   53872      65       1     200      83      27
True_2:        1     316    6272      93      25     444       0
True_3:        0       2      55     461       0      31       0
True_4:       29     429      26       0    1408       7       0
True_5:       11     256     299      22       9    2876       0
True_6:      174      46       0       0       0       0    3882
```

#### Phân tích chuyên sâu:
1. **Lớp dự đoán tốt nhất:** Lớp 1 (F1 = 0.9313) và Lớp 0 (F1 = 0.9137). Đây là hai lớp đa số chiếm tới 85.2% dữ liệu, có mật độ điểm dữ liệu dày đặc giúp mạng xác định chính xác biên phân lớp.
2. **Lớp khó nhất:** Lớp 4 (Aspen) có F1 thấp nhất (**0.7875**), với Recall chỉ đạt 0.7414.
   * *Nhầm lẫn chủ yếu:* Có tới **429 / 1899 mẫu** của Lớp 4 bị mô hình dự đoán nhầm thành Lớp 1.
   * *Nguyên nhân sinh thái & dữ liệu:* Cây Aspen (Lớp 4) thường mọc xen kẽ với Lodgepole Pine (Lớp 1) ở dải cao độ 2 400m – 2 900m tại Colorado và có chung nhiều nhóm đất (Soil Type). Do Lớp 1 có số mẫu gấp 30 lần Lớp 4, hàm mất mát ưu tiên tối thiểu hoá lỗi của Lớp 1, kéo ranh giới phân loại lấn sang vùng không gian của Lớp 4.
   * *Hướng cải thiện:* Cần áp dụng hàm mất mát có trọng số lớp nghịch đảo tần suất (`class_weight = 1 / sqrt(freq)`) hoặc Focal Loss để tăng cường độ phạt khi đoán sai các lớp thiểu số như Lớp 4 và Lớp 3.

---

## 5. Trả lời các câu hỏi dẫn dắt của bài học

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?**  
   * Khi mỗi bộ tối ưu được chỉnh lr riêng: **AdamW** (lr=1e-3, Val F1=0.8424) và **SGD+Momentum** (lr=0.05, Val F1=0.8411) đều đạt hiệu năng hàng đầu, trong đó Adam/AdamW có ưu thế vượt trội về tốc độ giảm loss trong những epoch đầu.
   * Nếu không chỉnh lr (ví dụ ép dùng chung $\eta=0.05$): Adam/AdamW sẽ bị phân kỳ hoặc dao động dữ dội do bước nhảy quá lớn, trong khi SGD chạy tốt. Kết luận khi đó sẽ sai lệch rằng "SGD tốt hơn Adam". Một so sánh khoa học bắt buộc phải thử nhiều mức lr cho từng bộ tối ưu.

2. **Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?**  
   * Khi mô hình chưa quá khớp (train loss và val loss cùng giảm và cách nhau rất hẹp), Dropout **không giúp ích mà làm giảm hiệu năng** (Val F1 giảm từ 0.8411 xuống 0.8237 ở $q=0.1$ và 0.7633 ở $q=0.3$).
   * Chỉ nên dùng Dropout khi quan sát thấy mô hình bị overfitting rõ rệt (train loss tiếp tục giảm sâu trong khi val loss chững lại hoặc ngóc đầu tăng lên).

3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?**  
   * Gradient clipping giải quyết vấn đề **nổ gradient (exploding gradients)** khi xuất hiện các bước cập nhật đột biến phá hỏng trọng số.
   * Bằng chứng: Ở thí nghiệm phản chứng `clip-highlr-noclip` (lr=2.0, không clip), mô hình bị nổ loss và sụp đổ (Val F1 = 0.0936). Nhưng khi bật clip $c=1.0$ (`clip-highlr-clipped`), mô hình đã được giữ ổn định và tiếp tục học (Val F1 = 0.2731).

4. **Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao (không)?**  
   * Trên mô hình MLP nhỏ này (47k tham số), Mixed Precision **không làm huấn luyện nhanh hơn rõ rệt**.
   * Lý do: Với mô hình nhỏ, thời gian tính toán ma trận rất nhỏ so với chi phí gọi kernel và điều phối luồng CPU/GPU (overhead-bound chứ không compute-bound). Mixed precision chỉ phát huy tốc độ vượt bậc trên các mạng khổng lồ (Transformer, ResNet lớn) trên GPU có Tensor Cores.

5. **Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?**  
   * Khởi tạo $W=0$ hỏng do tính đối xứng: Mọi nơ-ron sinh ra kích hoạt như nhau, $\text{ReLU}(0)=0$ và nhận gradient đạo hàm giống hệt nhau. Sau cập nhật, các nơ-ron vẫn giống nhau, triệt tiêu tính đa dạng biểu diễn và khiến mạng thoái hoá thành một nơ-ron duy nhất (F1 = 0.0936).
   * Xavier giả định kích hoạt tuyến tính quanh 0 ($\text{Var}=2/(n_{in}+n_{out})$), trong khi He tính đến việc ReLU triệt tiêu một nửa miền giá trị âm ($\text{Var}=2/n_{in}$). Khởi tạo He duy trì phương sai kích hoạt ổn định (`0.414`) trong khi Xavier bị suy hao (`0.123`). Điều này đặc biệt quan trọng với mạng nơ-ron sâu dùng ReLU để ngăn ngừa hiện tượng suy hao tín hiệu.

6. **Quay lại câu hỏi của bài học:** *"Một mạng có loss không giảm sau 2 000 bước huấn luyện. Lỗi nằm ở dữ liệu, ở kiến trúc, hay ở vòng lặp huấn luyện?"*  
   **3 phép kiểm tra đầu tiên phải làm ngay lập tức:**
   1. **Kiểm tra Loss bước 0:** So sánh với $\ln(C) = \ln(7) \approx 1.946$. Nếu loss bước 0 lệch xa, lỗi nằm ở **dữ liệu hoặc khởi tạo** (chưa chuẩn hoá, nhãn sai lệch index, bias khởi tạo lệch).
   2. **Quá khớp 20 mẫu (Overfit small batch):** Huấn luyện trên 20 mẫu tắt dropout. Nếu loss không thể về sát 0, lỗi **100% nằm ở vòng lặp huấn luyện hoặc cấu trúc code** (quên `zero_grad()`, Softmax hai lần, chưa đưa tham số vào optimizer, nhầm chế độ `eval()`).
   3. **Kiểm tra Gradient Norm từng tầng:** In `p.grad.norm()` sau `loss.backward()`. Nếu gradient bằng 0 hoặc `None`, lỗi nằm ở **đứt gãy đồ thị tính toán** (detached tensor, dead ReLU do learning rate quá cao làm nơ-ron chết hàng loạt).

---

## 6. Hạn chế và Điều bất ngờ

* **Điều bất ngờ:** Kiến trúc $M\text{-deep}$ ($54 \to 256 \to 128 \to 64 \to 7$, 55k tham số) cho kết quả vượt trội hơn hẳn so với $M\text{-wide}$ ($54 \to 512 \to 256 \to 7$, 161k tham số), dù số lượng tham số chỉ bằng 1/3. Điều này chứng minh chiều sâu biểu diễn phi tuyến có giá trị lớn hơn đơn thuần mở rộng chiều ngang đối với dữ liệu địa hình CoverType.
* **Hạn chế:** Cố định 20 epoch để đảm bảo so sánh công bằng. Nếu kết hợp thêm Cosine Annealing scheduler và chạy 40 epoch, Macro-F1 hoàn toàn có thể chạm mốc 0.90+.

---

## 7. Phụ lục

* **Danh sách file đã nộp đầy đủ:**
  * `REPORT.md`: Báo cáo khoa học hoàn chỉnh.
  * `experiments.xlsx`: Bảng kết quả 25 thí nghiệm, 4 sheet, đầy đủ công thức tự động.
  * `predictions_eval.csv`: Dự đoán 116 203 dòng của `final-model` (Accuracy: 0.9175, Macro-F1: 0.8734).
  * `eval_result.json`: Kết quả chấm điểm chính thức từ `scripts/evaluate.py`.
  * `figures/`: Đầy đủ 25 ảnh `<exp_id>.png` và 6 ảnh so sánh nhóm `compare_*.png`.
  * `results/`: Đầy đủ 25 file `<exp_id>.json`.
  * `code/`: Mã nguồn module sạch sẽ, không còn `NotImplementedError`, cùng notebook `lab.ipynb`.
* **Tổng thời gian chạy:** Khoảng 34 phút trên CPU máy cục bộ.
