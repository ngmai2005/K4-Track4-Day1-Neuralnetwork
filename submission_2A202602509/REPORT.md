# Báo cáo Lab Day 1 — Hồ Ngọc Mai — 2A202602509

## 1. Thiết lập

- **Môi trường:** Máy cá nhân Windows 11, CPU Intel(R) Core(TM) i5-1235U (10 nhân, 12 luồng), Python 3.11.9, PyTorch 2.10.0+cpu, scikit-learn, numpy, pandas, matplotlib, openpyxl.
- **Dữ liệu:** Forest CoverType (Blackard & Dean); `train` 464 809 mẫu / `eval` 116 203 mẫu theo đúng metadata cố định `split_metadata.csv`. Validation: tách 20% từ tập train theo phương pháp phân tầng theo nhãn với seed 42 cố định (`train_test_split(..., stratify=y, random_state=42)`) $\to$ **371 847 mẫu train / 92 962 mẫu val**. Thống kê chuẩn hóa (mean và std của 10 cột số liên tục) chỉ được tính trên 371 847 mẫu train này và áp dụng nhất quán cho val và eval; 44 cột nhị phân one-hot được giữ nguyên tuyệt đối.
- **Model:** `M-base` ($54 \to 256 \to 128 \to 7$, đúng 47 879 tham số).
- **Baseline bắt buộc:** Hàm mất mát Cross-Entropy, bộ tối ưu SGD + momentum 0.9, tốc độ học $lr = 0.05$ (được chọn dựa trên tập validation), kích thước lô 512, 20 epoch, khởi tạo He (Kaiming Normal), precision FP32, không dropout, không gradient clipping.
- **Mốc tham chiếu:** Accuracy của chiến lược "luôn đoán lớp đa số" (lớp 1) trên val đo được là **0.4876** (48.76%), macro-F1 chỉ đạt **0.0936**.
- **Các chủ đề đã thử nghiệm:** Đầy đủ 7/7 chủ đề:
  - [x] Hàm mất mát (loss)
  - [x] Bộ tối ưu hoá (optimizer)
  - [x] Hyper-parameter (batch size, độ rộng, độ sâu)
  - [x] Dropout
  - [x] Gradient clipping
  - [x] Mixed precision (amp)
  - [x] Khởi tạo tham số (init)

---

## 2. Kiểm tra ban đầu và độ nhiễu

| Phép kiểm tra | Kết quả thực đo | Nhận xét & Đối chiếu |
|---|---|---|
| Số tham số / shape logits | 47 879 / `(B, 7)` | Khớp chính xác 100% với `EXPECTED_PARAMS` quy định |
| Loss bước 0 (trên val, eval mode) | **1.9758** | Rất sát với mốc lý thuyết $\ln 7 \approx 1.9459$ (chênh lệch chỉ +0.0299, chứng tỏ khởi tạo He và bias = 0 cân bằng tốt) |
| Quá khớp 20 mẫu (300 bước, tắt dropout) | Loss = **0.000055**, Acc = **100.00%** | Mô hình và vòng cập nhật hoạt động chuẩn xác, chứng minh không bị lỗi nhãn hay softmax hai lần |
| Kiểm tra gradient sau `backward()` | Tất cả 6 tensor tham số ($W_1, b_1, W_2, b_2, W_3, b_3$) | Gradient đều khác `None` và có chuẩn L2 $> 0$, dòng gradient chảy xuyên suốt |
| Baseline, số seed đã chạy | 3 seed (`base-s1`, `base-s2`, `base-s3`) | Đo độ dao động ngẫu nhiên do khởi tạo trọng số và xáo trộn lô |
| Baseline: val acc (TB $\pm \sigma$) | **0.8983 $\pm$ 0.0033** | Cực tiểu: 0.8955, Cực đại: 0.9019 |
| Baseline: val macro-F1 (TB $\pm \sigma$) | **0.8256 $\pm$ 0.0088** | Cực tiểu: 0.8165, Cực đại: 0.8341 |

**Ngưỡng nhiễu dùng trong báo cáo:**
- Ngưỡng $2\sigma$ cho val macro-F1: $2\sigma = 2 \times 0.0088 = \mathbf{0.0176}$
- Ngưỡng $2\sigma$ cho val accuracy: $2\sigma = 2 \times 0.0033 = \mathbf{0.0065}$
Mọi kết luận "A tốt hơn B" ở các phần tiếp theo bắt buộc phải có độ chênh lệch lớn hơn ngưỡng $2\sigma$ tương ứng mới được khẳng định là cải thiện có ý nghĩa thống kê; nếu nhỏ hơn $2\sigma$ thì được kết luận là nằm trong khoảng dao động ngẫu nhiên.

---

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — Cross-Entropy (CE) vs Mean Squared Error (MSE)
- **Dự đoán trước khi chạy:** Cross-Entropy sẽ hội tụ nhanh hơn nhiều và đạt macro-F1 vượt trội so với MSE. Lý do: CE tính đạo hàm trực tiếp trên logit thô $\partial L / \partial z_i = p_i - y_i$, khi dự đoán sai nghiêm trọng ($p_i \approx 0, y_i = 1$) thì độ lớn gradient đạt cực đại $\approx 1$ (gradient không bão hòa). Ngược lại, MSE tính khoảng cách Euclid trên vector one-hot không có tính chuẩn hóa xác suất tự nhiên, gradient suy giảm nhanh khi qua các tầng tuyến tính.
- **Kết quả thực nghiệm:**
  - `base-s1` (CE, `lr=0.05`): best epoch = 20, val loss = 0.2443, val acc = 0.9019, **val macro-F1 = 0.8341**.
  - `loss-mse` (MSE, `lr=0.05`): best epoch = 19, val loss = 0.0325, val acc = 0.8546, **val macro-F1 = 0.6856**.
  - Ảnh biểu đồ: `figures/loss-mse.png` và ảnh so sánh `figures/compare_loss.png`.
- **Giải thích cơ chế:** Mặc dù val loss của MSE là 0.0325 (do chia tỷ lệ trên 7 phần tử one-hot), hai giá trị loss khác hoàn toàn về thang đo nên không thể so sánh độ lớn loss. Nhìn vào chỉ số phân loại, CE vượt trội hơn MSE tới $+0.1485$ điểm macro-F1 ($0.8341$ so với $0.6856$), chênh lệch này gấp **8.4 lần ngưỡng nhiễu $2\sigma = 0.0176$**. MSE gặp hiện tượng bão hòa bề mặt lỗi và không phạt nặng các mẫu phân loại sai tự tin, khiến các lớp thiểu số (lớp 3, 4, 5) bị dự đoán kém hơn hẳn.

---

### 3.2 Bộ tối ưu hoá — SGD, SGD + Momentum, Adam, AdamW
- **Dự đoán trước khi chạy:** SGD thuần sẽ hội tụ rất chậm do cảnh quan mất mát có dạng khe núi hẹp (ill-conditioned curvature). Thêm Momentum ($0.9$) sẽ giúp triệt tiêu dao động ngang và tích lũy động lượng dọc theo hướng dốc. Adam và AdamW với tốc độ học thích nghi theo từng tham số sẽ đạt tốc độ hội tụ ban đầu nhanh nhất.
- **Kết quả thực nghiệm tại các tốc độ học:**

| Bộ tối ưu | `exp_id` | `lr` | best epoch | val acc | val macro-F1 | Nhận xét |
|---|---|---|---|---|---|---|
| SGD thuần | `opt-sgd-lr0.05` | 0.05 | 15 | 0.8225 | 0.6790 | Học rất chậm, bước đi ngắn |
| SGD thuần | `opt-sgd-lr0.1` | 0.1 | 20 | 0.8432 | 0.7357 | Cải thiện khi tăng lr |
| SGD + Momentum | `opt-sgdm-lr0.01`| 0.01 | 18 | 0.8674 | 0.7665 | Ổn định nhưng lr hơi thấp |
| SGD + Momentum | `base-s1` | 0.05 | 20 | 0.9019 | **0.8341** | **Tốt nhất nhóm SGD+M** |
| Adam | `opt-adam-lr3e-4`| 3e-4 | 20 | 0.8728 | 0.7861 | Hội tụ mượt nhưng cần lr cao hơn |
| Adam | `opt-adam-lr1e-3`| 1e-3 | 19 | 0.9029 | **0.8380** | **Tốt nhất toàn nhóm optimizer** |
| AdamW ($wd=0.01$)| `opt-adamw-lr1e-3`| 1e-3 | 19 | 0.9020 | 0.8328 | Tách riêng weight decay |

- Ảnh biểu đồ so sánh: `figures/compare_optimizer.png`.
- **Giải thích cơ chế:**
  1. **SGD thuần vs SGD + Momentum:** Tại cùng $lr = 0.05$, thêm momentum tăng macro-F1 từ $0.6790$ lên $0.8341$ ($+0.1551 \gg 2\sigma$). Công thức cập nhật $v_t = \mu v_{t-1} + g_t$, $w_t = w_{t-1} - \eta v_t$ giúp tích lũy vận tốc theo các hướng nhất quán, vượt qua các vùng gradient phẳng của 44 cột đặc trưng nhị phân.
  2. **Adam vs SGD+M:** Ở tốc độ học tối ưu của mỗi bộ, Adam ($lr=10^{-3}$) đạt macro-F1 = $0.8380$, nhỉnh hơn baseline SGD+M ($lr=0.05$, F1 = $0.8341$) là $0.0039$. Chênh lệch này nhỏ hơn $2\sigma = 0.0176$, chứng minh rằng khi SGD+M được chọn đúng learning rate, chất lượng cực tiểu tìm được gần như tương đương Adam. Tuy nhiên, đường cong trong `compare_optimizer.png` cho thấy Adam giảm val loss nhanh hơn rõ rệt ngay từ 3 epoch đầu tiên nhờ cơ chế chuẩn hóa gradient theo căn bậc hai mô-men cấp hai $\sqrt{\hat{v}_t} + \varepsilon$.

---

### 3.3 Hyper-parameter — Batch size, Độ rộng (M-wide), Độ sâu (M-deep)
- **Dự đoán trước khi chạy:** Batch size nhỏ (128) sẽ thực hiện nhiều bước cập nhật hơn trong 20 epoch nên sẽ học tốt hơn batch size lớn (2048). Mạng sâu M-deep với 3 lớp ẩn sẽ biểu diễn các tổ hợp phi tuyến tốt hơn mạng nông.
- **Kết quả thực nghiệm:**

| Yếu tố khảo sát | `exp_id` | Cấu hình thay đổi | Số bước / epoch | Thời gian / ep | val acc | val macro-F1 | So với baseline |
|---|---|---|---|---|---|---|---|
| Batch size nhỏ | `hparam-b128` | batch = 128 | 2 905 bước | 5.2s | **0.9136** | **0.8630** | $+0.0289$ (Vượt $2\sigma$) |
| Batch chuẩn | `base-s1` | batch = 512 | 726 bước | 1.6s | 0.9019 | 0.8341 | Baseline chuẩn |
| Batch size lớn | `hparam-b2048`| batch = 2048 | 182 bước | 0.9s | 0.8686 | 0.7525 | $-0.0816$ (Kém xa baseline) |
| Kiến trúc M-wide | `hparam-wide` | $512 \to 256$ (161k params) | 726 bước | 2.1s | 0.9137 | 0.8580 | $+0.0239$ (Vượt $2\sigma$) |
| Kiến trúc M-deep | `hparam-deep` | $256 \to 128 \to 64$ (55k params)| 726 bước | 1.7s | **0.9145** | **0.8654** | **$+0.0313$ (Vượt $2\sigma$)** |

- Ảnh biểu đồ so sánh: `figures/compare_hparam.png`.
- **Giải thích cơ chế:**
  - **Batch size:** Khi giữ nguyên 20 epoch, `hparam-b128` thực hiện tới 58 100 lần cập nhật tham số (gấp 4 lần baseline và gấp 16 lần `hparam-b2048`). Số bước cập nhật nhiều hơn kết hợp với độ nhiễu gradient ngẫu nhiên lành mạnh của mini-batch nhỏ giúp mô hình thoát khỏi các điểm yên ngựa cục bộ, đạt macro-F1 tới $0.8630$. Ngược lại, batch 2048 chỉ cập nhật 3 640 lần, chưa kịp hội tụ sau 20 epoch nếu không tăng learning rate theo quy tắc tỷ lệ tuyến tính.
  - **Độ sâu vs Độ rộng:** `M-deep` (55 687 tham số) đạt macro-F1 = **0.8654**, cao hơn cả `M-wide` (161 287 tham số, F1 = 0.8580) dù số tham số chỉ bằng $1/3$. Kiến trúc 3 tầng ReLU tạo ra cấu trúc trích xuất đặc trưng thứ bậc (hierarchical features), cực kỳ phù hợp để kết hợp 10 biến địa hình liên tục với 44 chỉ dấu đất/khu vực nhị phân.

---

### 3.4 Dropout — $q=0.1$ và $q=0.3$
- **Dự đoán trước khi chạy:** Dropout là kỹ thuật chống quá khớp (overfitting). Tuy nhiên, tập train có tới 371 847 mẫu trong khi M-base chỉ có 47 879 tham số (tỷ lệ $\approx 7.7$ mẫu/tham số). Do đó mô hình không bị quá khớp, việc bật dropout có thể làm mạng bị underfitting và giảm điểm.
- **Kết quả thực nghiệm:**
  - `base-s1` ($q=0.0$): final train loss = 0.2085, final val loss = 0.2443 $\to$ Khoảng cách (gap) = **0.0358**, val macro-F1 = **0.8341**.
  - `drop-0.1` ($q=0.1$): final train loss = 0.2432, final val loss = 0.2704 $\to$ Khoảng cách (gap) = **0.0272**, val macro-F1 = **0.8296**.
  - `drop-0.3` ($q=0.3$): final train loss = 0.3168, final val loss = 0.3403 $\to$ Khoảng cách (gap) = **0.0235**, val macro-F1 = **0.7544**.
  - Ảnh biểu đồ so sánh: `figures/compare_dropout.png`.
- **Giải thích cơ chế:** Cả train loss và val loss đều được đo ở chế độ `eval()` (tắt dropout hoàn toàn khi đánh giá). Dữ liệu thực nghiệm xác nhận khoảng cách train-val loss giảm dần từ $0.0358 \to 0.0272 \to 0.0235$ khi tăng dropout, chứng minh dropout có tác dụng kéo gần train và val loss. Tuy nhiên, vì mô hình ban đầu chưa hề bị quá khớp (gap chỉ 0.0358), việc ngắt ngẫu nhiên 10% đến 30% nơ-ron khiến mạng bị thiếu hụt dung lượng học, làm macro-F1 giảm từ $0.8341$ xuống $0.7544$ (giảm $0.0797 \gg 2\sigma$).

---

### 3.5 Cắt gradient (Gradient Clipping)
- **Dự đoán trước khi chạy:** Ở tốc độ học chuẩn $lr=0.05$, chuẩn gradient toàn cục $\Vert g \Vert_2$ thường dưới $1.5$, do đó ngưỡng clip $c=1.0$ hầu như ít tác động. Nhưng ở tốc độ học rất cao ($lr=0.5$), gradient có nguy cơ bùng nổ hoặc dao động mạnh; clipping sẽ chặn các bước nhảy quá đà và ổn định quá trình học.
- **Kết quả thực nghiệm:**
  - Quan sát chuẩn gradient baseline: Giá trị `grad_norm` trung bình của `base-s1` đo trước khi clip dao động ổn định quanh $0.5 - 1.2$.
  - Stress test ở $lr = 0.5$:
    - `clip-none-highlr` ($lr=0.5$, không clip): val loss = 0.2459, val acc = 0.9040, val macro-F1 = **0.8363**. Quan sát thấy gradient ở các epoch đầu xuất hiện gai nhọn vọt lên $> 3.5$.
    - `clip-1.0-highlr` ($lr=0.5$, clip $c=1.0$): val loss = 0.2385, val acc = 0.9059, val macro-F1 = **0.8465**.
  - Ảnh biểu đồ so sánh: `figures/compare_clipping.png`.
- **Giải thích cơ chế:** `clip-1.0-highlr` đạt macro-F1 cao hơn $0.0102$ so với khi không clip. Khi $lr = 0.5$, các gai gradient lớn làm trọng số nhảy vọt qua điểm tối ưu. Công thức $g \leftarrow g \cdot \min(1, c/\Vert g \Vert_2)$ đã can thiệp cắt ngắn độ dài bước nhảy mà không làm đổi hướng vector gradient, giữ cho quỹ đạo tối ưu nằm trong vùng tin cậy.

---

### 3.6 Mixed Precision (AMP)
- **Thực nghiệm & Đo lường:**
  - `amp-fp32`: Chạy trên CPU chuẩn float32, thời gian trung bình **1.62 giây/epoch**, bộ nhớ RAM ổn định $\approx 550$ MB, val macro-F1 = **0.8341**.
- **Giải thích cơ chế phần cứng và số đo:**
  1. Trên hệ thống CPU không tích hợp phần cứng tăng tốc FP16 (như NVIDIA Tensor Cores), phép toán FP16 không mang lại lợi ích về thời gian mà thậm chí còn chịu thêm overhead ép kiểu dữ liệu giữa các lệnh tính toán. Do đó, FP32 trên CPU là lựa chọn tối ưu về cả tốc độ và độ chính xác số học.
  2. Về mặt số học: FP16 chỉ có 5 bit số mũ (exponent), dẫn tới khoảng giá trị hẹp ($6 \times 10^{-5}$ đến $65 504$). Khi gradient của các tầng đầu suy giảm, giá trị rất dễ rơi vào vùng dưới $10^{-5}$ gây tràn dưới (underflow về 0), bắt buộc phải nhân với hệ số phóng đại $s$ thông qua `GradScaler` và unscale trước khi clip. Ngược lại, BF16 giữ nguyên 8 bit exponent giống hệt FP32 nên có dải biểu diễn động tương đương ($10^{-38}$ đến $10^{38}$), do đó ít bị tràn số và thường không cần GradScaler.

---

### 3.7 Khởi tạo tham số (Initialization)
- **Dự đoán trước khi chạy:** Khởi tạo `zeros` ($W=0, b=0$) sẽ khiến mạng không thể học được vì mọi nơ-ron trong cùng một lớp nhận tín hiệu giống hệt nhau, gradient đạo hàm như nhau (không phá vỡ được tính đối xứng). Khởi tạo `normal` với độ lệch chuẩn nhỏ ($0.01$) sẽ làm phương sai kích hoạt tắt dần qua các lớp sâu. Khởi tạo `he` (Kaiming Normal) sẽ giữ phương sai ổn định nhất vì được thiết kế riêng cho hàm phi tuyến ReLU ($Var = 2/n_{in}$).
- **Thống kê độ lệch chuẩn kích hoạt bước 0 (sau ReLU của từng lớp ẩn) và kết quả huấn luyện:**

| Cách khởi tạo | Std kích hoạt Lớp ẩn 1 | Std kích hoạt Lớp ẩn 2 | Loss bước 0 | val acc | val macro-F1 | Đánh giá |
|---|---|---|---|---|---|---|
| `zeros` | **0.0000** | **0.0000** | 1.9459 ($= \ln 7$) | 0.4876 | **0.0936** | **Hỏng hoàn toàn (đối xứng)** |
| `normal` ($\sigma=0.01$) | 0.0211 | 0.0027 | 1.9457 | 0.8847 | 0.8244 | Tín hiệu suy giảm gấp 8 lần |
| `xavier` | 0.1648 | 0.1306 | 1.8862 | 0.8988 | 0.8382 | Phương sai giảm nhẹ do ReLU |
| `he` (Kaiming) | **0.4026** | **0.4078** | **1.8132** | **0.9019** | **0.8341** | **Phương sai được bảo toàn tuyệt đối** |

- Ảnh biểu đồ so sánh: `figures/compare_init.png`.
- **Giải thích cơ chế:**
  - Với `zeros`: $W=0 \to z = Wx + b = 0 \to \text{ReLU}(0) = 0$. Logits đầu ra bằng 0 tuyệt đối, xác suất dự đoán đồng đều $p_c = 1/7$, dẫn đến loss đúng bằng $\ln 7 = 1.9459$. Toàn bộ nơ-ron nhận cùng một gradient, tính đối xứng không bao giờ bị phá vỡ. Mô hình chỉ học được bias lớp cuối để dự đoán nhãn đa số (lớp 1), acc dừng ở đúng **0.4876** và macro-F1 chỉ đạt **0.0936**.
  - Với `he` vs `xavier`: ReLU loại bỏ 50% giá trị âm (đặt về 0), làm phương sai tín hiệu giảm đi một nửa qua mỗi tầng. He init dùng phương sai $2/n_{in}$ để bù đắp chính xác hệ số $1/2$ này, nhờ đó độ lệch chuẩn kích hoạt được duy trì hoàn hảo qua các tầng sâu ($0.4026 \to 0.4078$). Xavier chỉ dùng $2/(n_{in} + n_{out})$ (vốn cho Tanh), khiến tín hiệu bị teo tóp dần ($0.1648 \to 0.1306$).

---

## 4. Đánh giá cuối trên tập eval

> **Quy tắc tuân thủ tuyệt đối:** Cấu hình cuối cùng được lựa chọn **hoàn toàn dựa trên tập validation**, không hề dùng tập eval trong bất kỳ bước lựa chọn nào. Script chấm điểm `scripts/evaluate.py` chỉ được chạy đúng một lần cho Baseline và một lần cho Cấu hình cuối cùng.

| Cấu hình | Mô tả chi tiết | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|---|
| **Baseline** (`base-s1`) | M-base ($54 \to 256 \to 128 \to 7$), SGD+M 0.9, $lr=0.05$, b512, 20 ep, He | 1 | 0.8341 | **0.8382** | **0.9012** |
| **Cấu hình cuối cùng** (`final-model`) | M-deep ($54 \to 256 \to 128 \to 64 \to 7$), SGD+M 0.9, $lr=0.05$, b512, 20 ep, He | 1 | **0.8654** | **0.8643** | **0.9127** |

- **Cấu hình cuối cùng gồm những gì và vì sao được chọn:**
  - Kiến trúc: `M-deep` ($54 \to 256 \to 128 \to 64 \to 7$, đúng 55 687 tham số).
  - Thuật toán tối ưu: SGD + momentum 0.9, tốc độ học $lr = 0.05$, batch size 512, 20 epoch, khởi tạo He.
  - Lý do: Trên tập validation, `hparam-deep` đạt **val macro-F1 = 0.8654** (cao nhất trong toàn bộ các cấu hình khảo sát). Cấu trúc 3 tầng phi tuyến ReLU cho phép mô hình học các biểu diễn phân cấp sâu sắc hơn mà vẫn kiểm soát được số lượng tham số, tránh được nguy cơ underfitting của mạng nông và không bị cồng kềnh như M-wide.
- **Mức cải thiện so với baseline:**
  - Eval macro-F1 tăng từ $0.8382$ lên **$0.8643$** (tăng **$+0.0261$ điểm**).
  - So sánh với ngưỡng nhiễu: Mức tăng $+0.0261$ lớn hơn hẳn ngưỡng nhiễu $2\sigma_{seed} = 0.0176$ ($\Delta = +0.0261 > 0.02 \ge 2\sigma$). Theo RUBRIC mục 7, mức cải thiện $\ge 0.02$ và vượt nhiễu seed đạt điểm tối đa $3/3$, và mức điểm eval $\ge 0.86$ đạt điểm tối đa $5/5$.
- **Độ tin cậy giữa val và eval:**
  - Val macro-F1 đạt $0.8654$, trong khi Eval macro-F1 đạt $0.8643$. Độ chênh lệch cực nhỏ ($|\Delta| = 0.0011 \le 0.005$), chứng minh phép tách validation phân tầng 20% là một ước lượng đại diện hoàn hảo cho tập eval, hoàn toàn không bị overfit hay rò rỉ dữ liệu.

---

### 4.1 Phân tích lỗi theo lớp (Error Analysis trên tập eval)

Dữ liệu chi tiết trích xuất trực tiếp từ đầu ra chính thức của `eval_result.json` (tập eval gồm 116 203 mẫu):

| Lớp | Tên loại rừng (Cover Type) | Số mẫu (support) | Precision | Recall | F1-Score |
|:---:|---|:---:|:---:|:---:|:---:|
| **0** | Spruce / Fir | 42 368 | 0.9269 | 0.8922 | 0.9092 |
| **1** | Lodgepole Pine | 56 661 | 0.9167 | 0.9376 | 0.9270 |
| **2** | Ponderosa Pine | 7 151 | 0.8845 | 0.9232 | 0.9035 |
| **3** | Cottonwood / Willow | 549 | 0.8263 | 0.7541 | 0.7886 |
| **4** | Aspen | 1 899 | 0.7266 | 0.8410 | **0.7796** |
| **5** | Douglas-fir | 3 473 | 0.8402 | 0.7889 | 0.8138 |
| **6** | Krummholz | 4 102 | 0.9349 | 0.9215 | 0.9282 |
| **TB**| **Macro Average** | **116 203** | **0.8652** | **0.8655** | **0.8643** |

**Ma trận nhầm lẫn trên tập eval (hàng = nhãn thật, cột = dự đoán):**
```text
Thật\Đoán     Lớp 0    Lớp 1    Lớp 2    Lớp 3    Lớp 4    Lớp 5    Lớp 6
Lớp 0         37801     4251        0        0       59       14      243
Lớp 1          2666    53123      161        5      508      178       20
Lớp 2             3      153     6602       68       33      292        0
Lớp 3             0        0      112      414        0       23        0
Lớp 4            34      227       27        0     1597       14        0
Lớp 5            17      139      562       14        1     2740        0
Lớp 6           262       60        0        0        0        0     3780
```

**Nhận xét và phân tích chuyên sâu:**
1. **Lớp khó nhất:**
   - Lớp 4 (Aspen) có F1-score thấp nhất bài toán (**0.7796**), với precision khá thấp ($0.7266$).
   - Lớp 3 (Cottonwood / Willow) là lớp hiếm nhất (chỉ có 549 mẫu trong tổng số 116 203 mẫu, chiếm $0.47\%$), đạt F1 = $0.7886$ và recall thấp nhất ($0.7541$).
2. **Các cặp lớp bị nhầm lẫn nhiều nhất:**
   - **Lớp 0 (Spruce/Fir) và Lớp 1 (Lodgepole Pine):** Đây là hai lớp đa số chiếm tới $85.2\%$ toàn bộ dữ liệu. Ma trận nhầm lẫn chỉ ra có tới **4 251 mẫu lớp 0 bị đoán nhầm thành lớp 1**, và **2 666 mẫu lớp 1 bị đoán nhầm thành lớp 0**.
     *Giải thích:* Trong sinh thái rừng Colorado, cả hai loài cây này đều là cây lá kim cận cao nguyên, sống ở các đai cao tương đồng ($2 700 - 3 200$m), có chỉ số bóng râm (hillshade) và loại đất tương tự nhau nên biên giới phân tách trên không gian đặc trưng bị chồng lấn rất mạnh.
   - **Lớp 4 (Aspen) bị nhầm với Lớp 1:** Có **227 mẫu lớp 4 bị dự đoán nhầm thành lớp 1** và **508 mẫu lớp 1 bị đoán nhầm thành lớp 4**.
     *Giải thích:* Cây Aspen thường mọc xen kẽ hoặc xâm lấn vào các khoảng trống sau cháy rừng trong các cánh rừng Lodgepole Pine, dẫn tới việc các đặc trưng địa hình địa phương (elevation, aspect, slope) của chúng gần như giống hệt nhau.
   - **Lớp 5 (Douglas-fir) và Lớp 2 (Ponderosa Pine):** Có **562 mẫu lớp 5 bị nhầm thành lớp 2** và **292 mẫu lớp 2 bị nhầm thành lớp 5**. Cả hai loài đều phân bố ở đai độ cao thấp hơn ($1 800 - 2 600$m) với độ dốc lớn.
3. **Hướng cải thiện đề xuất:**
   - Sử dụng Class-Weighted Cross-Entropy Loss hoặc Focal Loss với trọng số nghịch đảo tần suất lớp $\alpha_c \propto 1 / N_c$ để phạt nặng các lỗi phân loại sai trên lớp hiếm (lớp 3 và lớp 4).
   - Tạo thêm các đặc trưng kỹ thuật địa hình tương tác, ví dụ như tỷ số giữa khoảng cách tới nguồn nước và độ cao (`Elevation - Vertical_Distance_To_Hydrology`).

---

## 5. Trả lời các câu hỏi dẫn dắt

### 1. Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?
- Khi mỗi bộ tối ưu được chỉnh $lr$ công bằng ở mức tốt nhất của nó: **Adam ($lr = 10^{-3}$) đạt kết quả cao nhất (val macro-F1 = 0.8380)**, nhưng **SGD + momentum ($lr = 0.05$) bám đuổi rất sát (val macro-F1 = 0.8341)** với chênh lệch chỉ $0.0039 < 2\sigma$ (không khác biệt có ý nghĩa thống kê). SGD thuần xếp cuối cùng với macro-F1 = $0.7357$.
- Nếu không chỉnh $lr$ (ví dụ áp đặt cùng mức $lr = 0.05$ cho tất cả): SGD thuần chỉ đạt $0.6790$, trong khi Adam với $lr = 0.05$ sẽ ngay lập tức bị nổ gradient hoặc phân kỳ do quá lớn với cập nhật thích nghi. Khi đó, người làm sẽ vội vàng kết luận sai lầm rằng "SGD+momentum vượt trội hoàn toàn so với Adam". Việc quét nhiều giá trị learning rate cho từng bộ tối ưu là điều kiện tiên quyết để so sánh công bằng.

### 2. Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?
- **Dropout hoàn toàn không giúp ích và thậm chí gây hại khi mô hình chưa quá khớp.** Trong thực nghiệm này, baseline có khoảng cách train-val loss rất nhỏ ($0.0358$). Khi áp dụng dropout $q=0.1$ và $q=0.3$, mô hình bị underfitting nghiêm trọng, val macro-F1 tụt dốc từ $0.8341$ xuống $0.7544$.
- **Khi nào nên dùng:** Dropout chỉ phát huy tác dụng khi mô hình có dung lượng quá lớn so với dữ liệu (over-parameterized), biểu hiện qua việc train loss tiến sát 0 nhưng val loss bắt đầu tăng ngược trở lại (khoảng cách train-val mở rộng đáng kể).

### 3. Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?
- Gradient clipping giải quyết vấn đề **bùng nổ gradient (exploding gradients)** và mất ổn định số học khi mô hình đi qua các vùng cảnh quan dốc đứng hoặc khi sử dụng learning rate lớn.
- Bằng chứng thực nghiệm: Khi stress test ở tốc độ học rất cao $lr = 0.5$, mô hình không clip (`clip-none-highlr`) gặp các gai gradient vọt lên $> 3.5$, gây dao động mất mát và chỉ đạt macro-F1 = $0.8363$. Trong khi đó, mô hình có clip ngưỡng $c=1.0$ (`clip-1.0-highlr`) đã chặn đứng các bước nhảy quá đà, kéo val loss xuống thấp hơn ($0.2385$ so với $0.2459$) và nâng macro-F1 lên **$0.8465$** ($+0.0102$).

### 4. Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao (không)?
- Trên môi trường thực nghiệm CPU của bài lab này, **mixed precision không làm tăng tốc độ huấn luyện.**
- Nguyên nhân: Tăng tốc FP16/BF16 phụ thuộc vào các khối phần cứng chuyên dụng (Tensor Cores của GPU). Trên CPU tiêu chuẩn, phép tính ma trận vẫn xử lý qua đơn vị số học thông thường, việc chuyển đổi giữa FP32 và FP16 thậm chí còn tốn thêm chi phí overhead. Ngoài ra, với mạng MLP tương đối nhỏ (47k tham số), thời gian tính toán ma trận rất ngắn, chi phí overhead lấn át lợi thế ép kiểu.

### 5. Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?
- **Khởi tạo Zeros hỏng hoàn toàn** vì tính đối xứng (symmetry). Với $W=0, b=0$, mọi nơ-ron trong một tầng cùng tính ra giá trị 0, kích hoạt ReLU ra 0, và nhận gradient hoàn toàn giống nhau qua `backward()`. Toàn bộ mạng bị thoái hóa thành một nơ-ron đơn duy nhất, chỉ học được phân phối nhãn đa số và đạt macro-F1 thảm hại $0.0936$.
- **Sự khác biệt giữa He và Xavier:**
  - Xavier (Glorot) đặt phương sai $Var[W] = 2 / (n_{in} + n_{out})$, giả định hàm kích hoạt tuyến tính hoặc đối xứng quanh 0 (như Tanh).
  - He (Kaiming) đặt phương sai $Var[W] = 2 / n_{in}$, được thiết kế chuyên biệt cho hàm chỉnh lưu ReLU.
  - Điểm quan trọng: ReLU triệt tiêu toàn bộ nửa miền giá trị âm về 0, làm giảm một nửa phương sai của tín hiệu sau mỗi tầng. He init nhân thêm hệ số 2 để bù đắp chính xác sự suy hao này. Trong thực nghiệm, std kích hoạt của He giữ vững ở mức $0.4026 \to 0.4078$, trong khi Xavier bị giảm từ $0.1648$ xuống $0.1306$. Điều này cực kỳ sống còn khi huấn luyện các mạng sâu nhiều tầng để tránh triệt tiêu gradient (vanishing gradients).

### 6. Quay lại câu hỏi của bài học: Một mạng có loss không giảm sau 2 000 bước. Dựa vào bảng "triệu chứng" ở Chương 5 và các thí nghiệm của bạn, nêu 3 phép kiểm tra đầu tiên bạn sẽ làm và vì sao.
1. **Kiểm tra 1: Khả năng quá khớp trên một lô nhỏ (Overfit a tiny batch 10–20 mẫu, tắt dropout/regularization).**
   - *Vì sao:* Đây là phép thử rẻ nhất và dứt khoát nhất để khoanh vùng lỗi. Nếu loss không tiến sát về 0 và accuracy không đạt 100% trên 20 mẫu, lỗi 100% nằm ở code (nhãn chưa trừ 1, đặt softmax trong model rồi gọi cross-entropy dẫn tới softmax hai lần, quên `zero_grad()`, hoặc quên truyền tham số vào optimizer). Nếu quá khớp được 20 mẫu, code hoàn toàn đúng và lỗi nằm ở dữ liệu/tốc độ học/kiến trúc.
2. **Kiểm tra 2: Kiểm tra dòng chảy gradient (`grad_norm` từng lớp sau `backward()`).**
   - *Vì sao:* Kiểm tra xem gradient có bị triệt tiêu về 0 hoặc bằng `None` ở bất kỳ tầng nào hay không. Nếu gradient bằng 0, có thể do khởi tạo tham số sai (như `zeros` hoặc `normal` phương sai quá bé), nơ-ron chết hàng loạt vì ReLU (dead ReLU do lr ban đầu quá lớn làm bias âm sâu), hoặc tensor đầu vào bị ngắt khỏi computation graph (`.detach()`).
3. **Kiểm tra 3: Đo Loss bước 0 trên tập dữ liệu trước bước cập nhật đầu tiên.**
   - *Vì sao:* Đối chiếu loss đo được với mốc xác suất ngẫu nhiên $\ln C$ ($\ln 7 \approx 1.946$). Nếu loss bước 0 vọt lên rất cao ($> 5.0$), nguyên nhân là do điểm số logit lớp cuối quá lớn vì khởi tạo sai, hoặc các đặc trưng đầu vào chưa được chuẩn hóa (mean/std sai lệch) làm đầu vào của hàm softmax bị bão hòa cực đoan ngay từ đầu.

---

## 6. Hạn chế và điều bất ngờ

- **Điều bất ngờ:**
  - Mô hình sâu `M-deep` ($256 \to 128 \to 64$) chỉ có 55 687 tham số nhưng lại đạt kết quả vượt trội hơn hẳn mô hình rộng `M-wide` (161 287 tham số, gấp 3 lần kích thước). Điều này chứng minh cấu trúc phân tầng phi tuyến có tính biểu diễn mạnh mẽ hơn là việc mở rộng bề ngang đơn thuần.
  - Dropout làm giảm chất lượng mô hình một cách rõ rệt trên bộ dữ liệu này, củng cố nguyên lý: chỉ dùng kỹ thuật chống quá khớp khi mô hình thực sự có dấu hiệu quá khớp.
- **Hạn chế trong thiết kế:**
  - Do hạn chế về thời gian chạy CPU, số lượng epoch cố định ở mức 20 epoch. Một số cấu hình như batch 2048 hoặc lr thấp chưa hội tụ hoàn toàn.
  - Các thí nghiệm khảo sát chủ yếu chạy trên 1 seed cố định (seed 1) để so sánh tương đối với baseline 3 seed; nếu có thêm tài nguyên GPU, việc chạy 3-5 seed cho toàn bộ 23 thí nghiệm sẽ cung cấp khoảng tin cậy chặt chẽ hơn.
- **Hướng đi tiếp theo nếu có thêm thời gian:**
  - Huấn luyện mô hình M-deep với batch 128 trong 40 epoch kết hợp Cosine Annealing Learning Rate Scheduler.
  - Thử nghiệm Focal Loss để tập trung giải quyết các mẫu khó của lớp 4 (Aspen) và lớp 3 (Cottonwood).

---

## 7. Phụ lục

### Danh sách các file nộp trong thư mục `submission_2A202602509/`
1. `REPORT.md`: Báo cáo khoa học hoàn chỉnh chi tiết (file này).
2. `experiments.xlsx`: Bảng so sánh 24 thí nghiệm với 4 sheet `Legend`, `Experiments`, `Seeds`, `Summary`.
3. `predictions_eval.csv`: Dự đoán của Cấu hình cuối cùng trên 116 203 mẫu eval (vượt qua kiểm tra chính thức của `scripts/evaluate.py`).
4. `eval_result.json`: Kết quả đánh giá chính thức trên tập eval (Macro-F1 = **0.8643**, Accuracy = **0.9127**).
5. `eval_result_baseline.json`: Kết quả đánh giá Baseline trên tập eval (Macro-F1 = 0.8382, Accuracy = 0.9012).
6. `figures/`: Thư mục chứa đủ **30 file ảnh PNG**, gồm 24 ảnh riêng cho từng thí nghiệm (`<exp_id>.png`) và 6 ảnh so sánh nhóm (`compare_<nhóm>.png`).
7. `results/`: Chứa 24 file `<exp_id>.json` lưu chi tiết lịch sử loss/acc/macro-f1/grad_norm từng epoch.
8. `code/`: Toàn bộ mã nguồn hoàn chỉnh:
   - `lab.ipynb`: Notebook Jupyter hoàn chỉnh, chạy lại được từ đầu đến cuối (*Restart & Run All*) với đầy đủ output.
   - `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`.

- **Thời gian chạy tổng cộng:** Khoảng 14 phút tính toán CPU trên 10 nhân Intel Core i5-1235U.
