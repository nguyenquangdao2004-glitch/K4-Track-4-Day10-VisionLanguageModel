Họ tên: Nguyễn Quang Đạo
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | - | - | 0.0602 | [0.0410, 0.0823] | 1.23 | 0.0988 | 0.0142 | 0.1328 |
| E1 | LoRA LLM | 33,030,144 | 6.30 | 40.64 | 0.2620 | [0.2261, 0.2983] | 1.32 | 0.2381 | 0.1837 | 0.4980 |
| E2 | + projector | 142,135,296 | 9.56 | 41.15 | 0.3123 | [0.2724, 0.3534] | 1.10 | 0.2130 | 0.2244 | 0.5861 |
| E3 | + LoRA ViT | 148,426,752 | 8.38 | 50.59 | 0.4129 | [0.3681, 0.4573] | 0.97 | 0.2762 | 0.3214 | 0.7103 |
| E4 | coord_scale: pixel | 142,135,296 | 10.31 | 41.23 | 0.3076 | [0.2696, 0.3478] | 1.15 | 0.1921 | 0.2339 | 0.5461 |

*(Ghi chú E4: Giá trị so sánh đối ứng `coord_scale: norm1000` chính là baseline E2 với F1 = 0.3123, CI95 = [0.2724, 0.3534]).*

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Quy mô từng khối của Qwen3-VL-4B** (tổng ~4.44B tham số):
  - **LLM backbone (CausalLM)**: ~4.0B tham số (36 transformer layers).
  - **ViT (Vision Transformer)**: ~415M tham số (24 visual blocks).
  - **Projector (Vision Merger)**: ~109M tham số (bao gồm `visual.merger` ~27M tham số và `visual.deepstack_merger_list` 3 levels ~82M tham số kết nối các tầng đặc trưng trung gian 5, 11, 17 của ViT).
- **Các khối được cập nhật qua từng setup**:
  - **E1**: Chỉ cập nhật LoRA trên LLM (252 module: `q, k, v, o, gate, up, down_proj`), tương ứng **33,030,144** tham số (chiếm ~0.74% model). Toàn bộ ViT và Projector đóng băng.
  - **E2**: Cập nhật LoRA LLM (33.03M) + mở toàn bộ Projector ở float32 (**109,105,152** tham số của `visual.merger` và `deepstack_merger_list`), tổng cộng **142,135,296** tham số (~3.20% model).
  - **E3**: Cập nhật LoRA LLM (33.03M) + toàn bộ Projector (109.11M) + bổ sung LoRA trên ViT (96 module: `attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2`), tổng cộng **148,426,752** tham số (~3.34% model).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- `parse_valid_rate` của E0 là **0.8232** (tức 17.68% output không parse được JSON hợp lệ).
- **Hai lỗi phổ biến nhất của model gốc**:
  1. **Ảo giác nhãn và sai toạ độ thang đo**: Model tự sinh các toạ độ không theo chuẩn 0–1000 hoặc đoán nhầm bóng cây/vệt loang mặt đường thành vết nứt, dẫn đến lượng False Positive rất cao (FP/ảnh = **1.23**), F1 lớp `crack` chỉ đạt **0.0142**.
  2. **Kèm văn bản thừa ngoài JSON**: Model không tuân thủ hoàn toàn chỉ dẫn "chỉ trả về JSON thuần" mà thường bọc thêm câu dẫn tiếng Anh ("Here is the detected damage bounding box:") hoặc bọc định dạng markdown không đóng kín.
- **Ví dụ output thô của E0**:
  ```text
  [{"label": "crack", "bbox_2d": [512, 198, 540, 245]}, {"label": "pothole", "bbox_2d": [120, 680, 210, 750]}]
  ```

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Kết quả khi chấm lệch thang**: F1 sẽ rơi thẳng về **0.0**.
- **Nguyên nhân**: Điểm IoU threshold là 0.5. Bounding box của ảnh dashcam theo pixel chỉ nằm trong phạm vi $[0..640] \times [0..360]$. Nếu model dự đoán theo thang $[0..1000]$, giá trị toạ độ thường vượt xa 640 (ví dụ $x_2 = 850$). Khi tính giao cắt (`inter`) giữa một box thang 1000 và ground-truth thang pixel (hoặc ngược lại), vùng giao nhau gần như bằng 0, không có cặp box nào đạt $\text{IoU} \ge 0.5$.
- **Kiểm chứng**: Đã kiểm chứng qua bộ kiểm thử `tests/test_score.py` và thực tế thí nghiệm E4: Khi chuyển hẳn cả prompt, nhãn và chấm điểm bằng `--coord pixel`, mô hình mới đạt F1 = 0.3076; nếu không chuyển đổi đồng nhất, F1 sụp đổ hoàn toàn về 0.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Vì sao chỉ tính loss trên câu trả lời**: Bài toán là Conditional Visual Grounding. Ta cần tối ưu hoá xác suất có điều kiện $P(\text{answer} \mid \text{image}, \text{prompt})$. Việc đặt `labels = -100` trên prompt và 220 visual tokens buộc mô hình chỉ dùng năng lực học của adapter để học cách trích xuất toạ độ và phân loại hư hại mặt đường.
- **Nếu tính loss trên cả prompt và token ảnh**: 
  - Token ảnh (220 tokens) và prompt (hàng chục tokens) chiếm hơn 90% độ dài chuỗi đầu vào so với chỉ ~20–30 tokens của câu trả lời JSON.
  - Gradient tính trên prompt và ảnh sẽ chiếm ưu thế tuyệt đối, khiến mô hình lãng phí phần lớn trọng số LoRA để học tái tạo lại các từ ngữ tiếng Việt thông dụng của prompt thay vì tập trung học ánh xạ vị trí trực quan. Kết quả là mô hình hội tụ kém, loss cho phần JSON cao và khả năng định vị suy giảm rõ rệt.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện nếu ViT bị quantize 4-bit**:
  - Khi bật `load_in_4bit=True` mà không loại trừ, các lớp linear của projector (`visual.merger` và `deepstack_merger_list`) sẽ bị biến đổi thành `Linear4bit` với trọng số nén dạng `Params4bit` (uint8). Các trọng số này bị đóng băng lượng tử, không hỗ trợ cập nhật gradient vi phân liên tục.
- **Cách xử lý**:
  - Cấu hình `llm_int8_skip_modules=["visual", "merger", "deepstack", "deepstack_merger_list", "lm_head"]` khi load base model để giữ nguyên float16 cho toàn bộ ViT và Projector.
  - Trong hàm `apply_setup`, chủ động kiểm tra nếu có layer dạng `Linear4bit` thì dequantize giải nén về `torch.nn.Linear` chuẩn ở `float32` và bật `requires_grad = True`.
- **Vì sao Projector phải ở float32**:
  - Projector làm nhiệm vụ ánh xạ không gian biểu diễn đa chiều từ ViT sang LLM. Do gradient truyền ngược từ LLM qua Projector có phương sai và biên độ lớn, nếu để `fp16` sẽ rất dễ bị tràn số (underflow/overflow) hoặc mất mát độ chính xác số học. Hơn nữa, PyTorch `GradScaler` yêu cầu các tham số tối ưu hoá chính thức (master weights) phải ở `float32` để tính toán bước unscale gradient chuẩn xác.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên cho F1 cao giả**: Dữ liệu thu thập từ video dashcam theo từng phiên (`session`). Các frame ảnh liên tiếp trong cùng một phiên có góc quay camera, mặt đường, thời tiết, ánh sáng và thậm chí chính các ổ gà/vết nứt giống hệt nhau. Nếu chia ngẫu nhiên từng frame, các frame gần nhau của cùng một ổ gà sẽ xuất hiện ở cả tập Train và Test (data leakage), mô hình chỉ cần "nhớ" bối cảnh nền đường là có thể đạt điểm cao giả tạo, nhưng sẽ thất bại hoàn toàn khi triển khai trên đoạn đường mới.
- **Số ảnh và số box từng lớp theo `data/splits/meta.json`**:
  - **Train** (1,422 ảnh): 1,074 pothole, 1,750 crack, 691 manhole (Tổng 3,515 boxes).
  - **Val** (276 ảnh): 66 pothole, 470 crack, 103 manhole (Tổng 639 boxes).
  - **Test** (311 ảnh): 121 pothole, 299 crack, 163 manhole (Tổng 583 boxes).
  - Không có bất kỳ session nào bị trùng giữa 3 split.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **Số liệu thực nghiệm**:
  - **E1**: F1 = **0.2620**, CI95: [0.2261, 0.2983], VRAM: **6.30 GB**, Train: **40.64 phút**.
  - **E2**: F1 = **0.3123**, CI95: [0.2724, 0.3534], VRAM: **9.56 GB**, Train: **41.15 phút**.
  - **E3**: F1 = **0.4129**, CI95: [0.3681, 0.4573], VRAM: **8.38 GB**, Train: **50.59 phút**.
- **Đánh giá**:
  - **Mở Projector (E2) rất đáng**: F1 tăng vọt từ 0.262 lên 0.312 (+5.03%) trong khi thời gian huấn luyện hầu như không đổi (41.15 phút so với 40.64 phút), VRAM 9.56 GB hoàn toàn nằm trong giới hạn an toàn của GPU T4 16GB.
  - **Thêm LoRA trên ViT (E3) cực kỳ đáng giá**: F1 tăng thêm gần 10.06% (đạt 0.4129, tổng tăng +15.09% so với E1). Khoảng tin cậy của E3 ([0.3681, 0.4573]) tách biệt hoàn toàn và cao hơn hẳn khoảng tin cậy của E1 ([0.2261, 0.2983]), chứng minh LoRA trên ViT cho phép mô hình trích xuất đặc trưng thị giác chuyên sâu cho hư hại đường bộ. Thời gian train chỉ tăng thêm ~9 phút.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- Trên 311 ảnh test, khoảng tin cậy 95% bootstrap có độ rộng khoảng $\Delta \approx 0.07 - 0.09$ (E1: độ rộng 0.072; E2: độ rộng 0.081; E3: độ rộng 0.089).
- **Giới hạn kết luận**:
  - Đối với bước nhảy lớn như giữa E3 (0.4129) và E1 (0.2620), hai khoảng CI95 tách rời hoàn toàn, kết luận E3 vượt trội là chắc chắn về mặt thống kê.
  - Tuy nhiên, giữa E1 ([0.226, 0.298]) và E2 ([0.272, 0.353]), hoặc giữa E2 và E4 (pixel vs norm1000), các khoảng CI95 có vùng chồng lấn nhẹ. Điều này cho thấy với kích thước mẫu 311 ảnh, các chênh lệch nhỏ ($\le 3\%$) vẫn có thể chịu ảnh hưởng một phần bởi biến thiên ngẫu nhiên của mẫu test, chưa thể kết luận tuyệt đối nếu không mở rộng kích thước tập test.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Lớp tốt nhất**: `manhole` (nắp cống) đạt F1 cao nhất ở tất cả các setup (E1: 0.498, E2: 0.586, E3: **0.710**). Do nắp cống có hình dạng hình học tròn/chữ nhật rõ ràng, ranh giới cứng và cấu trúc tương phản mạnh với bề mặt đường nhựa.
- **Lớp kém nhất**: `crack` (vết nứt) ở E1/E2 và `pothole` (ổ gà) ở E3 (F1 chỉ đạt **0.2762**). 
  - Vết nứt có tỷ lệ dự đoán nhầm (False Positive) rất cao (E1 có 324 FP, E2 có 239 FP) do các vệt dầu, vạch kẻ đường mờ hoặc bóng cây đổ xuống đường thường bị mô hình nhận nhầm là vết nứt.
  - Ổ gà thường có mép nông, độ tương phản thấp và hình dạng bất định, khiến diện tích giao nhau IoU thường bị dưới ngưỡng 0.5.
- **Ảnh minh hoạ**: Đã xuất ảnh so sánh Ground Truth (xanh lá) và Dự đoán (đỏ) tại thư mục `report/images/summary_samples.png` và các ảnh chi tiết `pred_sample_1.jpg` đến `pred_sample_6.jpg`.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến ablation**: `coord_scale` so sánh giữa `norm1000` (thang toạ độ chuẩn hoá [0..1000]) và `pixel` (thang toạ độ pixel thực tế [0..640, 0..360]).
- **Kết quả thu được**:
  - `norm1000` (baseline E2): F1 = **0.3123**, CI95: [0.2724, 0.3534], parse_valid_rate = **1.000**, VRAM = 9.56 GB.
  - `pixel`: F1 = **0.3076**, CI95: [0.2696, 0.3478], parse_valid_rate = **0.9936**, VRAM = 10.31 GB.
- **Độ tin cậy của kết luận**: Thang `norm1000` cho kết quả nhỉnh hơn (+0.47% F1) và tỷ lệ cú pháp JSON hợp lệ tuyệt đối (100% so với 99.36%). Điều này khớp với thiết kế nguyên bản của dòng mô hình Qwen-VL (vốn được pre-train grounding trên thang 0-1000). Tuy nhiên, vì hai khoảng CI95 gần như trùng khít nhau, mức chênh lệch 0.47% nằm trong biên độ sai số ngẫu nhiên của tập test 311 ảnh.

## Nhìn lại

- **Khó khăn và bài học lớn nhất**: Việc kiểm soát kiểu dữ liệu giữa các thành phần mô hình hỗn hợp (LoRA fp32, base model 4-bit NF4, ViT fp16) và cơ chế AMP `GradScaler`. Cần cẩn trọng khi unfreeze các module đã bị bitsandbytes quantize 4-bit để tránh lỗi `size mismatch` và lỗi mismatch dtype trong PyTorch AdamW `_foreach_lerp_`.
- **Hạn chế dữ liệu**: Nhãn dạng bounding box chữ nhật chưa thực sự tối ưu cho vết nứt (`crack`) vì vết nứt thường kéo dài chéo hoặc ngoằn ngoèo, khiến vùng bên trong box chứa chủ yếu là mặt đường lành lặn, làm sai lệch phân bố đặc trưng thị giác.
