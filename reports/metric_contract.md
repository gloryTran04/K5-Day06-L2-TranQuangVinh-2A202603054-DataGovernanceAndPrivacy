# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | `T06` · Dataset Semantic Integrity & Provenance Gate |
| Primary metric (tên đúng như evaluator in ra) | `Silent-defect Detection F1` |
| Tăng tốt hay giảm tốt? | `higher` (càng cao càng tốt, tối đa = 1.0000) |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | `release` (mỗi bản phát hành ứng viên `r01`..`r20` là 1 đơn vị đánh giá độc lập) |
| Công thức + numerator / denominator | Positive = bản release bị lỗi (`is_defective == 1`) và bị chặn (`canonical_status(decision) != "GO"`, tức `HOLD` hoặc `REWORK`). Với $\text{TP} = \sum \mathbb{I}(\text{is\_defective}=1 \land \text{blocked}=1)$, $\text{FP} = \sum \mathbb{I}(\text{is\_defective}=0 \land \text{blocked}=1)$, $\text{FN} = \sum \mathbb{I}(\text{is\_defective}=1 \land \text{blocked}=0)$: $$\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}, \quad \text{F1} = \frac{2\cdot\text{TP}}{2\cdot\text{TP}+\text{FP}+\text{FN}}$$ |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | Ghép chính xác 1–1 theo khóa `release_id` (`gt.merge(pred, on="release_id", how="left")`). Không được trùng dòng `release_id`. Bản release bị thiếu quyết định sẽ bị tính mặc định là `GO` (trường hợp xấu nhất — lọt lỗi). |
| Candidate pool cố định | 20 bản release ứng viên (`r01`..`r20`) trong `fixtures/t06_release_gate/releases.csv`, đối chiếu với bản gốc `releases/v1.0` (100 ảnh, 697 annotations, `label_map_version = lm-3`) và `releases/label_maps.json`. |
| Review budget B / K | `B = 10` (đưa tối đa 10 bản release có điểm nghi ngờ `score` cao nhất vào `review_queue.csv` kèm bằng chứng trực quan trong `contact_sheet.html`). |
| Seed | `42` |
| Threshold policy (chọn trên dev như thế nào) | Chốt trên tập `dev` (`r01`..`r10`) dựa trên bất biến ngữ nghĩa (semantic invariants), tách biệt hoàn toàn với thay đổi lành tính (thêm 1 ảnh mới hợp lệ `seq_extra` hoặc nhiễu tọa độ $< 1.0\text{ px}$ ở $\le 5$ bbox): (1) `stale_map`: danh sách `categories` không khớp đủ tập ID của `label_maps[label_map_version]` hoặc tồn tại `category_id` mồ côi; (2) `class_swap`: tập ID khớp nhưng ánh xạ `id -> name` bị hoán vị so với `label_maps[label_map_version]`; (3) `id_collision`: số ID trùng trong `annotations` hoặc `images` $\ge 1$; (4) `dropped_frames`: số `image.id` của bản gốc `v1.0` bị mất trong bản ứng viên $\ge 1$; (5) `bbox_shift`: số annotation có độ lệch tọa độ $\max(|\Delta x|, |\Delta y|, |\Delta w|, |\Delta h|) > 2.0\text{ px}$ so với bản gốc $\ge 10$. Nếu vi phạm bất kỳ quy tắc nào $\rightarrow$ `HOLD` (kèm `root_cause`), ngược lại $\rightarrow$ `PASS` (`clean`). |
| Eval split (không dùng để tune) | `split = eval` (`r11`..`r20`, $N = 10$ bản phát hành; tập `dev` gồm `r01`..`r10`, $N = 10$ bản phát hành). Báo cáo cả kết quả trên `eval` (chính thức) và `all` ($N = 20$, tự kiểm tra). |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | (1) `False-block rate` = số bản release sạch bị chặn / tổng số bản release sạch (mục tiêu = 0.0000); (2) `1 - Defect recall` = tỷ lệ bỏ lọt bản release lỗi ngầm; (3) `Root-cause accuracy` = tỷ lệ chẩn đoán đúng nguyên nhân gốc trên các bản lỗi bị chặn. |
| Slices sẽ báo cáo (≥2) | 6 slices theo `corruption_type` (`stale_map`, `dropped_frames`, `bbox_shift`, `class_swap`, `id_collision`, `clean`) và 2 slices theo `split` (`dev`, `eval`). |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | Tính khoảng tin cậy 95% Wilson CI (`src.metrics.wilson_ci`) cho tỷ lệ phát hiện lỗi (`Defect recall`) và tỷ lệ chặn nhầm (`False-block rate`), đồng thời lập bảng kết quả từng bản phát hành (`r01`..`r20`). |
| Ngày giờ chốt + người chốt | `2026-11-14T15:30:00+07:00` · Tran Quang Vinh (`2A202603054` — Data Governance Lead) |

Nếu baseline = 0 thì ghi rõ "baseline = 0" thay vì phần trăm cải thiện.
