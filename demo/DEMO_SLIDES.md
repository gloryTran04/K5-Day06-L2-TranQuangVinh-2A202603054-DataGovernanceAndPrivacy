# Track T06 · Dataset Semantic Integrity & Provenance Gate (Showcase Slides)

---

## Slide 1 · Bài toán & Điểm nghẽn (Problem & Risk Gate)

- **Bối cảnh:** Khi nhận các bản phát hành dữ liệu mới (`v1.1-r01` .. `v1.1-r20`) từ đối tác gán nhãn (`vendor_0`, `vendor_1`, `vendor_2`) kế thừa từ bản gốc `releases/v1.0` (100 ảnh, 697 bbox, `label_map: lm-3`), các pipeline CI/CD thông thường chỉ kiểm tra cú pháp JSON, COCO schema và sự tồn tại của file ảnh.
- **Hậu quả khi lọt lỗi ngầm (Silent Semantic Defects):**
  1. **Class Swap (`r07`, `r20`):** Hoán đổi ID giữa `car` (`id=1`) và `pedestrian` (`id=2`) $\rightarrow$ mô hình nhận diện người đi bộ thành ô tô, gây rủi ro an toàn nghiêm trọng.
  2. **Stale Map (`r01`, `r09`):** Dùng nhầm label map cũ `lm-2` (4 class) thay vì `lm-3` (5 class) $\rightarrow$ mất toàn bộ định nghĩa lớp `truck` (`id=5`), sinh nhãn mồ côi.
  3. **Bbox Shift (`r05`, `r10`):** Dịch chuyển hệ thống 210 bbox lệch tới ~30 px $\rightarrow$ suy giảm mAP nghiêm trọng khi huấn luyện.
  4. **ID Collision (`r08`, `r18`):** Trùng lặp 11–12 `annotation.id` $\rightarrow$ ghi đè mất nhãn khi merge/index vào database.
  5. **Dropped Frames (`r02`, `r16`):** Rơi mất 7–9 ảnh trong chuỗi video (`seq3`, `seq1`) $\rightarrow$ đứt gãy chuỗi thời gian và hụt độ phủ.
- **Bẫy False-Block:** Cả 10 bản release sạch (`r03, r04, r06, r11..r15, r17, r19`) đều có thay đổi lành tính (+1 ảnh mới `seq_extra` hoặc nhiễu làm tròn float $\le 1.0\text{ px}$ trên 5 bbox). Gate chặn theo `SHA-256 diff` hoặc `file size diff` sẽ chặn nhầm 100% bản sạch!

---

## Slide 2 · Kiến trúc 5 bước & Bất biến ngữ nghĩa (Semantic Invariants)

```
[01 Load & Provenance]  --> Kiểm tra manifest.csv (SHA-256) + Round-trip COCO <-> YOLO (100 ảnh, 697 boxes)
         |
[02 Semantic Process]   --> Kiểm định 6 bất biến ngữ nghĩa so với Parent v1.0 & label_maps.json:
         |                  (1) Parent SHA-256 provenance   (4) ID Uniqueness (image.id & annotation.id)
         |                  (2) Stale Map vs lm-3 + orphans (5) Frame Completeness & Sequence Continuity
         |                  (3) Class ID<->Name Bijection   (6) Spatial Bbox Stability (tol=2.0px, min_count=10)
         |
[03 Budgeted Rank]      --> Xếp hạng Top-B (B=10) theo suspicion score -> review_queue.csv + contact_sheet.html
         |
[04 Human Review]       --> Reviewer kiểm tra bằng chứng bbox trực quan -> approve/edit (REWORK) -> audit_log.jsonl
         |
[05 Release Decision]   --> Xuất release_decisions.csv + Croissant 1.1 JSON-LD + decision.json (GO/HOLD/REWORK)
```

---

## Slide 3 · Kết quả thực nghiệm: Baseline vs. Ablation vs. PoC

| Metric (Eval Split, $N=10$) | Baseline (Schema & File) | Ablation (Naive Diff Gate) | PoC (Semantic Gate) | $\Delta$ (PoC vs Baseline) |
|---|---|---|---|---|
| **Silent-defect Detection F1** [PRIMARY] | `0.0000` | `0.4615` | **`1.0000`** | `+1.0000` (`baseline=0`) |
| **Defect Recall** (3/3 lỗi ngầm) | `0.0000` | `1.0000` | **`1.0000`** | `+1.0000` (95% Wilson CI: `[0.4385, 1.0000]`) |
| **False-block Rate** (0/7 bản sạch) | `0.0000` | `1.0000` (7/7 bị chặn nhầm) | **`0.0000`** | `0.0000` (95% Wilson CI: `[0.0000, 0.3543]`) |
| **Root-cause Accuracy** | `n/a` | `0.0000` | **`1.0000`** | `3/3` chẩn đoán đúng nguyên nhân |
| **Runtime / 100 items (CPU)** | `1.5700 s` | `1.9200 s` | **`3.9680 s`** | `~0.79 s` cho toàn bộ 20 releases |

- Trên toàn bộ tập ứng viên (`split=all`, $N=20$): PoC đạt **F1 = `1.0000`**, **Defect Recall = `1.0000` (10/10)**, **False-block Rate = `0.0000` (0/10)**, **Root-cause Accuracy = `1.0000` (10/10)** across tất cả 6 slices (`stale_map`, `class_swap`, `bbox_shift`, `id_collision`, `dropped_frames`, `clean`).

---

## Slide 4 · Phân tích Failure Modes & Shock Tests

1. **Failure Mode 1 — Lọt lỗi ngầm ở Schema-Only Baseline (`r07`, `r20`, `r01`, `r08`):**
   - File `annotations.json` của `r20` hợp lệ 100% về cú pháp JSON và kiểu dữ liệu, nhưng đảo `{"id": 2, "name": "car"}` và `{"id": 1, "name": "pedestrian"}`. Baseline chỉ kiểm schema nên cho `PASS`. PoC đối chiếu song ánh `id <-> name` với `label_maps.json["lm-3"]` và chặn chính xác với `root_cause = class_swap`.
2. **Failure Mode 2 — Chặn nhầm bản sạch ở Naive Diff Gate (`r11..r15`, `r17`, `r19`):**
   - Các bản sạch có thêm 1 frame mới hợp lệ (`id=101`, `seq_extra`) và 5 bbox lệch $< 1.0\text{ px}$ do làm tròn số thực khi convert format. PoC tách biệt **mất ảnh gốc** ($\text{missing\_parent\_ids} \ge 1$) với **thêm ảnh mới**, và dùng ngưỡng dung sai $\Delta_{\max} > 2.0\text{ px}$ trên $\ge 10$ boxes để đạt `False-block rate = 0.0000`.
3. **Shock Test — Corrupt JSON & Missing File (`test_t06_shock_corrupt_json_and_missing_file`):**
   - Cắt nửa file `releases/candidates/r03/annotations.json`: Bước `01 load` bắt lỗi lệch mã băm `SHA-256` trong `manifest.csv` (exit code `2`); khi tắt `strict_provenance`, bước `02 process` bắt `JSONDecodeError` và cô lập `r03` sang `HOLD` (`root_cause = corrupt_json`).

---

## Slide 5 · Human-in-the-Loop, Audit Trail & Reproducibility

- **Contact Sheet trực quan (`out/poc/contact_sheet.html`):** Vẽ trực tiếp bbox gốc `v1.0` (xanh lá) và bbox lỗi của ứng viên (đỏ) kèm nhãn nguyên nhân lên ảnh mẫu của từng release trong `review_queue.csv` ($B=10$).
- **Override & Audit Log (`out/poc/audit_log.jsonl`):** Reviewer `Tran Quang Vinh` duyệt 10 release nghi ngờ, giữ nguyên `HOLD` cho 8 bản và chuyển `r05`, `r10` (`bbox_shift`) sang `REWORK` để yêu cầu vendor xuất lại tọa độ, đồng thời bảo toàn file gốc `raw_release_decisions.csv`.
- **Chuẩn hóa Provenance mở rộng:** Xuất tự động `out/poc/croissant_metadata.json` theo chuẩn **MLCommons Croissant 1.1** (tích hợp từ vựng `prov:wasDerivedFrom` và `rai:*`).
