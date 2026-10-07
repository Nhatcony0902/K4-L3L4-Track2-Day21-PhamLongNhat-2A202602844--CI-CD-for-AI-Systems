# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Phạm Long Nhật |
| MSSV | 2A202602844 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/Nhatcony0902/K4-L3L4-Track2-Day21-PhamLongNhat-2A202602844--CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

Cloud: AWS, gồm S3 `income-lab-2a202602844` (DVC remote và model) và EC2 `income-api`.

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | **0.878** |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.846 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.874 |
| 4 | 200 | 0.05 | 3 | 0.7014 | 0.874 |
| 5 | 100 | 0.2 | 5 | **0.7207** | 0.876 |

**Bộ siêu tham số đã chọn:** `n_estimators=100`, `learning_rate=0.2`, `max_depth=5`.

**Lý do:** Bộ này có F1 cao nhất (0.7207). Lần có accuracy cao nhất là lần 1 (0.878), nhưng F1 của nó chỉ đứng thứ ba. Accuracy chỉ dao động 0.846–0.878, còn F1 dao động 0.605–0.721, nên accuracy gần như không phân biệt được các mô hình. Bộ 50 cây với lr 0.05 bị underfit và rơi dưới ngưỡng 0.65. Khi giảm lr xuống 0.05, phải tăng gấp đôi số cây (lần 4) mới gần bằng lần 1.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ 24,8% mẫu thuộc lớp thu nhập > 50K. Một mô hình luôn đoán "thu nhập thấp" đạt accuracy 0,752 mà không tìm được người thu nhập cao nào, nên ngưỡng trên accuracy vẫn để mô hình vô dụng đó qua. F1 lớp dương kết hợp precision và recall của lớp thu nhập cao. Mô hình kia có F1 = 0 và bị gate chặn. Tôi không dùng `average="weighted"` hay `"macro"` vì hai cách này cộng thêm F1 rất cao của lớp đa số, làm điểm bị kéo lên và ngưỡng 0,65 mất ý nghĩa.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `pip install` chạy hơn 20 phút mà không xong. | `boto3` không pin nên pip backtrack qua nhiều phiên bản botocore. | Dùng `uv pip compile` tìm bộ phiên bản tương thích rồi pin các gói AWS. |
| MLflow lỗi `FallbackAsyncAdaptedQueuePool`. | SQLAlchemy 2.1 không tương thích mlflow 2.13. | Pin `sqlalchemy==2.0.54`. |
| Push lên `main` không kích hoạt Actions. | Repo là fork, GitHub tắt workflow cho đến khi chủ repo bật. | Bước 2 chạy bằng `workflow_dispatch`, sau đó bật workflow trong tab Actions để Bước 3 tự chạy khi push. |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7207 | 0.876 |
| Bước 3 (thêm `train_batch2`) | 0.7297 | 0.880 |

**Nhận xét:** Gấp đôi dữ liệu chỉ làm F1 tăng 0,009. Holdout có khoảng 124 mẫu dương, nên đúng thêm vài mẫu đã tạo ra chênh lệch cỡ này. Hai batch cùng phân phối, nên đây không phải bằng chứng rằng thêm dữ liệu luôn tốt hơn. Điều được kiểm chứng là commit dữ liệu tự kích hoạt cả 4 job và VM được cập nhật mô hình mới mà không cần thao tác tay.

---

## 5. Phần Bonus Đã Thực Hiện

Số liệu lấy từ artifact của run #3 (ảnh `06-actions-bonus.png`), huấn luyện trên 44.722 mẫu.

- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: ngưỡng tối ưu là **0.30**, cho F1 = 0.7452, so với 0.7297 ở ngưỡng 0.5. Hạ ngưỡng giúp bắt thêm người thu nhập cao vì lớp dương chiếm thiểu số. Giá trị được ghi vào `report.json` và MLflow.
- [x] Bonus 3 - Báo cáo precision / recall tự động (`src/report.py` → `outputs/detail.txt`): lớp 1 có precision 0.83 và recall 0.65, tức mô hình bỏ sót 43/124 người thu nhập cao nhưng chỉ gán nhầm 17/376 người. Nếu mục đích là tìm khách hàng thu nhập cao, bỏ sót (recall thấp) tốn kém hơn vì mất cơ hội thật, còn gán nhầm chỉ tốn một lần tiếp cận.
- [x] Bonus 5 - Cảnh báo lệch lạc dữ liệu: tỷ lệ lớp dương trong tập train là 24.8%, nằm trong ngưỡng ±5 điểm %. Nếu lệch quá ngưỡng, pipeline in `::warning::` lên trang run. Giá trị được ghi vào `report.json`.
