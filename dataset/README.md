# 📊 AcciVision — Hướng Dẫn Bộ Dữ Liệu (Dataset Guide)

Thư mục `dataset/` chứa các bộ dữ liệu đặc trưng động học và video thực tế đã được **kiểm toán, lọc nhiễu và gán nhãn đa lớp (4 cột nhãn)** phục vụ huấn luyện và đánh giá hệ thống phát hiện tai nạn giao thông AcciVision.

---

## 1. Danh Sách Các Tập Dữ Liệu Hiện Có

| Tên File / Thư Mục | Số Mẫu / Quy Mô | Thành Phần | Mô Tả Chi Tiết |
| :--- | :---: | :--- | :--- |
| **`cleaned_train_dataset.csv`** | **137,130 dòng** | `128,402 NORMAL` (93.6%)<br>`8,728 ACCIDENT` (6.4%) | **Tập huấn luyện chính chuẩn hóa (39 cột, 0 giá trị khuyết).** Hợp nhất dữ liệu thực tế từ 150 video CCTV/Dashcam + **19,309 mẫu thực tế phương tiện 2 bánh & người đi bộ (`real_vru_motorcycle_dataset.csv`)** + dữ liệu đa kịch bản và 14,000 mẫu âm tính khó (hard negatives). |
| **`real_vru_motorcycle_dataset.csv`** | **19,309 dòng** | `16,651 NORMAL`<br>`2,658 ACCIDENT` | **Dữ liệu động học thực tế chuyên biệt cho phương tiện 2 bánh (`motorcycle`, `bicycle`) và người đi bộ (`person`)** trích xuất từ **114 video thực tế** thuộc 5 bộ dữ liệu chuẩn quốc tế (`DoTA`, `CAP-DATA`, `DADA-2000`, `MANUAL-DATA`, `Smart-Dashcam`). |
| **`vru_real_videos/`** | **114 video (`.mp4`)** | `52` Xe máy tai nạn<br>`28` Xe đạp tai nạn<br>`28` Người đi bộ tai nạn<br>`6` Xe máy bình thường | Kho video thực tế đã tải và xác thực từ `VRU-Accident` (`DoTA`, `CAP-DATA`, `DADA-2000`, `MANUAL-DATA`) và `Smart-Dashcam Motorcycle Benchmark`, kèm metadata chi tiết trong `vru_metadata.json`. |
| **`vru_annotations/`** | **8 tệp JSON** | `1,000` nhãn VQA + Dense Caption | Bộ nhãn chuẩn Ground-Truth gốc (`*_VQA_annotation.json` và `*_Dense_Caption.json`) của 4 bộ dữ liệu học thuật `DoTA`, `CAP_DATA`, `DADA_2000`, `MANUAL_DATA`. |
| **`combined_train_dataset.csv`** | **93,606 dòng** | 93.0% Video thực tế<br>7.0% Mô phỏng vật lý | Dữ liệu trích xuất từ 150 video tai nạn thực tế (YouTube/DoTA/Dashcam) đã loại bỏ 17,709 dòng gán nhãn sai. |
| **`train_diverse_features.csv`** | **10,215 dòng** | Đa kịch bản & Đa phương tiện | Tập huấn luyện đa dạng kịch bản va chạm (ô tô, xe máy, xe đạp, người đi bộ, xe tải, xe buýt). |
| **`benchmark_diverse_dataset.csv`** | **13,050 dòng** | Tập kiểm chuẩn đa kịch bản | Toàn bộ các chuỗi kịch bản vật lý đa dạng đã gán nhãn đầy đủ. |
| **`val_diverse_features.csv`** | **1,400 dòng** | Tập thẩm định (Validation) | Đánh giá độ tổng quát hóa của mô hình trên các kịch bản độc lập. |
| **`test_diverse_features.csv`** | **1,435 dòng** | Tập kiểm thử (Test) | Bộ benchmark đánh giá hiệu năng độc lập. |
| **`output_features.csv`** | — | Đầu ra runtime | Tệp đặc trưng tự động xuất ra sau mỗi lần chạy `main.py` hoặc `app.py`. |

---

## 2. Cấu Trúc Nhãn Dữ Liệu (4 Cột Nhãn Cuối Mỗi Dòng)

Mọi tệp dữ liệu huấn luyện trong `dataset/` đều được chuẩn hóa đồng nhất **39 cột** (tương thích 100% với `FEATURE_COLUMNS` trong `core/classifier.py` và `OBJECT_TYPE_MAP` trong `core/tracker.py`) với **4 cột nhãn tường minh ở cuối bảng**:

1. **`status`**: Trạng thái cảnh báo (`"NORMAL"` hoặc `"POSSIBLE ACCIDENT"`).
2. **`label`**: Nhãn nhị phân phục vụ huấn luyện (`0` = Bình thường, `1` = Tai nạn).
3. **`accident_type`**: Mã định danh loại tai nạn tiếng Anh (9 lớp):
   - `normal`
   - `rear_end_collision`
   - `head_on_collision`
   - `side_impact_collision`
   - `car_motorcycle_collision`
   - `motorcycle_fall_or_crash`
   - `vehicle_pedestrian_collision`
   - `single_vehicle_spin_or_rollover`
   - `single_vehicle_crash`
4. **`accident_label_vn`**: Mô tả chi tiết loại tai nạn bằng tiếng Việt (ví dụ: `"Va chạm xe máy với ô tô / xe lớn"`, `"Tai nạn xe máy tự ngã / trượt đường"`, `"Va chạm người đi bộ / xe đạp"`...).

---

## 3. Phân Bố Mẫu Tai Nạn Theo Phương Tiện & Kịch Bản (`cleaned_train_dataset.csv`)

### Theo nhóm phương tiện (`8,728` mẫu tai nạn xác thực):
- **Ô tô con (`car`):** 4,683 mẫu
- **Xe máy (`motorcycle`):** 2,079 mẫu
- **Người đi bộ (`person`):** 704 mẫu
- **Xe tải (`truck`):** 673 mẫu
- **Xe buýt (`bus`):** 350 mẫu
- **Xe đạp (`bicycle`):** 236 mẫu

### Theo kịch bản tai nạn (`accident_label_vn`):
- **Va chạm xe máy với ô tô / xe lớn (`car_motorcycle_collision`):** 2,316 mẫu
- **Va chạm người đi bộ / xe đạp (`vehicle_pedestrian_collision`):** 2,009 mẫu
- **Va chạm phía sau - Tông đuôi (`rear_end_collision`):** 1,800 mẫu
- **Va chạm sườn xe / Giao lộ (`side_impact_collision`):** 994 mẫu
- **Tai nạn xe máy tự ngã / trượt đường (`motorcycle_fall_or_crash`):** 523 mẫu
- **Xe mất lái quay vòng / Lật xe (`single_vehicle_spin_or_rollover`):** 446 mẫu
- **Va chạm đối đầu trực diện (`head_on_collision`):** 406 mẫu
- **Xe đâm vật cản / Dải phân cách (`single_vehicle_crash`):** 234 mẫu

---

## 4. Các Công Cụ Quản Lý & Huấn Luyện Dữ Liệu

```bash
# 1. Làm sạch nhiễu nhãn và tái tạo bộ dữ liệu chuẩn hóa (137,130 mẫu)
python dataset/clean_and_relabel_dataset.py

# 2. Huấn luyện mô hình Random Forest trên bộ dữ liệu chuẩn hóa
python train_rf_baseline.py

# 3. Xuất bộ 5 biểu đồ đánh giá thực nghiệm (300 DPI) vào outputs/evaluation_charts/
python dataset/evaluate_and_plot_charts.py
```
