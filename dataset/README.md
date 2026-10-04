# 📊 AcciVision — Hướng Dẫn Bộ Dữ Liệu (Dataset Guide)

Thư mục `dataset/` chứa các bộ dữ liệu đặc trưng phục vụ huấn luyện, thẩm định (validation) và kiểm thử (testing) hệ thống phát hiện tai nạn giao thông AcciVision.

---

## 1. Các File Dataset Có Sẵn

| Tên File | Số Mẫu | Mục Đích | Mô Tả |
| :--- | :--- | :--- | :--- |
| **`output_features.csv`** | 608 | Dữ liệu video thực tế | Đặc trưng trích xuất từ video `traffic.mp4` qua pipeline AcciVision (YOLO + ByteTrack + Perspective Transform). |
| **`train_diverse_features.csv`** | 6,615 | Tập huấn luyện (Train) | Đa dạng kịch bản: va chạm phía sau, đối đầu, đâm sườn, mất lái, dừng đột ngột, cao tốc bình thường, dừng-chạy đô thị, chuyển làn. |
| **`val_diverse_features.csv`** | 1,400 | Tập thẩm định (Validation) | Đánh giá độ tổng quát hóa của mô hình trên các kịch bản độc lập. |
| **`test_diverse_features.csv`** | 1,435 | Tập kiểm thử (Test) | Bộ benchmark đánh giá hiệu năng cuối cùng. |
| **`combined_train_dataset.csv`** | 7,142 | Tập huấn luyện tổng hợp | Kết hợp đặc trưng video thật và bộ đa kịch bản. **Khuyến nghị dùng để huấn luyện Random Forest.** |
| **`benchmark_diverse_dataset.csv`** | 9,450 | Tập toàn diện | Toàn bộ các chuỗi kịch bản mô phỏng vật lý đa dạng. |

---

## 2. Hướng Dẫn Huấn Luyện Mô Hình

### 2.1 Huấn luyện trên tập kết hợp (Khuyến nghị)

```bash
python train_rf_baseline.py --csv dataset/combined_train_dataset.csv --output models/accident_rf_model.pkl
```

### 2.2 Huấn luyện trên tập đa kịch bản

```bash
python train_rf_baseline.py --csv dataset/train_diverse_features.csv --output models/accident_rf_model.pkl
```

### 2.3 Huấn luyện lại khi có thêm dữ liệu

```bash
python dataset/generate_diverse_dataset.py
python train_rf_baseline.py --csv dataset/combined_train_dataset.csv
```

---

## 3. Trích Xuất Đặc Trưng Từ Video Mới

Nếu bạn có thêm video giao thông (`.mp4`, `.avi`, `.mkv`), sử dụng công cụ trích xuất tự động:

```bash
# Trích xuất từ thư mục video tai nạn:
python dataset/extract_dataset_from_videos.py --video-dir dataset/accident_videos --label-accident --output dataset/extracted_accidents.csv

# Trích xuất từ thư mục video giao thông bình thường:
python dataset/extract_dataset_from_videos.py --video-dir dataset/normal_videos --output dataset/extracted_normal.csv
```

---

## 4. Bộ Dữ Liệu Video Công Khai (Public Benchmarks)

Các nguồn video tai nạn thực tế từ cộng đồng học thuật quốc tế:

1. **DoTA (Detection of Traffic Anomaly)**
   - 4,677 video giám sát thực tế với 9 loại bất thường
   - [GitHub](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)

2. **CCD (Car Crash Dataset)**
   - 4,500 video va chạm từ camera hành trình
   - [GitHub](https://github.com/coeusguo/Car-Crash-Dataset-CCD)

3. **CADP (Car Accident Detection and Prediction)**
   - 1,416 video từ camera CCTV giám sát giao thông
   - [GitHub](https://github.com/ankitshah009/Car-Accident-Detection-and-Prediction-Dataset)

4. **DAD / A3D (Dashcam Accident Dataset)**
   - 1,750 video hành trình với nhãn thời điểm va chạm
