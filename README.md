# 🚗 AcciVision — Hệ Thống Phát Hiện Tai Nạn Giao Thông Thông Minh

> **AcciVision** (Accident + Vision) là hệ thống phát hiện tai nạn giao thông thời gian thực sử dụng Thị giác Máy tính (Computer Vision) và Học Máy (Machine Learning).

---

## 📋 Tổng Quan Hệ Thống

Hệ thống AcciVision sử dụng pipeline xử lý đa tầng để phân tích video giao thông và phát hiện sự kiện tai nạn:

```
Video đầu vào
    ↓
[1] Phát hiện phương tiện (YOLOv8)
    ↓
[2] Theo dõi đa đối tượng (ByteTrack)
    ↓
[3] Quản lý quỹ đạo (Trajectory Management)
    ↓
[4] Chuyển đổi phối cảnh (Perspective Transform → Bird's Eye View)
    ↓
[5] Trích xuất đặc trưng động học (Kinematic Feature Extraction)
    ↓
[6] Phân loại tai nạn (Random Forest Classifier)
    ↓
[7] Phát hiện & quản lý sự kiện (Event Detection & Lifecycle)
    ↓
Video kết quả + Báo cáo CSV + Dashboard
```

### Các Loại Tai Nạn Phát Hiện Được

| Loại tai nạn | Mô tả |
| :--- | :--- |
| Va chạm phía sau (Rear-end) | Xe đâm vào đuôi xe phía trước |
| Va chạm đối đầu (Head-on) | Hai xe đâm trực diện từ hai hướng ngược nhau |
| Va chạm sườn/chữ T (Side Impact) | Xe đâm vào sườn xe khác |
| Va chạm liên hoàn (Pileup) | Nhiều xe va chạm liên tiếp |
| Mất lái quay vòng (Spin/Rollover) | Xe mất kiểm soát xoay vòng hoặc lật |
| Mất lái chệch quỹ đạo (Loss of Control) | Xe chệch khỏi làn đường |
| Dừng đột ngột (Sudden Stop) | Phanh gấp hoặc đâm vật cản |

---

## 🛠 Cài Đặt

### Yêu Cầu Hệ Thống
- Python 3.9 trở lên
- pip (Python Package Manager)

### Bước 1: Tạo Môi Trường Ảo

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Bước 2: Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

Hoặc cài thủ công:

```bash
pip install ultralytics opencv-python pandas numpy scikit-learn streamlit pyyaml torch torchvision joblib
```

---

## 🚀 Hướng Dẫn Sử Dụng

### Chạy Giao Diện Web (Streamlit Dashboard)

```bash
streamlit run app.py
```

Trình duyệt sẽ mở tại `http://localhost:8501` với giao diện tải video và phân tích trực quan.

### Chạy Bằng Dòng Lệnh (CLI)

```bash
# Chạy với cấu hình mặc định
python main.py --config config.yaml

# Chạy với video cụ thể
python main.py --config config.yaml --input path/to/video.mp4

# Chạy không hiển thị cửa sổ (cho server)
python main.py --config config.yaml --no-display
```

---

## 📁 Cấu Trúc Dự Án

```
AcciVision/
├── app.py                  # Giao diện web Streamlit Dashboard
├── main.py                 # Pipeline xử lý chính & CLI
├── config.yaml             # Tệp cấu hình hệ thống
├── requirements.txt        # Danh sách thư viện phụ thuộc
├── train_rf_baseline.py    # Script huấn luyện mô hình Random Forest
├── core/                   # Các module xử lý lõi
│   ├── detector.py         #   → Phát hiện phương tiện (YOLOv8)
│   ├── tracker.py          #   → Theo dõi đa đối tượng (ByteTrack)
│   ├── trajectory.py       #   → Quản lý quỹ đạo chuyển động
│   ├── transformer.py      #   → Chuyển đổi phối cảnh (BEV)
│   ├── features.py         #   → Trích xuất đặc trưng động học
│   ├── classifier.py       #   → Phân loại tai nạn (Random Forest)
│   └── event_detector.py   #   → Phát hiện & quản lý sự kiện
├── utils/                  # Các hàm tiện ích
│   ├── visualization.py    #   → Vẽ bounding box, trajectory, trạng thái
│   ├── math_utils.py       #   → Hàm toán học (khoảng cách, góc, tốc độ)
│   └── interpolation.py    #   → Nội suy quỹ đạo
├── models/                 # Mô hình đã huấn luyện
│   ├── yolov8n.pt          #   → Mô hình YOLOv8 Nano
│   └── accident_rf_model.pkl #  → Mô hình Random Forest phân loại tai nạn
├── dataset/                # Bộ dữ liệu huấn luyện & kiểm thử
│   ├── input_video/        #   → Video đầu vào
│   └── *.csv               #   → Các tệp đặc trưng (features)
└── outputs/                # Kết quả đầu ra (video, CSV)
```

---

## ⚙️ Cấu Hình (config.yaml)

Tệp `config.yaml` cho phép tùy chỉnh toàn bộ pipeline. Các thông số quan trọng:

| Thông số | Mô tả | Giá trị mặc định |
| :--- | :--- | :--- |
| `yolo.confidence` | Ngưỡng tin cậy phát hiện YOLOv8 | 0.30 |
| `classifier.threshold` | Ngưỡng xác suất phân loại tai nạn | 0.70 |
| `perspective.enabled` | Bật/tắt chuyển đổi phối cảnh | true |
| `features.temporal_window` | Số frame cửa sổ trượt thống kê | 8 |
| `output.draw_trajectory` | Hiển thị quỹ đạo trên video | true |

---

## 📊 Huấn Luyện Mô Hình

### Huấn luyện trên tập dữ liệu tổng hợp (khuyến nghị)

```bash
python train_rf_baseline.py --csv dataset/combined_train_dataset.csv --output models/accident_rf_model.pkl
```

### Trích xuất đặc trưng từ video mới

```bash
python dataset/extract_dataset_from_videos.py --video-dir path/to/videos --label-accident --output dataset/new_features.csv
```

---

## 🔬 Công Nghệ Sử Dụng

| Thành phần | Công nghệ |
| :--- | :--- |
| Phát hiện đối tượng | YOLOv8 (Ultralytics) |
| Theo dõi đa đối tượng | ByteTrack |
| Chuyển đổi phối cảnh | OpenCV Perspective Transform |
| Trích xuất đặc trưng | Phân tích động học (Kinematic Analysis) |
| Phân loại tai nạn | Random Forest (scikit-learn) |
| Giao diện web | Streamlit |
| Xử lý video | OpenCV |

---

## 📝 Ghi Chú

- **Track ID** là mã định danh phiên theo dõi (ByteTrack session ID), **không phải** biển số xe.
- Hệ thống cần **calibration camera** (điều chỉnh `perspective.source_points` trong `config.yaml`) để tính toán khoảng cách/tốc độ chính xác theo mét thực tế.
- Mô hình YOLOv8 Nano (`yolov8n.pt`) được chọn để tối ưu tốc độ xử lý. Có thể thay bằng `yolov8s.pt`, `yolov8m.pt` để tăng độ chính xác.

---

## 📜 Giấy Phép

Dự án phục vụ mục đích nghiên cứu và học thuật.

---

> **AcciVision** — *Phát hiện tai nạn, bảo vệ an toàn giao thông.*
