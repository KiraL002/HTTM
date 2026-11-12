# 🚗 AcciVision — Hệ Thống Phát Hiện & Phân Loại Tai Nạn Giao Thông Thông Minh

> **AcciVision** (*Accident + Vision*) là hệ thống phát hiện, xác thực và phân loại tai nạn giao thông thời gian thực sử dụng kết hợp **Thị Giác Máy Tính (YOLO11 + ByteTrack + MobileNetV3)** và **Phân Tích Động Học Vật Lý (Random Forest Classifier)**.

---

## 📋 Tổng Quan Kiến Trúc Đa Tầng (3-Stage Hybrid Pipeline)

Hệ thống AcciVision xử lý video giao thông qua quy trình 8 bước kết hợp **Động học quỹ đạo (Kinematics)**, **Biến dạng hình học khung bao (Bounding-Box Geometry)** và **Xác thực thị giác kép (Visual Verification)**:

```
Video đầu vào (CCTV / Camera hành trình / Giao lộ)
    ↓
[1] Phát hiện phương tiện & tai nạn (YOLO11s - yolo11s_accident_vn.pt, 7 lớp)
    ↓
[2] Theo dõi đa đối tượng liên tục (ByteTrack Multi-Object Tracker)
    ↓
[3] Quản lý & Nội suy quỹ đạo (Trajectory Management & Gap Interpolation)
    ↓
[4] Chuyển đổi phối cảnh BEV & Tự động hiệu chỉnh tỷ lệ (Perspective Transform + Auto-Scale)
    ↓
[5] Trích xuất 26 đặc trưng động học & biến dạng khung bao (Kinematic & BBox Geometry)
    ↓
[6] Phân loại tai nạn tốc độ cao (Random Forest Cython Engine + MobileNetV3 Visual Verifier)
    ↓
[7] Quản lý vòng đời sự kiện & Phân loại 9 kịch bản va chạm (Event Detector)
    ↓
Video kết quả + Ảnh cắt hiện trường + Báo cáo CSV/JSON + Giao diện Streamlit Dashboard
```

### 🎯 9 Kịch Bản & Loại Tai Nạn Hệ Thống Phân Loại Được

| Mã định danh (`accident_type`) | Nhãn hiển thị Tiếng Việt (`accident_label_vn`) | Đặc trưng nhận diện chính |
| :--- | :--- | :--- |
| `rear_end_collision` | **💥 Va chạm phía sau (Tông đuôi)** | Hai xe cùng chiều ($\Delta\theta \le 45^\circ$), khoảng cách $\le 3.5\text{m}$, tốc độ tiếp cận cao |
| `head_on_collision` | **⚡ Va chạm đối đầu (Trực diện)** | Hai xe ngược chiều ($\Delta\theta > 125^\circ$), xung đột trực diện, giảm tốc đột ngột |
| `side_impact_collision` | **📐 Va chạm sườn xe / Giao lộ (T-bone)** | Va chạm cắt ngang tại nút giao ($45^\circ < \Delta\theta \le 125^\circ$), chồng lấp IoU |
| `car_motorcycle_collision` | **🛵 Va chạm xe máy với ô tô** | Tương tác cự ly gần giữa `motorcycle`/`bicycle` và `car`/`truck`/`bus` |
| `motorcycle_fall_or_crash` | **🏍️ Tai nạn xe máy tự ngã / Trượt đổ** | Xe máy giảm tốc nhanh kèm biến dạng tỷ lệ khung bao $\Delta(W/H) \ge 45\%$ |
| `vehicle_pedestrian_collision` | **🚶 Phương tiện va chạm người đi bộ** | Xung đột quỹ đạo cự ly gần giữa phương tiện cơ giới và `person` |
| `multi_vehicle_pileup` | **⛓️ Tai nạn liên hoàn ($\ge 3$ xe)** | Từ 3 phương tiện trở lên va chạm dồn toa trong cùng vùng không gian |
| `single_vehicle_spin_or_rollover` | **🔄 Xe mất lái quay đầu / Lật xe** | Tốc độ xoay thân xe `yaw_rate >= 80°/s`, tổng góc lệch $\ge 90^\circ$ hoặc biến dạng khung bao |
| `single_vehicle_crash` | **🛑 Xe đâm vật cản / Dải phân cách** | Sự cố đơn xe đang chạy tốc độ cao giảm tốc khẩn cấp về $0\text{ m/s}$ (`stopped_after_motion`) |

---

## 📊 Hiệu Năng Thực Nghiệm & Bộ Dữ Liệu

Mô hình được huấn luyện và đánh giá trên bộ dữ liệu chuẩn hóa [`dataset/cleaned_train_dataset.csv`](dataset/cleaned_train_dataset.csv) gồm **117.821 mẫu đặc trưng** (93% trích xuất từ 150 video CCTV/Dashcam thực tế + 7% kịch bản mô phỏng bổ sung cho xe máy, xe đạp, người đi bộ):

| Chỉ số đánh giá (Holdout Test Set - 23.565 mẫu) | Trước khi làm sạch nhãn | Sau khi chuẩn hóa & nâng cấp |
| :--- | :---: | :---: |
| **Accuracy (Độ chính xác tổng thể)** | 94.80% | **99.02%** |
| **Precision (Độ chuẩn xác cảnh báo tai nạn)** | 77.66% | **84.70%** |
| **Recall (Độ nhạy phát hiện tai nạn)** | 98.96% | **98.93%** |
| **F1-Score (Trung bình điều hòa)** | 87.03% | **91.26%** |
| **ROC-AUC** | 0.9840 | **0.9972** |

> 📈 Trọn bộ biểu đồ đánh giá độ phân giải cao (300 DPI) phục vụ báo cáo đồ án được lưu tại thư mục [`outputs/evaluation_charts/`](outputs/evaluation_charts/):
> - `01_confusion_matrix.png` — Ma trận nhầm lẫn (Số lượng mẫu & Tỷ lệ %)
> - `02_roc_and_pr_curves.png` — Đường cong ROC-AUC và Precision-Recall
> - `03_feature_importance.png` — Top 15 đặc trưng động học quan trọng nhất
> - `04_dataset_accident_distribution.png` — Phân bố 9 loại tai nạn và phương tiện
> - `05_metrics_comparison_before_after.png` — So sánh hiệu năng Trước & Sau khi làm sạch dữ liệu

---

## 📁 Cấu Trúc Dự Án

```
HTTM/
├── app.py                              # Giao diện web Streamlit Dashboard
├── main.py                             # Pipeline xử lý chính & CLI
├── config.yaml                         # Tệp cấu hình hệ thống
├── requirements.txt                    # Danh sách thư viện phụ thuộc
├── train_rf_baseline.py                # Script huấn luyện mô hình Random Forest
├── core/                               # Các module xử lý lõi
│   ├── tracker.py                      #   → Phát hiện (YOLO11) & Theo dõi đa đối tượng (ByteTrack)
│   ├── trajectory.py                   #   → Quản lý quỹ đạo chuyển động
│   ├── transformer.py                  #   → Chuyển đổi phối cảnh BEV & Auto-Scale theo kích thước xe
│   ├── features.py                     #   → Trích xuất đặc trưng động học & biến dạng khung bao
│   ├── classifier.py                   #   → Phân loại tai nạn (Random Forest Cython Engine - nhanh gấp 25x)
│   ├── visual_classifier.py            #   → Xác thực thị giác kép (MobileNetV3 Visual Verifier)
│   └── event_detector.py               #   → Phát hiện, gom xe & phân loại 9 kịch bản tai nạn
├── utils/                              # Các hàm tiện ích
│   ├── visualization.py                #   → Vẽ bounding box, trajectory, trạng thái cảnh báo
│   ├── math_utils.py                   #   → Hàm toán học (khoảng cách, góc, vận tốc)
│   └── interpolation.py                #   → Nội suy quỹ đạo khi bị che khuất tạm thời
├── models/                             # Trọng số các mô hình AI đã huấn luyện
│   ├── yolo11s_accident_vn.pt          #   → Mô hình YOLO11s chuyên biệt giao thông VN (mAP50 = 85.3%)
│   ├── yolo11n_accident_vn.pt          #   → Mô hình YOLO11n nhẹ cho thiết bị cấu hình thấp
│   ├── accident_rf_model.pkl           #   → Mô hình Random Forest phân loại động học (F1 = 91.26%)
│   └── visual_accident_classifier.pth  #   → Mô hình MobileNetV3 xác thực ảnh cắt hiện trường
├── dataset/                            # Bộ dữ liệu huấn luyện & công cụ xử lý
│   ├── cleaned_train_dataset.csv       #   → Tập dữ liệu chuẩn hóa 117.821 mẫu (có đủ 4 cột nhãn)
│   ├── combined_train_dataset.csv      #   → Tập dữ liệu kết hợp từ 150 video thực tế đã làm sạch
│   ├── clean_and_relabel_dataset.py    #   → Script kiểm toán, lọc nhiễu và gán nhãn đa lớp tự động
│   ├── extract_dataset_from_videos.py  #   → Script trích xuất đặc trưng từ video thực tế mới
│   └── evaluate_and_plot_charts.py     #   → Script xuất bộ 5 biểu đồ đánh giá mô hình (300 DPI)
└── outputs/                            # Kết quả đầu ra
    ├── videos/                         #   → Video kết quả đã vẽ cảnh báo
    ├── snapshots/                      #   → Ảnh chụp cận cảnh hiện trường từng vụ tai nạn
    └── evaluation_charts/              #   → Bộ biểu đồ đánh giá thực nghiệm (300 DPI)
```

---

## 🛠 Cài Đặt & Hướng Dẫn Sử Dụng

### 1. Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

### 2. Chạy Giao Diện Web (Streamlit Dashboard)

```bash
python -m streamlit run app.py --server.port 8501
```
Mở trình duyệt tại `http://localhost:8501` để tải video lên, xem kết quả phân tích thời gian thực, ảnh cắt hiện trường và mô tả chi tiết từng vụ va chạm.

### 3. Chạy Bằng Dòng Lệnh (CLI)

```bash
# Phân tích video cụ thể ở chế độ không mở cửa sổ (nhanh nhất)
python main.py --config config.yaml --input dataset/input_video/traffic.mp4 --no-display
```

### 4. Huấn Luyện Lại & Xuất Biểu Đồ Đánh Giá

```bash
# Làm sạch và chuẩn hóa nhãn dữ liệu
python dataset/clean_and_relabel_dataset.py

# Huấn luyện mô hình Random Forest
python train_rf_baseline.py --csv dataset/cleaned_train_dataset.csv --output models/accident_rf_model.pkl --n-estimators 250 --max-depth 14

# Xuất bộ 5 biểu đồ đánh giá mô hình (300 DPI) vào outputs/evaluation_charts/
python dataset/evaluate_and_plot_charts.py
```

---

## 🔬 Công Nghệ Sử Dụng

| Thành phần | Công nghệ / Thuật toán |
| :--- | :--- |
| **Phát hiện phương tiện & tai nạn** | Ultralytics YOLO11s (`yolo11s_accident_vn.pt`) |
| **Theo dõi đa đối tượng** | ByteTrack + Kalman Filter + Nội suy tuyến tính |
| **Chuyển đổi phối cảnh** | OpenCV Homography BEV + **Auto-Scale theo kích thước xe** |
| **Trích xuất đặc trưng** | 26 đặc trưng Động học vật lý + Biến dạng hình học khung bao ($\Delta W/H$) |
| **Phân loại động học** | Random Forest Classifier (Tối ưu Cython Tree Inference) |
| **Xác thực thị giác kép** | PyTorch MobileNetV3 (`visual_accident_classifier.pth`) |
| **Giao diện trực quan** | Streamlit Dashboard |

---

> **AcciVision** — *Phát hiện tai nạn thông minh, bảo vệ an toàn giao thông.*
