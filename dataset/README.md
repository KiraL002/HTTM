# Dữ liệu dự án

- `input_video/`: sáu video tai nạn demo; giữ riêng với dữ liệu đánh giá độc lập.
- `TAD-benchmark/`: 404 video, gồm 277 ACCIDENT và 127 NORMAL. Nhãn cấp video không có nghĩa mọi frame hoặc mọi xe trong clip đều đang gặp tai nạn.
- `normal_sources/aicity21_extracted/`: 99 tệp nguồn AI City, gồm video, tài liệu và điều khoản sử dụng.
- `normal_sources/detrac_holdout_frames/`: 10.000 ảnh JPEG nguồn của 20 đoạn DETRAC; cần giữ để đánh giá theo frame gốc.
- `annotations/`: nhãn tham chiếu và kết quả duyệt; giữ nguyên cả nguồn nhãn và thông tin chưa chắc chắn.
- `httm_training_checkpoint.joblib`: đặc trưng của lượt trích 507 video train ACCIDENT, không chứa video gốc.
- `pair_probe_features.npz`: đặc trưng của phép thử cặp, giữ để đối chiếu.

Video ACCIDENT gốc dùng cho các lượt chuẩn bị nằm ngoài thư mục này trên ổ D. Các script Kaggle có thể đọc từ dataset gắn vào notebook; không suy ra video gốc từ checkpoint đặc trưng.

Các nhãn NORMAL/positive đã duyệt, manifest chia tập và hồ sơ nguồn còn nằm trong `outputs/`. Cần giữ chúng khi huấn luyện tiếp. Nhãn do người duyệt xác nhận không được dùng để gán sẵn kết luận cho video khi chạy ứng dụng.

Mã huấn luyện TAD/X3D, Temporal, RF và cặp xe nằm ở thư mục gốc. Các ZIP setup được giữ khi còn chứa phiên bản mã, encoder hoặc metadata gốc riêng. Không đổi tên trọng số `.pt` thành `.pkl` và không ghi đè model chỉ vì trùng tên.
