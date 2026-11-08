# AcciVision

Ứng dụng Streamlit phân tích video giao thông trên máy tính. Theo dõi ô tô, xe tải, xe buýt, xe máy, xe đạp và người đi bộ. Nhận diện sự cố hiện chỉ áp dụng cho ô tô, xe tải, xe buýt; kết quả là **nghi sự cố**, chưa phải xác nhận tai nạn bởi con người.

## Chạy ứng dụng

### Cài mới từ GitHub trên Windows

Dùng Python 3.14, tương ứng môi trường đã kiểm tra với các mô hình lưu trong repo:

```powershell
git clone https://github.com/KiraL002/HTTM.git
cd HTTM
py -3.14 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

Repo chứa mã nguồn, cấu hình và các trọng số cần cho ứng dụng. Video gốc, bộ dữ liệu huấn luyện, cache, lịch sử và video kết quả không được đưa lên GitHub. Sau khi cài, chọn **Tải video** để dùng video trên máy; có thể đặt video mẫu riêng trong `dataset/input_video/`. Không cần tải toàn bộ bộ dữ liệu để chạy ứng dụng. Lịch sử được tạo mới trên máy chạy.

Ứng dụng kiểm tra checksum mã nguồn và trọng số. Giữ nguyên byte tệp khi clone (repo có `.gitattributes` ngăn đổi kiểu xuống dòng). Sửa mã suy luận cần kiểm tra lại cấu hình toàn vẹn trước khi sử dụng kết quả lưu. Bản mặc định dùng CPU; video có thể mất nhiều phút để xử lý.

### Chạy sau khi đã cài

Nhấp đúp `Chay_demo.cmd`, hoặc chạy tại `D:\HTTM`:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Mở http://127.0.0.1:8501. Chọn video mẫu hoặc tải MP4/MOV/AVI/MKV, tối đa 200 MB và khoảng 2 phút.

- **Bắt đầu phân tích / Phân tích lại**: chạy toàn bộ mô hình từ video gốc. Dữ liệu suy luận trung gian nằm trong thư mục tạm và tự dọn khi kết thúc.
- **Mở kết quả đã lưu**: mở nhanh kết quả đã có, không chạy mô hình lại.
- **Lịch sử**: xem sự kiện, ID xe, ảnh bằng chứng và video kết quả. Mục **Xóa lịch sử và kết quả đã lưu** xóa các bản ghi cùng video/ảnh đầu ra, giữ nguyên video gốc và mô hình.

## Cấu trúc sau dọn dẹp

- `app.py`: giao diện; `main.py`: dòng lệnh.
- `evaluate_vehicle_pairs.py`: bộ thu thập theo dõi và tạo ứng viên, được pipeline nhập trực tiếp nên cần giữ.
- `core/`, `utils/`: phát hiện xe, theo dõi, xác minh sự cố, lưu trữ và xuất video.
- `config.yaml`, `configs/`, `.streamlit/`: cấu hình.
- `models/`: trọng số; `venv/`: môi trường Python đang dùng.
- `dataset/`: dữ liệu gốc, nhãn và đặc trưng phục vụ huấn luyện tiếp.
- `outputs/vehicle_pair_training/`, `outputs/vehicle_pair_preparation/`, `outputs/single_vehicle_training/`: mô hình và hồ sơ huấn luyện cần giữ.
- `outputs/vehicle_pair_runtime/runtime_policy.json`: trọng số, ngưỡng và checksum mã đã kiểm tra.
- `outputs/history.sqlite3`, `outputs/pair_demo/`: lịch sử và các kết quả người dùng đã lưu; sao lưu cùng nhau.

Các script huấn luyện/đánh giá cũ, thư mục `tests`, các thử nghiệm và gói Kaggle đã được gom vào `D:\HTTM_archive\research_20261107_170018.zip`. Gói giữ đường dẫn tương đối và `ARCHIVE_MANIFEST.json` chứa SHA-256 từng tệp. Khi cần tiếp tục một thí nghiệm, giải nén các thành phần liên quan về đúng đường dẫn gốc; không giải nén đè toàn bộ lên phiên bản ứng dụng mới. Các script cũ thường kiểm tra checksum và đường dẫn gốc.

## Nguyên lý và giới hạn

YOLOv8m + ByteTrack → xác minh cặp xe bằng đặc trưng hình học và hình ảnh; nếu chưa có sự kiện cặp xe, kiểm tra thêm nhánh sự cố một xe. Cảnh báo đỏ giữ từ mốc phát hiện đến cuối video; khung ID vẫn hiện khi NORMAL.

Nhánh hình học một xe loại các cửa sổ nhảy ID và yêu cầu xe đã có chuyển động trước bất thường (trên 0,6 đường chéo khung xe/giây trên ảnh). Đoạn giảm tốc được kiểm tra thêm trên lưới thời gian lệch 0,2 giây; hai cửa sổ xoay mạnh phải cùng vượt ngưỡng RGB. Không thay đặc trưng đầu vào hoặc trọng số đã train.

Nếu một sự kiện giảm tốc được mô hình xác minh, hệ thống có thể quy về **nghi va chạm hai xe** khi có thêm quan sát hai ID tách nhau rồi tiến sát nhau và xe thứ hai đồng thời giảm chuyển động, đổi hướng. Nếu nhiều xe cùng thỏa điều kiện thì giữ kết quả một xe, không tự chọn đối tác. Đây là nhánh kết hợp bằng chứng chuyển động/tiếp xúc; không có nghĩa bộ phân loại cặp xe đã vượt ngưỡng. Mốc hiển thị là tiếp xúc khung ảnh ước lượng, sử dụng cả ngữ cảnh sau đó để xác minh trong chế độ xử lý video ngoại tuyến.

NORMAL có nghĩa **chưa phát hiện sự cố**, không bảo đảm video không có tai nạn. Mốc sự kiện là ước lượng, không phải toàn bộ khoảng đầu–cuối tai nạn. Chưa hỗ trợ sự cố xe máy/người đi bộ. Kết quả trên các video demo dùng trong phát triển không đại diện độ chính xác với video bất kỳ. Việc dọn thư mục không thay mô hình hoặc ngưỡng phát hiện.

## Nhãn đối tượng và duyệt kết quả

- Video mới có nhãn `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `person`. ID A thuộc nhánh theo dõi bổ sung, tách khỏi ID xe trong nhánh sự cố. Việc bổ sung đối tượng không tự mở rộng phạm vi nhận diện tai nạn.
- Kết quả có bảng thống kê theo loại đối tượng và xuất CSV. Số ID là số phiên theo dõi, không phải số phương tiện duy nhất ngoài đời; mất dấu có thể làm tăng số này.
- Trong Lịch sử, xem lại video rồi dùng **Duyệt kết quả** để xác nhận, đánh dấu báo nhầm/bỏ sót hoặc ghi không rõ. Đánh giá người dùng lưu riêng với dự đoán AI; không tự sửa nhãn mô hình hoặc tự đưa vào training. Xóa lịch sử cũng xóa đánh giá tương ứng.
- Nhánh đối tượng bổ sung dùng YOLOv8m/ByteTrack ở 640, confidence 0,25 trong giai đoạn xuất video nên làm tăng thời gian xử lý CPU. Nhánh sự cố gốc và các ngưỡng của nó giữ nguyên.
- Tham khảo ý tưởng lớp đối tượng và tách cảnh báo AI khỏi xác nhận người dùng từ [Accident_Detect](https://github.com/K1ethoang/Accident_Detect) và [Surveillance-Camera-System](https://github.com/K1ethoang/Surveillance-Camera-System). Không tích hợp các dịch vụ camera/Kafka/cloud vì phiên bản này xử lý video trên máy.

## Giải thích sự kiện

Kết quả và CSV mô tả dấu hiệu quan sát được: nghi giảm chuyển động đột ngột, nghi xoay hướng mạnh, nghi thay đổi quỹ đạo, hoặc nghi va chạm giữa hai xe. Mô tả được tính từ đặc trưng của từng sự kiện, không dựa vào tên video hay nhãn tham chiếu. Khi thiếu đặc trưng, hệ thống ghi chưa rõ dạng sự cố. Các mô tả này không xác định nguyên nhân vật lý như hỏng phanh/nổ lốp, không xác nhận xe ra khỏi làn hay đâm vật cản và không xác định bên gây tai nạn. Bản ghi AI và ngưỡng nhận diện được giữ nguyên.
