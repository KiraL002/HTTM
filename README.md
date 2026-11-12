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
- **Mở kết quả đã lưu**: mở nhanh kết quả đã có, không chạy mô hình lại. Nếu cấu hình đã đổi, vẫn mở được bản lịch sử của cùng nội dung video, kèm thông báo cấu hình cũ; không coi đó là lần phân tích mới. Thông tin xe liên quan, mốc nghi sự cố và ảnh bằng chứng hiển thị trước video kết quả, cả khi xem lại lịch sử.
- **Lịch sử**: xem sự kiện, ID xe, ảnh bằng chứng và video kết quả. Để xóa riêng một lượt, chọn ở **Chọn lượt để xem lại**, mở **Xóa lượt phân tích đã chọn**, tích xác nhận và bấm **Xóa lượt này**. Thao tác xóa cả đánh giá và video/ảnh kết quả của lượt đó; các lượt khác, video gốc và mô hình được giữ nguyên. Tệp kết quả dùng chung chỉ bị xóa khi không còn lượt nào tham chiếu. Mục **Xóa lịch sử và kết quả đã lưu** dùng để xóa toàn bộ; không chọn mục này nếu muốn giữ kết quả video 1–6.

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

## Điểm bất thường và xác minh sự cố

Ứng dụng quét VideoMAE-S/Simple-TAD trước để chọn các đoạn có điểm bất thường cao liên tiếp (10 khung lấy mẫu/giây, cửa sổ 16 khung, bước 4 khung). Trọng số được tiếp tục fine-tune trong [traffic_checker](https://github.com/Pac1FF1zM/traffic_checker), đặt tại `models/traffic_anomaly_videomae_s.pth` và được kiểm tra SHA-256. Đây là trọng số tham khảo DoTA/DADA, chưa được hiệu chỉnh cho CCTV.

Các đoạn được mở thêm 1,2 giây ngữ cảnh mỗi phía rồi gộp. Trong những đoạn này, hệ thống bổ sung cửa sổ cặp xe quanh tiếp xúc đã quan sát và cửa sổ một xe trên lưới 0,2 giây. Sau đó dùng đúng bộ xác minh đã huấn luyện, đặc trưng, điều kiện chuyển động và ngưỡng hiện có. Điểm bất thường không tự tạo sự kiện, ID hay thời điểm va chạm. Khi thiếu bằng chứng, đoạn được ghi **chưa đủ bằng chứng tai nạn**, không tính vào số sự kiện.

Các ứng viên chuyển động/tiếp xúc cũ vẫn được kiểm tra trên toàn video. Sự kiện cũ đã được xác minh giữ nguyên xe, thời điểm và điểm số; nhánh bổ sung chỉ thêm sự kiện không trùng ở đoạn chưa được giải thích. Nếu một đoạn đã có sự kiện được xác minh, không dùng điểm toàn cảnh để gán thêm xe trong đoạn đó; các đoạn còn lại ưu tiên kiểm tra cặp xe trước nhánh một xe. Điều này tránh việc Simple-TAD bỏ sót làm mất kết quả của nhánh cũ. Có thể xem đường điểm và kết quả từng đoạn trong mục **Các đoạn bất thường đã được kiểm tra**, kể cả trong lịch sử mới.

Ngưỡng mở đoạn 0,75 và thoát đoạn 0,5 là cấu hình thử nghiệm; không phải xác suất tai nạn được hiệu chỉnh. Đỉnh điểm số và đầu/cuối cửa sổ không được dùng làm mốc tiếp xúc. Quá trình xử lý mới thực hiện suy luận thật và mất thêm thời gian CPU; chỉ nút mở kết quả đã lưu mới sử dụng lịch sử. Chưa có đánh giá độc lập chứng minh nhánh mới tăng độ chính xác toàn hệ thống. Không huấn luyện lại hoặc đổi ngưỡng bộ xác minh trong thay đổi này.

`core/temporal_anomaly.py` là bộ suy luận thời gian; `utils/temporal_review.py` chọn đoạn và gửi cửa sổ bổ sung tới bộ xác minh. Trạng thái ứng dụng nằm trong `outputs/vehicle_pair_runtime/runtime_policy.json` (`temporal_proposals.enabled`). Không cần các thư mục dự án tham khảo để chạy ứng dụng.

Để đối chiếu riêng một checkpoint mà không thay kết luận ứng dụng:

```powershell
.\venv\Scripts\python.exe -m core.temporal_anomaly dataset/input_video/1.mp4 --checkpoint models/traffic_anomaly_videomae_s.pth --output outputs/reference_review/temporal_check.json
```

Phần tối ưu giải mã đọc một lượt cho các cửa sổ RGB cặp xe chồng lấn (`iter_clips`) giữ đầu vào giống từng pixel với bộ đọc tuần tự cũ. Không tạo cache video mới trên đĩa.

Ghi công: Svetlana Orlova, Tommie Kerssies, Brunó B. Englert và Gijs Dubbelman, *Simplifying Traffic Anomaly Detection with Video Foundation Models*, ICCVW 2025, [bài báo](https://arxiv.org/abs/2507.09338), [mã Simple-TAD](https://github.com/tue-mps/simple-tad). Bộ đọc chuyển attention sang PyTorch SDPA. Mã/trọng số Simple-TAD thuộc CC BY-NC 4.0; xem `licenses/simple-tad-LICENSE.txt`. Bản trọng số tiếp tục fine-tune lấy từ traffic_checker. Phạm vi sử dụng của dự án là học tập/nghiên cứu phi thương mại.
