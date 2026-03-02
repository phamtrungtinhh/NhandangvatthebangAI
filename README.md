# Hệ thống Nhận Diện và Phân Tích Vật Thể Ứng Dụng YOLO

## 1. Tên dự án
**AI-based Object Detection and Analysis System using YOLO**  
(Hệ thống nhận diện và phân tích vật thể sử dụng YOLO)

## 2. Mục tiêu và ý nghĩa của dự án
### 2.1 Mục tiêu
- Xây dựng hệ thống nhận diện vật thể tự động từ **ảnh** và **video**.
- Ứng dụng mô hình học sâu YOLO để phát hiện vật thể nhanh và chính xác.
- Thống kê số lượng vật thể theo từng lớp để hỗ trợ phân tích dữ liệu.
- Hiển thị kết quả trực quan, dễ hiểu, phù hợp cho người dùng phổ thông.

### 2.2 Ý nghĩa
- Minh họa quy trình triển khai một bài toán thị giác máy tính hoàn chỉnh: từ đầu vào dữ liệu đến hiển thị kết quả.
- Tạo nền tảng thực nghiệm cho các bài toán mở rộng như giám sát thông minh, thống kê sản phẩm, hỗ trợ nông nghiệp.
- Góp phần nâng cao khả năng ứng dụng AI vào các sản phẩm thực tế ở quy mô đồ án sinh viên.

## 3. Công nghệ sử dụng

| Công nghệ | Vai trò trong hệ thống | Bước tham gia trong quy trình |
|---|---|---|
| **YOLO (Ultralytics)** | Mô hình chính để phát hiện vật thể, trả về lớp, bounding box và độ tin cậy. | Nhận diện (inference), hậu xử lý kết quả. |
| **Python** | Ngôn ngữ điều phối toàn bộ pipeline, tích hợp mô hình, xử lý dữ liệu và giao diện. | Xuyên suốt toàn bộ quy trình từ nhập dữ liệu đến xuất kết quả. |
| **OpenCV** | Đọc video, tách khung hình, chuyển đổi định dạng ảnh và hỗ trợ vẽ kết quả. | Tiền xử lý ảnh/video, hiển thị kết quả trực quan. |
| **Streamlit (Web App)** | Xây dựng giao diện web cho upload dữ liệu, cấu hình tham số và hiển thị thống kê. | Nhập dữ liệu người dùng, hiển thị đầu ra và dashboard. |
| **NumPy** | Biểu diễn ảnh dạng mảng, xử lý số học và hỗ trợ thao tác dữ liệu trung gian. | Tiền xử lý dữ liệu, tổng hợp và thống kê sau nhận diện. |
| **Pillow (PIL)** | Đọc, chuẩn hóa ảnh đầu vào từ nhiều định dạng trước khi đưa vào mô hình. | Tiếp nhận ảnh và chuẩn hóa đầu vào. |
| **PyTorch** | Nền tảng deep learning để chạy mô hình YOLO trên CPU/GPU. | Thực thi suy luận mô hình và tối ưu hiệu năng tính toán. |
| **Torchvision** | Thư viện hỗ trợ hệ sinh thái PyTorch cho các tác vụ thị giác máy tính. | Hỗ trợ xử lý dữ liệu và môi trường suy luận/model stack. |
| **SQLite** | Lưu lịch sử phân tích (tệp, thời gian, kết quả đếm) phục vụ tra cứu và báo cáo. | Lưu trữ dữ liệu sau khi xử lý, truy xuất lịch sử. |
| **Requests** | Gọi API ngoài để bổ sung thông tin mô tả khi cần. | Hậu xử lý/khai thác thông tin bổ sung cho kết quả. |
| **JSON** | Chuẩn hóa cấu trúc dữ liệu kết quả để lưu trữ và trao đổi dữ liệu. | Đóng gói dữ liệu đầu ra, lưu lịch sử và truyền dữ liệu nội bộ. |
| **CUDA (nếu có GPU)** | Tăng tốc tính toán khi suy luận mô hình, giảm độ trễ xử lý. | Giai đoạn inference và xử lý thời gian thực. |
| **Virtual Environment (`venv`)** | Quản lý phụ thuộc và phiên bản thư viện, đảm bảo tính tái lập khi triển khai. | Cài đặt, cấu hình và vận hành môi trường chạy dự án. |
| **Weights & Biases (WandB)** *(tuỳ chọn)* | Theo dõi thí nghiệm, log chỉ số huấn luyện/đánh giá mô hình. | Giai đoạn huấn luyện, đánh giá và so sánh mô hình. |

## 4. Các chức năng chính của hệ thống
### 4.1 Tải ảnh hoặc video
- Người dùng có thể tải lên tệp ảnh (`.jpg`, `.jpeg`, `.png`) hoặc video (`.mp4`, `.avi`, `.mov`) ngay trên giao diện web.

### 4.2 Nhận diện vật thể
- Hệ thống thực hiện suy luận bằng YOLO để phát hiện vị trí và lớp của vật thể trong ảnh/video.

### 4.3 Đếm số lượng vật thể
- Sau khi phát hiện, hệ thống tự động tổng hợp và đếm số lượng vật thể theo từng lớp.

### 4.4 Hiển thị nhãn tiếng Việt
- Tên lớp vật thể được chuyển đổi sang tiếng Việt để người dùng dễ theo dõi kết quả.

### 4.5 Hiển thị độ chính xác và thời gian xử lý
- Giao diện hiển thị các chỉ số: tổng số vật thể, độ tin cậy trung bình, độ trễ xử lý và thông tin lớp nổi bật.

## 5. Quy trình hoạt động của hệ thống
1. **Người dùng tải dữ liệu** (ảnh/video) lên hệ thống.
2. **Hệ thống tiền xử lý** dữ liệu đầu vào theo định dạng phù hợp cho mô hình.
3. **Mô hình YOLO suy luận** để phát hiện vật thể.
4. **Hậu xử lý kết quả** (lọc, chuẩn hóa, gom trùng lặp) để tăng độ ổn định.
5. **Đếm vật thể và dịch nhãn tiếng Việt** cho kết quả cuối.
6. **Hiển thị trực quan** ảnh có bounding box, bảng thống kê và lưu lịch sử phân tích.

## 6. Ứng dụng thực tế
- **Giám sát an ninh**: Phát hiện người và vật thể trong camera.
- **Bán lẻ/thương mại**: Thống kê sản phẩm, theo dõi kệ hàng.
- **Nông nghiệp**: Theo dõi cây trồng, hoa quả trong ảnh thực địa.
- **Giáo dục nghiên cứu**: Làm nền tảng cho thực hành môn AI/Computer Vision.
- **Phát triển nguyên mẫu**: Mở rộng sang hệ thống cảnh báo hoặc phân tích chuyên sâu.

## 7. Hướng dẫn chạy dự án cơ bản

### 7.1 Yêu cầu môi trường
- Python 3.8 trở lên.
- Kết nối mạng ở lần chạy đầu (nếu cần tải trọng số mô hình).

### 7.2 Cài đặt thư viện
```bash
python -m venv .venv
```

Kích hoạt môi trường ảo:

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Cài dependencies:
```bash
pip install -r requirements.txt
```

### 7.3 Chạy ứng dụng
```bash
streamlit run app.py
```

Sau khi chạy, mở trình duyệt tại địa chỉ local do Streamlit cung cấp (thường là `http://localhost:8501`).

## 8. Kết luận
Đồ án đã xây dựng được một hệ thống nhận diện vật thể theo hướng ứng dụng thực tế, có giao diện thân thiện và quy trình xử lý rõ ràng. Kết quả đạt được là nền tảng tốt để tiếp tục cải tiến về dữ liệu huấn luyện, tối ưu mô hình và mở rộng chức năng trong các hướng nghiên cứu tiếp theo.

---


