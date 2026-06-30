# Hướng dẫn chạy file main.py

File `main.py` được sử dụng để chạy model đánh giá sentiment ẩn (implicit sentiment analysis) trên dữ liệu đầu vào.

## 1. Cài đặt các thư viện yêu cầu

Trước khi chạy, hãy chắc chắn bạn đã cài đặt các thư viện Python cần thiết:

```bash
pip install pandas openai tqdm scikit-learn backoff
```

## 2. Cấu hình thông số

Mở file `main.py` và tìm đến phần cấu hình trực tiếp (dòng 17-22):

```python
# ========== CẤU HÌNH TRỰC TIẾP ==========
API_KEY = "your-api-key-here"
CSV_FILE = "D:\\Nghiên cứu khoa học\\conference_4\\THOR-NEW\\implicit_sentiment_laptop.csv"
MODEL_NAME = "gpt-5.1"
API_ENDPOINT = ""  # Để trống nếu dùng OpenAI mặc định
# =======================================
```

* **API_KEY:** Nhập API Key của OpenAI (hoặc provider hỗ trợ chuẩn OpenAI) của bạn.
* **CSV_FILE:** Đường dẫn trỏ tới file dữ liệu cần xử lý (ví dụ: `implicit_sentiment_laptop.csv`). File CSV bắt buộc phải có 3 cột: `text`, `target`, `label`.
* **MODEL_NAME:** Tên model bạn muốn sử dụng để suy luận.
* **API_ENDPOINT:** (Tuỳ chọn) Thay đổi endpoint nếu bạn không dùng server OpenAI mặc định.

## 3. Chạy script

Sau khi cấu hình xong, mở terminal/command prompt và chạy lệnh sau trong thư mục chứa file:

```bash
python main.py
```

## Kết quả đầu ra
Sau khi chạy hoàn thành, script sẽ in ra màn hình các thông tin:
* Các chỉ số đánh giá mô hình: Accuracy, F1-Macro và bảng Classification Report.
* **TIMING METRICS:** Tổng thời gian chạy, thời gian inference, thời gian trung bình.
* **TOKEN USAGE:** Tổng số token (input, output) đã tiêu thụ.
