# E-commerce Review Sentiment Analysis

## 1. Giới thiệu dự án

**E-commerce Review Sentiment Analysis** là hệ thống phân tích cảm xúc đánh giá sản phẩm thương mại điện tử bằng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) và học máy (Machine Learning).

Hệ thống cho phép người dùng nhập nội dung nhận xét về sản phẩm. Sau đó, mô hình đã được huấn luyện sẽ phân tích và phân loại nhận xét thành ba nhóm cảm xúc:

- **Positive:** Tích cực.
- **Negative:** Tiêu cực.
- **Neutral:** Trung lập.

Mục tiêu của dự án là hỗ trợ phân tích ý kiến khách hàng, từ đó giúp người bán hiểu rõ hơn về trải nghiệm và phản hồi của người mua.

## 2. Công nghệ sử dụng

| Công nghệ | Mục đích |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| Pandas | Đọc, xử lý và làm sạch dữ liệu |
| NumPy | Hỗ trợ xử lý dữ liệu |
| Matplotlib | Xây dựng biểu đồ |
| Seaborn | Trực quan hóa dữ liệu |
| Scikit-learn | Xây dựng và đánh giá mô hình Machine Learning |
| TF-IDF | Chuyển đổi văn bản thành vector đặc trưng |
| Naive Bayes | Mô hình phân loại cảm xúc |
| Logistic Regression | Mô hình phân loại cảm xúc |
| Streamlit | Xây dựng giao diện ứng dụng |
| Joblib | Lưu và tải mô hình đã huấn luyện |

## 3. Dữ liệu sử dụng

Dự án sử dụng tập dữ liệu `ViSFD.csv`, gồm 11.122 nhận xét sản phẩm.

Các thuộc tính chính:

- `comment`: Nội dung nhận xét.
- `n_star`: Số sao đánh giá.
- `type`: Nhóm dữ liệu train, dev hoặc test.
- `date_time`: Thời gian đánh giá.
- `label`: Nhãn khía cạnh và cảm xúc của nhận xét.

Nhãn cảm xúc được chuẩn bị từ số sao đánh giá:

- 1–2 sao: Negative.
- 3 sao: Neutral.
- 4–5 sao: Positive.

Đây là cách tạo nhãn dựa trên số sao, không phải kết quả gán nhãn thủ công độc lập cho từng nhận xét.

## 4. Các chức năng chính

### 4.1. Tiền xử lý dữ liệu

- Đọc dữ liệu bằng Pandas.
- Kiểm tra dữ liệu thiếu và dữ liệu trùng lặp.
- Chuẩn hóa nội dung nhận xét.
- Chuyển đổi số sao thành nhãn cảm xúc.
- Lưu dữ liệu sau xử lý vào `data/processed/reviews_cleaned.csv`.

### 4.2. Trực quan hóa dữ liệu

Hệ thống tạo các biểu đồ:

- Phân phối số sao đánh giá.
- Độ dài nhận xét theo nhóm cảm xúc.
- 15 từ khóa phổ biến trong nhận xét tích cực.
- 15 từ khóa phổ biến trong nhận xét tiêu cực.

### 4.3. Huấn luyện mô hình

Dự án sử dụng TF-IDF để chuyển văn bản thành vector đặc trưng và huấn luyện hai mô hình:

- Naive Bayes.
- Logistic Regression.

Các mô hình được lưu trong thư mục `models`.

### 4.4. Đánh giá mô hình

Sử dụng các chỉ số:

- Accuracy.
- Precision.
- Recall.
- Macro F1-score.
- Weighted F1-score.
- Confusion Matrix.

### 4.5. Ứng dụng phân tích cảm xúc

Giao diện Streamlit cho phép người dùng:

1. Nhập nội dung nhận xét sản phẩm.
2. Nhấn nút phân tích cảm xúc.
3. Nhận kết quả dự đoán.
4. Xem độ tin cậy dự đoán và xác suất của từng nhóm cảm xúc.

## 5. Kết quả thực nghiệm

Kết quả đánh giá trên tập test gồm 2.224 nhận xét:

| Mô hình | Accuracy | Macro F1-score |
|---|---:|---:|
| Naive Bayes | 81,21% | 57,30% |
| Logistic Regression | 82,28% | 72,76% |

Logistic Regression đạt kết quả tốt hơn Naive Bayes theo cả Accuracy và Macro F1-score trên tập test.

Đặc biệt, Naive Bayes chưa phân loại chính xác được nhóm Neutral trong kết quả thử nghiệm. Logistic Regression thể hiện khả năng phân loại cân bằng hơn giữa ba nhóm cảm xúc.

## 6. Cấu trúc thư mục

```text
Ecommerce_Sentiment_Analysis/
│
├── data/
│   ├── raw/
│   │   └── ViSFD.csv
│   └── processed/
│       └── reviews_cleaned.csv
│
├── models/
│   ├── naive_bayes.pkl
│   └── logistic_regression.pkl
│
├── notebooks/
│   ├── 02_data_preprocessing.py
│   ├── 03_data_visualization.py
│   ├── 04_model_training.py
│   └── 05_model_evaluation.py
│
├── outputs/
│   ├── figures/
│   ├── classification_report_test.csv
│   ├── model_comparison_dev.csv
│   └── model_comparison_test.csv
│
├── report/
│
├── app.py
├── check_dataset.py
├── requirements.txt
└── README.md
```

## 7. Hướng dẫn cài đặt và chạy chương trình

### Bước 1: Chuẩn bị môi trường

Cài đặt Python và mở thư mục dự án bằng Visual Studio Code.

### Bước 2: Tạo môi trường ảo

```powershell
python -m venv .venv
```

### Bước 3: Kích hoạt môi trường ảo trên Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Bước 4: Cài đặt thư viện

```powershell
pip install -r requirements.txt
```

### Bước 5: Chạy ứng dụng

```powershell
streamlit run app.py
```

Sau khi chạy, mở địa chỉ Local URL được hiển thị trong Terminal, thường là:

`http://localhost:8501`

## 8. Kết luận

Dự án đã xây dựng được quy trình phân tích cảm xúc nhận xét sản phẩm thương mại điện tử, từ bước tiền xử lý dữ liệu, trực quan hóa, huấn luyện mô hình đến triển khai ứng dụng dự đoán.

Qua thực nghiệm, Logistic Regression cho kết quả tốt hơn