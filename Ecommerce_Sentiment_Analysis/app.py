
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ==========================================
# 1. CAU HINH GIAO DIEN
# ==========================================

st.set_page_config(
    page_title="E-commerce Sentiment Analysis",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ E-commerce Review Sentiment Analysis")
st.write(
    "Hệ thống phân tích cảm xúc đánh giá sản phẩm "
    "bằng Machine Learning."
)

st.divider()

# ==========================================
# 2. TAI MO HINH
# ==========================================

@st.cache_resource
def load_model():
    model_path = Path("models/logistic_regression.pkl")

    if not model_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy models/logistic_regression.pkl"
        )

    saved_data = joblib.load(model_path)

    return (
        saved_data["vectorizer"],
        saved_data["model"],
        saved_data["labels"]
    )


try:
    vectorizer, model, labels = load_model()
    st.success("Mô hình Logistic Regression đã sẵn sàng.")
except Exception as error:
    st.error(f"Lỗi tải mô hình: {error}")
    st.stop()

# ==========================================
# 3. NHAP NOI DUNG DANH GIA
# ==========================================

st.subheader("Nhập nội dung đánh giá")

review = st.text_area(
    "Nội dung nhận xét sản phẩm:",
    placeholder="Ví dụ: Điện thoại sử dụng rất tốt, pin lâu và chụp ảnh đẹp...",
    height=150
)

# ==========================================
# 4. PHAN TICH CAM XUC
# ==========================================

if st.button("Phân tích cảm xúc", type="primary"):

    if not review.strip():
        st.warning("Vui lòng nhập nội dung đánh giá trước khi phân tích.")

    else:
        # Chuyen van ban thanh TF-IDF
        review_vector = vectorizer.transform([review])

        # Du doan nhan cam xuc
        prediction = model.predict(review_vector)[0]

        # Lay xac suat cua tung nhom
        probabilities = model.predict_proba(review_vector)[0]

        # Lay ten cac nhan theo dung thu tu cua mo hinh
        model_classes = model.classes_

        probability_df = pd.DataFrame({
            "Cảm xúc": model_classes,
            "Xác suất": probabilities
        })

        confidence = max(probabilities) * 100

        # Hien thi ket qua
        st.divider()
        st.subheader("Kết quả phân tích")

        if prediction == "Positive":
            st.success("😊 Cảm xúc: TÍCH CỰC")

        elif prediction == "Negative":
            st.error("😞 Cảm xúc: TIÊU CỰC")

        else:
            st.info("😐 Cảm xúc: TRUNG LẬP")

        st.metric(
            label="Mức độ tin cậy của dự đoán",
            value=f"{confidence:.2f}%"
        )

        st.caption(
            "Lưu ý: Đây là xác suất do mô hình ước lượng, "
            "không phải cam kết rằng dự đoán chắc chắn đúng."
        )

        # Bieu do xac suat
        st.subheader("Xác suất của từng nhóm cảm xúc")

        st.bar_chart(
            probability_df.set_index("Cảm xúc")
        )

        # Bang ket qua
        probability_df["Xác suất (%)"] = (
            probability_df["Xác suất"] * 100
        ).round(2)

        st.dataframe(
            probability_df[["Cảm xúc", "Xác suất (%)"]],
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# 5. THONG TIN HE THONG
# ==========================================

st.divider()

st.subheader("Thông tin hệ thống")

st.write("""
- **Bài toán:** Phân loại cảm xúc đánh giá sản phẩm.
- **Mô hình:** Logistic Regression.
- **Phương pháp trích xuất đặc trưng:** TF-IDF.
- **Đầu ra:** Positive, Negative, Neutral.
- **Dữ liệu huấn luyện:** 7.786 đánh giá.
""")

st.caption("E-commerce Sentiment Analysis Project")
