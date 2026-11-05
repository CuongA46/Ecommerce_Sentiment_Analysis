import pandas as pd
import numpy as np
import re
from pathlib import Path

# ==========================================
# 1. DOC DU LIEU GOC
# ==========================================

df = pd.read_csv("data/raw/ViSFD.csv")

print("===== DU LIEU BAN DAU =====")
print("So dong ban dau:", len(df))

# ==========================================
# 2. KIEM TRA VA XU LY GIA TRI THIEU
# ==========================================

print("\n===== GIA TRI THIEU TRUOC KHI XU LY =====")
print(df.isnull().sum())

# Loai bo dong thieu noi dung comment hoac label
df = df.dropna(subset=["comment", "label"])

# ==========================================
# 3. LAM SACH NOI DUNG COMMENT
# ==========================================

def clean_text(text):
    # Chuyen noi dung thanh chuoi va viet thuong
    text = str(text).lower()

    # Loai bo URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Loai bo cac ky tu HTML neu co
    text = re.sub(r"<[^>]+>", " ", text)

    # Chuan hoa khoang trang
    text = re.sub(r"\s+", " ", text)

    # Loai bo khoang trang dau va cuoi
    text = text.strip()

    return text


df["comment"] = df["comment"].apply(clean_text)

# Loai bo cac dong co comment rong
df = df[df["comment"] != ""]

# ==========================================
# 4. TAO NHAN CAM XUC TONG THE
# TU COT LABEL
# ==========================================

# Dem so nhan Positive va Negative trong moi dong
positive_count = df["label"].str.count("#Positive")
negative_count = df["label"].str.count("#Negative")

# Xac dinh nhan cam xuc tong the
conditions = [
    positive_count > negative_count,
    negative_count > positive_count
]

choices = [
    "Positive",
    "Negative"
]

df["sentiment"] = np.select(
    conditions,
    choices,
    default="Neutral"
)

# ==========================================
# 5. CHON CAC COT CAN THIET
# ==========================================

df_cleaned = df[
    [
        "comment",
        "n_star",
        "sentiment",
        "type",
        "date_time",
        "label"
    ]
].copy()

# ==========================================
# 6. TAO THU MUC VA LUU FILE
# ==========================================

output_folder = Path("data/processed")

# Neu thu muc chua ton tai thi tu dong tao
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "reviews_cleaned.csv"

df_cleaned.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 7. KIEM TRA KET QUA SAU KHI LAM SACH
# ==========================================

print("\n===== KET QUA SAU KHI LAM SACH =====")

print("So dong con lai:", len(df_cleaned))

print("\nThong ke nhan cam xuc:")
print(df_cleaned["sentiment"].value_counts())

print("\nThong ke nhom du lieu:")
print(df_cleaned["type"].value_counts())

print("\nSo gia tri thieu:")
print(df_cleaned.isnull().sum())

print("\nSo dong trung lap:")
print(df_cleaned.duplicated().sum())

print("\n10 dong du lieu sau khi lam sach:")
print(df_cleaned.head(10))

print("\nDa luu du lieu tai:", output_file)