
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re

# ==========================================
# 1. DOC DU LIEU DA LAM SACH
# ==========================================

df = pd.read_csv("data/processed/reviews_cleaned.csv")

print("===== THONG TIN DU LIEU =====")
print("Tong so danh gia:", len(df))

# Tao thu muc luu bieu do
output_folder = Path("outputs/figures")
output_folder.mkdir(parents=True, exist_ok=True)

# Cai dat giao dien bieu do
sns.set_theme(style="whitegrid")

# ==========================================
# 2. BIEU DO PHAN BO SO SAO
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="n_star",
    hue="n_star",
    palette="viridis",
    legend=False
)

plt.title("Distribution of Product Ratings", fontsize=14)
plt.xlabel("Number of Stars")
plt.ylabel("Number of Reviews")

plt.tight_layout()
plt.savefig(
    output_folder / "rating_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Da tao bieu do 1: rating_distribution.png")

# ==========================================
# 3. DO DAI NOI DUNG THEO CAM XUC
# ==========================================

# Dem so tu trong moi danh gia
df["review_length"] = df["comment"].apply(
    lambda text: len(str(text).split())
)

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="sentiment",
    y="review_length",
    hue="sentiment",
    palette="Set2",
    legend=False
)

plt.title("Review Length by Sentiment", fontsize=14)
plt.xlabel("Sentiment")
plt.ylabel("Number of Words")

plt.tight_layout()
plt.savefig(
    output_folder / "review_length_by_sentiment.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Da tao bieu do 2: review_length_by_sentiment.png")

# ==========================================
# 4. THONG KE TU KHOA
# ==========================================

# Danh sach tu dung thong dung
stopwords = {
    "là", "và", "của", "có", "cho", "được",
    "mình", "tôi", "bạn", "thì", "mà", "này",
    "đó", "như", "khi", "đã", "đang", "sẽ",
    "rất", "quá", "cũng", "với", "không",
    "một", "những", "các", "trong", "ra",
    "về", "theo", "nên", "vì", "lại", "còn",
    "thì", "nhưng", "nếu", "để", "tại", "vào",
    "em", "anh", "chị", "ơi", "ạ", "nhé",
    "nha", "thôi", "đây", "kia", "ấy"
}


def get_keywords(texts):
    words = []

    for text in texts:
        # Chuyen van ban thanh chu thuong
        text = str(text).lower()

        # Tach tu bang khoang trang va loai ky tu dac biet
        tokens = re.findall(r"\b\w+\b", text)

        # Loai bo stopwords va tu qua ngan
        tokens = [
            word for word in tokens
            if word not in stopwords and len(word) > 1
        ]

        words.extend(tokens)

    return Counter(words).most_common(15)


# ==========================================
# 5. BIEU DO TU KHOA TICH CUC
# ==========================================

positive_texts = df.loc[
    df["sentiment"] == "Positive", "comment"
]

positive_keywords = get_keywords(positive_texts)

positive_df = pd.DataFrame(
    positive_keywords,
    columns=["keyword", "frequency"]
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=positive_df,
    x="frequency",
    y="keyword",
    hue="keyword",
    palette="Greens_r",
    legend=False
)

plt.title("Top 15 Keywords in Positive Reviews", fontsize=14)
plt.xlabel("Frequency")
plt.ylabel("Keyword")

plt.tight_layout()
plt.savefig(
    output_folder / "positive_keywords.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Da tao bieu do 3: positive_keywords.png")

# ==========================================
# 6. BIEU DO TU KHOA TIEU CUC
# ==========================================

negative_texts = df.loc[
    df["sentiment"] == "Negative", "comment"
]

negative_keywords = get_keywords(negative_texts)

negative_df = pd.DataFrame(
    negative_keywords,
    columns=["keyword", "frequency"]
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=negative_df,
    x="frequency",
    y="keyword",
    hue="keyword",
    palette="Reds_r",
    legend=False
)

plt.title("Top 15 Keywords in Negative Reviews", fontsize=14)
plt.xlabel("Frequency")
plt.ylabel("Keyword")

plt.tight_layout()
plt.savefig(
    output_folder / "negative_keywords.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Da tao bieu do 4: negative_keywords.png")

# ==========================================
# 7. IN KET QUA THONG KE
# ==========================================

print("\n===== THONG KE SO SAO =====")
print(df["n_star"].value_counts().sort_index())

print("\n===== THONG KE CAM XUC =====")
print(df["sentiment"].value_counts())

print("\n===== 15 TU KHOA TICH CUC =====")
print(positive_df.to_string(index=False))

print("\n===== 15 TU KHOA TIEU CUC =====")
print(negative_df.to_string(index=False))

# ==========================================
# 8. KIEM TRA CAC FILE BIEU DO
# ==========================================

print("\n===== KIEM TRA FILE BIEU DO =====")

for file in output_folder.glob("*.png"):
    print(file.name, "-", "Da ton tai" if file.exists() else "Chua ton tai")

print("\nHOAN THANH TRUC QUAN HOA DU LIEU!")
print("Thu muc luu bieu do:", output_folder)
