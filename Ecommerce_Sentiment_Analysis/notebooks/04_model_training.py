
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. DOC DU LIEU
# ==========================================

df = pd.read_csv("data/processed/reviews_cleaned.csv")

print("===== THONG TIN DATASET =====")
print("Tong so dong:", len(df))
print("Cac cot:", df.columns.tolist())

# Kiem tra cac cot can thiet
required_columns = ["comment", "sentiment", "type"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Thieu cot du lieu: {column}")

# ==========================================
# 2. CHIA DU LIEU TRAIN, DEV, TEST
# ==========================================

train_df = df[df["type"] == "train"].copy()
dev_df = df[df["type"] == "dev"].copy()
test_df = df[df["type"] == "test"].copy()

print("\n===== SO LUONG DU LIEU =====")
print("Train:", len(train_df))
print("Dev:", len(dev_df))
print("Test:", len(test_df))

# Loai bo cac dong khong co noi dung hoac nhan
train_df = train_df.dropna(subset=["comment", "sentiment"])
dev_df = dev_df.dropna(subset=["comment", "sentiment"])
test_df = test_df.dropna(subset=["comment", "sentiment"])

# ==========================================
# 3. TAO TAP DU LIEU DAU VAO
# ==========================================

X_train = train_df["comment"].astype(str)
y_train = train_df["sentiment"]

X_dev = dev_df["comment"].astype(str)
y_dev = dev_df["sentiment"]

# Thu tu nhan co dinh
labels = ["Negative", "Neutral", "Positive"]

# Kiem tra cac nhan hop le
if not set(y_train.unique()).issubset(set(labels)):
    raise ValueError("Tap train co nhan cam xuc khong hop le")

if not set(y_dev.unique()).issubset(set(labels)):
    raise ValueError("Tap dev co nhan cam xuc khong hop le")

# ==========================================
# 4. CHUYEN VAN BAN THANH VECTOR TF-IDF
# ==========================================

print("\n===== BAT DAU TF-IDF =====")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

# Chi fit tren tap train de tranh ro ri du lieu
X_train_tfidf = vectorizer.fit_transform(X_train)

# Dev chi duoc transform, khong fit lai
X_dev_tfidf = vectorizer.transform(X_dev)

print("Kich thuoc ma tran train:", X_train_tfidf.shape)
print("Kich thuoc ma tran dev:", X_dev_tfidf.shape)
print("So dac trung TF-IDF:", len(vectorizer.get_feature_names_out()))

# ==========================================
# 5. KHOI TAO HAI MO HINH
# ==========================================

models = {
    "Naive Bayes": MultinomialNB(alpha=0.5),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
}

# Tao thu muc luu mo hinh
model_folder = Path("models")
model_folder.mkdir(parents=True, exist_ok=True)

# ==========================================
# 6. HUAN LUYEN VA DANH GIA TREN DEV
# ==========================================

results = []

for model_name, model in models.items():

    print("\n===================================")
    print("MO HINH:", model_name)
    print("===================================")

    # Huan luyen
    model.fit(X_train_tfidf, y_train)

    # Du doan tren tap dev
    y_pred = model.predict(X_dev_tfidf)

    # Tinh cac chi so
    accuracy = accuracy_score(y_dev, y_pred)

    macro_f1 = f1_score(
        y_dev,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1-score:", round(macro_f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_dev,
            y_pred,
            labels=labels,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_dev,
            y_pred,
            labels=labels
        )
    )

    # Luu ket qua
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro_F1": macro_f1
    })

    # Luu mo hinh kem TF-IDF
    model_filename = (
        "naive_bayes.pkl"
        if model_name == "Naive Bayes"
        else "logistic_regression.pkl"
    )

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
            "labels": labels
        },
        model_folder / model_filename
    )

    print("Da luu mo hinh:", model_folder / model_filename)

# ==========================================
# 7. SO SANH HAI MO HINH
# ==========================================

results_df = pd.DataFrame(results)

print("\n===== BANG SO SANH MO HINH TREN DEV =====")
print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/model_comparison_dev.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nDa luu bang so sanh tai outputs/model_comparison_dev.csv")

# ==========================================
# 8. KET THUC
# ==========================================

print("\nHOAN THANH HUAN LUYEN HAI MO HINH!")
