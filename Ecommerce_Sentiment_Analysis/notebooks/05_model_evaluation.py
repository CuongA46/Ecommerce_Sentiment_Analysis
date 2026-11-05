
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. DOC DU LIEU TEST
# ==========================================

df = pd.read_csv("data/processed/reviews_cleaned.csv")

test_df = df[df["type"] == "test"].copy()
test_df = test_df.dropna(subset=["comment", "sentiment"])

X_test = test_df["comment"].astype(str)
y_test = test_df["sentiment"]

labels = ["Negative", "Neutral", "Positive"]

print("===== THONG TIN TAP TEST =====")
print("So dong test:", len(test_df))
print("\nPhan bo nhan cam xuc:")
print(y_test.value_counts())

# ==========================================
# 2. TAO THU MUC LUU KET QUA
# ==========================================

output_folder = Path("outputs")
figure_folder = output_folder / "figures"

output_folder.mkdir(parents=True, exist_ok=True)
figure_folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================
# 3. DOC CAC MO HINH DA HUAN LUYEN
# ==========================================

model_files = {
    "Naive Bayes": "models/naive_bayes.pkl",
    "Logistic Regression": "models/logistic_regression.pkl"
}

results = []
detailed_reports = []

# ==========================================
# 4. DANH GIA TUNG MO HINH
# ==========================================

for model_name, model_path in model_files.items():

    print("\n===================================")
    print("DANH GIA:", model_name)
    print("===================================")

    # Tai mo hinh va TF-IDF
    saved_data = joblib.load(model_path)

    vectorizer = saved_data["vectorizer"]
    model = saved_data["model"]

    # Chuyen du lieu test thanh vector
    X_test_tfidf = vectorizer.transform(X_test)

    # Du doan cam xuc
    y_pred = model.predict(X_test_tfidf)

    # Tinh cac chi so
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("Macro Precision:", round(precision, 4))
    print("Macro Recall:", round(recall, 4))
    print("Macro F1-score:", round(macro_f1, 4))
    print("Weighted F1-score:", round(weighted_f1, 4))

    # Bao cao chi tiet tung lop
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    print("\nCLASSIFICATION REPORT:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            zero_division=0
        )
    )

    for label in labels:
        detailed_reports.append({
            "Model": model_name,
            "Sentiment": label,
            "Precision": report[label]["precision"],
            "Recall": report[label]["recall"],
            "F1-score": report[label]["f1-score"],
            "Support": report[label]["support"]
        })

    # ======================================
    # 5. CONFUSION MATRIX
    # ======================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    print("CONFUSION MATRIX:")
    print(cm)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")

    plt.tight_layout()

    figure_name = (
        "confusion_matrix_naive_bayes.png"
        if model_name == "Naive Bayes"
        else "confusion_matrix_logistic_regression.png"
    )

    plt.savefig(
        figure_folder / figure_name,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Da luu:", figure_folder / figure_name)

    # ======================================
    # 6. LUU KET QUA TONG HOP
    # ======================================

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1
    })


# ==========================================
# 7. SO SANH CAC MO HINH
# ==========================================

results_df = pd.DataFrame(results)

print("\n===================================")
print("BANG SO SANH KET QUA TREN TEST")
print("===================================")

print(results_df.to_string(index=False))

results_df.to_csv(
    output_folder / "model_comparison_test.csv",
    index=False,
    encoding="utf-8-sig"
)

# Luu bao cao chi tiet theo tung nhan
report_df = pd.DataFrame(detailed_reports)

report_df.to_csv(
    output_folder / "classification_report_test.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nDa luu bang so sanh:")
print("outputs/model_comparison_test.csv")

print("\nDa luu bao cao chi tiet:")
print("outputs/classification_report_test.csv")

# ==========================================
# 8. KET THUC
# ==========================================

print("\nHOAN THANH DANH GIA MO HINH TREN TAP TEST!")
