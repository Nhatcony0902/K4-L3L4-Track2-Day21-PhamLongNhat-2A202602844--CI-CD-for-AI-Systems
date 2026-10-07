import os
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

LABELS = ["thu_nhap_thap (0)", "thu_nhap_cao (1)"]


def build_detail_report(
    model_path: str = "models/model.joblib",
    eval_path: str = "data/holdout.csv",
) -> str:
    """
    Bonus 3: tao bao cao chi tiet tren tap holdout gom confusion matrix dang van ban
    va precision / recall / f1 rieng cho tung lop.
    """
    model = joblib.load(model_path)
    df_eval = pd.read_csv(eval_path)
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]
    preds = model.predict(X_eval)

    (tn, fp), (fn, tp) = confusion_matrix(y_eval, preds, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_eval, preds, labels=[0, 1], zero_division=0
    )

    lines = [
        f"Bao cao chi tiet tren holdout ({len(y_eval)} mau)",
        "",
        "Confusion matrix (hang = thuc te, cot = du doan):",
        f"{'':>20}{'du doan 0':>12}{'du doan 1':>12}",
        f"{'thuc te 0':>20}{tn:>12}{fp:>12}",
        f"{'thuc te 1':>20}{fn:>12}{tp:>12}",
        "",
        f"{'lop':<20}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}",
    ]
    for i, name in enumerate(LABELS):
        lines.append(f"{name:<20}{precision[i]:>10.4f}{recall[i]:>10.4f}{f1[i]:>10.4f}{support[i]:>10}")
    lines += [
        "",
        f"Bo sot nguoi thu nhap cao (FN): {fn} / {fn + tp}",
        f"Gan nham nguoi thu nhap thap la cao (FP): {fp} / {fp + tn}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    report = build_detail_report()
    print(report)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/detail.txt", "w") as f:
        f.write(report + "\n")
