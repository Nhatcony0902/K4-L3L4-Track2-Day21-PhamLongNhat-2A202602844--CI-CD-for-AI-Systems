import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65

# Bonus 5: ty le lop duong tham chieu va muc lech toi da cho phep (5 diem %)
REFERENCE_POSITIVE_RATE = 0.248
DRIFT_TOLERANCE = 0.05

# Bonus 2: dai nguong quyet dinh can quet (0.10 -> 0.90, buoc 0.05)
DECISION_THRESHOLDS = np.round(np.arange(0.10, 0.90 + 1e-9, 0.05), 2)


def check_label_drift(y: pd.Series) -> float:
    """Tinh ty le lop duong va in canh bao neu lech qua DRIFT_TOLERANCE so voi tham chieu."""
    positive_rate = float(y.mean())
    diff = positive_rate - REFERENCE_POSITIVE_RATE
    if abs(diff) > DRIFT_TOLERANCE:
        msg = (
            f"DATA DRIFT: ty le lop duong {positive_rate:.1%} lech {diff:+.1%} "
            f"so voi tham chieu {REFERENCE_POSITIVE_RATE:.1%} (nguong +/-{DRIFT_TOLERANCE:.0%})"
        )
        # Trong GitHub Actions, "::warning::" hien canh bao ngay tren trang tong ket cua run
        prefix = "::warning::" if os.environ.get("GITHUB_ACTIONS") else "WARNING: "
        print(prefix + msg)
    else:
        print(f"Ty le lop duong: {positive_rate:.1%} (trong nguong cho phep)")
    return positive_rate


def sweep_thresholds(y_true: pd.Series, proba: np.ndarray) -> tuple[float, float]:
    """Quet nguong quyet dinh, tra ve (nguong tot nhat, f1 tai nguong do)."""
    scores = [f1_score(y_true, (proba >= t).astype(int)) for t in DECISION_THRESHOLDS]
    best = int(np.argmax(scores))
    return float(DECISION_THRESHOLDS[best]), float(scores[best])


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: kiem tra phan phoi nhan truoc khi huan luyen
    positive_rate = check_label_drift(y_train)

    with mlflow.start_run():

        # Ghi nhan cac sieu tham so
        mlflow.log_params(params)
        mlflow.log_metric("train_positive_rate", positive_rate)

        # Khoi tao va huan luyen GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Du doan tren tap holdout va tinh chi so.
        # f1_score o day tinh cho LOP DUONG (target = 1), khong dung average.
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # Bonus 2: tim nguong quyet dinh toi uu F1 thay cho nguong mac dinh 0.5.
        # f1_score (dung cho quality gate) van tinh o nguong 0.5 vi API dang dung model.predict().
        best_threshold, best_f1 = sweep_thresholds(y_eval, model.predict_proba(X_eval)[:, 1])
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("f1_at_best_threshold", best_f1)
        print(f"Nguong tot nhat: {best_threshold:.2f} -> F1 {best_f1:.4f} (nguong 0.50 -> F1 {f1:.4f})")

        # Luu metrics ra file outputs/report.json (doc boi GitHub Actions o Buoc 2)
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    "best_threshold": best_threshold,
                    "f1_at_best_threshold": best_f1,
                    "train_positive_rate": positive_rate,
                },
                f,
            )

        # Luu mo hinh ra file models/model.joblib (upload len cloud storage o Buoc 2)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
