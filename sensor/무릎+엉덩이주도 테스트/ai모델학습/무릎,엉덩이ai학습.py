# train_rep_classifier.py
import json, joblib, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)

# ===== 0) 데이터 로드 =====
DATA = Path("rep_features_labeled.csv")   # ← 라벨링된 CSV 파일
df = pd.read_csv(DATA, encoding="utf-8-sig")

# BOM/공백 제거
df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

# ===== 1) 피처 & 라벨 설정 =====
FEATURES = [
    "tempo_desc_ms", "tempo_asc_ms", "rom_deg",
    "env_int_desc", "env_int_asc",
    "env_peak_desc", "env_peak_asc",
    "env_init_mean",
]
TARGET = "strategy_label"   # CSV에 실제 존재하는 라벨 컬럼명

# 컬럼 존재 확인
missing = [c for c in [TARGET] + FEATURES if c not in df.columns]
if missing:
    raise KeyError(f"누락된 컬럼: {missing}\n현재 컬럼: {df.columns.tolist()}")

X = df[FEATURES].copy()
y = df[TARGET].astype(str).copy()
print("[INFO] 클래스 분포:", y.value_counts().to_dict())

# ===== 2) 전처리 파이프라인 =====
num_prep = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
preprocess = ColumnTransformer(
    transformers=[("num", num_prep, FEATURES)],
    remainder="drop"
)

# ===== 3) 모델 후보 =====
candidates = {
    "logreg": Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            C=1.0, max_iter=2000, random_state=42))
    ]),
    "svm_linear": Pipeline([
        ("prep", preprocess),
        ("clf", SVC(kernel="linear", C=1.0, probability=True, random_state=42))
    ]),
    "rf": Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300, random_state=42))
    ]),
}

# ===== 4) LOOCV (Leave-One-Out Cross Validation) =====
loo = LeaveOneOut()
cv_rows = []
best_name, best_score, best_model = None, -np.inf, None

for name, pipe in candidates.items():
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)[0]
        y_true.append(y_te)
        y_pred.append(pred)

    f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cv_rows.append({"model": name, "macro_f1": f1, "balanced_acc": bal_acc})

    if f1 > best_score:
        best_score = f1
        best_name = name
        best_model = pipe.fit(X, y)

cv_df = pd.DataFrame(cv_rows).sort_values("macro_f1", ascending=False)
print("=== LOOCV 결과 ===")
print(cv_df.to_string(index=False))
print("\nBest:", best_name)

# ===== 5) 학습 데이터 기준 리포트 (참고용) =====
train_pred = best_model.predict(X)
print("\n=== Train Classification Report ===")
print(classification_report(y, train_pred, digits=4))

# ===== 6) Permutation Importance =====
result = permutation_importance(
    best_model, X, y, scoring="f1_macro", n_repeats=50, random_state=42
)
imp_df = pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values("importance_mean", ascending=False)

print("\n=== Permutation Importance ===")
print(imp_df.to_string(index=False))

# ===== 7) 산출물 저장 =====
outdir = Path("artifacts"); outdir.mkdir(exist_ok=True)
joblib.dump(best_model, outdir / "best_model.joblib")
cv_df.to_csv(outdir / "cv_report.csv", index=False)
imp_df.to_csv(outdir / "permutation_importance.csv", index=False)
with open(outdir / "feature_order.json", "w", encoding="utf-8") as f:
    json.dump(FEATURES, f, ensure_ascii=False, indent=2)

print(f"\n[Saved] artifacts/ (모델, 리포트, 중요도, 피처순서)")
