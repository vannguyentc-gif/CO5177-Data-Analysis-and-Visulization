"""Sinh 2 notebook (Tabular, Time series) bằng nbformat. Chạy: python build_notebooks.py"""
import os
import nbformat as nbf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "notebooks")
os.makedirs(OUT, exist_ok=True)

SETUP = r'''import warnings, sys
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import plotly.express as px, plotly.graph_objects as go, plotly.io as pio
from IPython.display import display, Markdown
if "google.colab" not in sys.modules:
    pio.renderers.default = "notebook_connected"
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 100
pd.set_option("display.max_columns", 60)
RANDOM_STATE = 42
def note(text):
    """Hiển thị caption/kết luận (Markdown) sinh từ số liệu thực tế."""
    display(Markdown(text))
'''


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(c.strip()) if k == "md" else nbf.v4.new_code_cell(c.strip())
        for k, c in cells
    ]
    path = os.path.join(OUT, name)
    nbf.write(nb, path)
    print("wrote", path)


# ---------------------------------------------------------------- TABULAR
T = []
md = lambda s: T.append(("md", s))
code = lambda s: T.append(("code", s))

md(r'''
# Bài 1 – Tabular: Dự đoán xe "lemon" (Kick) trong đấu giá xe cũ
**Môn:** Nền tảng lập trình cho phân tích và trực quan dữ liệu – HK 261 (2026–2027) · **GV:** Lê Thành Sách
**Nhóm:** BugIsFeature · **Sinh viên:** Nguyễn Thị Cẩm Vân · **MSSV:** 2570803
**Hình thức:** cá nhân · **Loại dữ liệu:** Tabular (bắt buộc)

## 1. Mô tả bài toán và tập dữ liệu
- **Nguồn:** OpenML `kick` (cuộc thi Kaggle *Don't Get Kicked!* của Carvana).
- **Bài toán:** phân loại nhị phân `IsBadBuy` – xe mua ở đấu giá có phải "lemon" (lỗi nặng, mua hỏng) hay không.
- **Quy mô:** 72.983 dòng × 33 cột (> 2000 dòng, > 10 cột).
- **Vì sao khó:** (1) **mất cân bằng lớp** (~12% là lemon), (2) **nhiều missing value**, (3) **outlier** ở giá/số km, (4) nhiều cột categorical có **cardinality cao** (Model ~1000 giá trị).
- **Metric chính:** do lớp lệch nên *không* dùng accuracy một mình mà dùng **F1, ROC-AUC, PR-AUC, Balanced Accuracy**.

**Quy trình:** Mô tả → EDA → Tiền xử lý (missing, outlier, encoding, scaling, split) → Mô hình sklearn → So sánh → Kết luận.
''')
code(SETUP + r'''
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score,
                             precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, roc_curve)
from sklearn.inspection import permutation_importance
''')
code(r'''
raw = fetch_openml("kick", as_frame=True, parser="auto").frame.copy()
raw["IsBadBuy"] = pd.to_numeric(raw["IsBadBuy"].astype(str)).astype(int)
raw["PurchDate"] = pd.to_datetime(raw["PurchDate"], unit="s")
# Các cột "mã" (ID) giữ dạng categorical, các cột category còn lại là chuỗi
for c in raw.columns:
    if str(raw[c].dtype) == "category":
        raw[c] = raw[c].astype(object)
print("Kích thước:", raw.shape)
raw.head()
''')
md(r'''
## 2. Khám phá dữ liệu (EDA)
### 2.1 Tổng quan và kiểu dữ liệu
''')
code(r'''
info = pd.DataFrame({
    "dtype": raw.dtypes.astype(str),
    "n_unique": raw.nunique(),
    "missing": raw.isna().sum(),
    "missing_%": (raw.isna().mean() * 100).round(2),
})
display(info)
display(raw.describe().T.round(2))
''')
md(r'''### 2.2 Phân phối biến mục tiêu (mất cân bằng lớp)''')
code(r'''
vc = raw["IsBadBuy"].value_counts().sort_index()
ratio = vc[0] / vc[1]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.countplot(x="IsBadBuy", data=raw, ax=ax[0])
for p in ax[0].patches:
    ax[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width()/2, p.get_height()), ha="center", va="bottom")
ax[0].set_title("Số lượng mỗi lớp")
ax[1].pie(vc, labels=["Tốt (0)", "Lemon (1)"], autopct="%.1f%%", colors=["#4c72b0", "#c44e52"])
ax[1].set_title("Tỉ lệ")
plt.tight_layout(); plt.show()
note(f"**Kết luận:** lớp lemon chỉ chiếm **{vc[1]/vc.sum():.1%}** (tỉ lệ 1:{ratio:.1f}). Một mô hình luôn đoán 'tốt' đã đạt accuracy **{vc[0]/vc.sum():.1%}** nhưng vô dụng -> phải dùng F1/PR-AUC và xử lý imbalance.")
''')
md(r'''### 2.3 Missing values''')
code(r'''
miss = raw.isna().mean().mul(100).sort_values(ascending=False)
miss = miss[miss > 0]
fig = px.bar(miss.reset_index(), x="index", y=0, labels={"index": "Cột", "0": "% missing"},
             title="Tỉ lệ missing theo cột (chỉ các cột có thiếu)")
fig.update_layout(yaxis_title="% missing", xaxis_title="", height=420)
fig.show()
note(f"**Kết luận:** có **{len(miss)}/{raw.shape[1]}** cột bị thiếu, tổng **{int(raw.isna().sum().sum()):,}** ô. "
     f"`PRIMEUNIT` ({miss.get('PRIMEUNIT',0):.1f}%) và `AUCGUART` ({miss.get('AUCGUART',0):.1f}%) gần như trống -> thiếu có thể mang ý nghĩa (xe không được bảo đảm) nên ta tạo cờ `is_missing` thay vì chỉ điền.")
''')
md(r'''### 2.4 Numerical: phân phối và outlier''')
code(r'''
num_show = ["VehOdo", "VehBCost", "WarrantyCost", "MMRCurrentAuctionAveragePrice"]
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for i, c in enumerate(num_show):
    sns.histplot(raw[c].dropna(), bins=50, ax=axes[0, i], color="#4c72b0")
    axes[0, i].set_title(c)
    sns.boxplot(x=raw[c], ax=axes[1, i], color="#55a868")
plt.tight_layout(); plt.show()
q1, q3 = raw[num_show].quantile(.25), raw[num_show].quantile(.75)
iqr = q3 - q1
out_pct = ((raw[num_show] < q1 - 1.5*iqr) | (raw[num_show] > q3 + 1.5*iqr)).mean().mul(100).round(2)
display(out_pct.rename("% outlier theo IQR").to_frame())
note(f"**Kết luận:** `WarrantyCost` có đuôi phải dài (outlier {out_pct['WarrantyCost']}%). Nên dùng **RobustScaler** (median/IQR) thay vì StandardScaler để outlier không kéo lệch thang đo.")
''')
md(r'''### 2.5 Mục tiêu so với các feature (target vs 2–3 feature)''')
code(r'''
d = raw.copy()
d["Odo_bin"] = pd.qcut(d["VehOdo"], 8, duplicates="drop")
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
r1 = d.groupby("VehicleAge")["IsBadBuy"].mean().mul(100)
ax[0].bar(r1.index.astype(str), r1.values, color="#c44e52"); ax[0].set_title("% lemon theo VehicleAge"); ax[0].set_ylabel("% lemon")
r2 = d.groupby("Odo_bin")["IsBadBuy"].mean().mul(100)
ax[1].bar(range(len(r2)), r2.values, color="#dd8452"); ax[1].set_title("% lemon theo nhóm VehOdo (octile)"); ax[1].set_xlabel("octile (thấp -> cao)")
top = d["Make"].value_counts().head(10).index
r3 = d[d["Make"].isin(top)].groupby("Make")["IsBadBuy"].mean().mul(100).sort_values()
ax[2].barh(r3.index, r3.values, color="#8172b3"); ax[2].set_title("% lemon theo Make (top 10)")
for a in ax: a.axhline(d["IsBadBuy"].mean()*100, color="k", ls="--", lw=1)
plt.tight_layout(); plt.show()
note(f"**Kết luận:** xe càng **già** và chạy càng **nhiều km** thì tỉ lệ lemon càng cao: từ **{r1.iloc[0]:.1f}%** (tuổi {r1.index[0]}) lên **{r1.iloc[-1]:.1f}%** (tuổi {r1.index[-1]}); "
     f"octile km thấp nhất **{r2.iloc[0]:.1f}%** vs cao nhất **{r2.iloc[-1]:.1f}%**. Hãng cũng khác biệt (từ {r3.iloc[0]:.1f}% đến {r3.iloc[-1]:.1f}%). Đường nét đứt = mức trung bình toàn bộ.")
''')
md(r'''### 2.6 Tương quan giữa các biến số''')
code(r'''
num_cols_eda = raw.select_dtypes("number").columns
corr = raw[num_cols_eda].corr()
plt.figure(figsize=(11, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
plt.title("Ma trận tương quan (numeric)"); plt.show()
tc = corr["IsBadBuy"].drop("IsBadBuy").sort_values(key=abs, ascending=False).head(5)
mmr = [c for c in corr.columns if c.startswith("MMR")]
note(f"**Kết luận:** 8 cột `MMR*` tương quan gần nhau (|r| lên tới **{corr.loc[mmr, mmr].where(~np.eye(len(mmr), dtype=bool)).max().max():.2f}**) -> đa cộng tuyến, ảnh hưởng mô hình tuyến tính. "
     f"Tương quan tuyến tính với target đều yếu (cao nhất: `{tc.index[0]}` r={tc.iloc[0]:.2f}) -> cần mô hình phi tuyến.")
''')
md(r'''
## 3. Chuẩn bị dữ liệu
1. **Feature engineering:** tách `PurchDate` thành năm/tháng; tạo `age_odo = VehOdo/(VehicleAge+1)`, chênh lệch giá `Acq_vs_Cur = MMRAcquisitionAuctionAveragePrice - MMRCurrentAuctionAveragePrice`, `Cost_vs_MMR = VehBCost / MMRCurrentAuctionAveragePrice`; cờ missing cho `PRIMEUNIT`, `AUCGUART`.
2. **Split 60/20/20 train/val/test, stratify theo target** (split *trước* khi fit bất kỳ bộ biến đổi nào -> tránh data leakage).
3. **Missing:** numeric -> median (fit trên train); categorical -> `"missing"`.
4. **Outlier:** giữ nguyên giá trị nhưng dùng RobustScaler (và cây quyết định vốn bền với outlier).
5. **Encoding:** cardinality ≤ 20 -> One-Hot; cardinality cao (Model, Trim, SubModel, BYRNO, VNZIP1, VNST, Make) -> **Frequency encoding** (fit trên train).
6. **Scaling:** RobustScaler cho numeric.
''')
code(r'''
df = raw.copy()
df["PurchYear"] = df["PurchDate"].dt.year
df["PurchMonth"] = df["PurchDate"].dt.month
df["age_odo"] = df["VehOdo"] / (df["VehicleAge"] + 1)
df["Acq_vs_Cur"] = df["MMRAcquisitionAuctionAveragePrice"] - df["MMRCurrentAuctionAveragePrice"]
df["Cost_vs_MMR"] = df["VehBCost"] / df["MMRCurrentAuctionAveragePrice"].replace(0, np.nan)
df["PRIMEUNIT_missing"] = df["PRIMEUNIT"].isna().astype(int)
df["AUCGUART_missing"] = df["AUCGUART"].isna().astype(int)
df = df.drop(columns=["PurchDate"])

y = df.pop("IsBadBuy")
X_train, X_tmp, y_train, y_tmp = train_test_split(df, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RANDOM_STATE)
print({k: (v.shape[0], round(v.mean(), 4)) for k, v in dict(train=y_train, val=y_val, test=y_test).items()})
''')
code(r'''
cat_all = X_train.select_dtypes(exclude="number").columns.tolist()
low_cat = [c for c in cat_all if X_train[c].nunique() <= 20]
high_cat = [c for c in cat_all if X_train[c].nunique() > 20]
num_cols = X_train.select_dtypes("number").columns.tolist()
print("numeric:", len(num_cols), "| low-card cat:", low_cat, "\nhigh-card cat:", high_cat)

# Frequency encoding: fit trên TRAIN, áp dụng cho val/test
freq_maps = {c: X_train[c].fillna("missing").value_counts(normalize=True) for c in high_cat}
def add_freq(X):
    X = X.copy()
    for c in high_cat:
        X[c + "_freq"] = X[c].fillna("missing").map(freq_maps[c]).fillna(0.0)
    return X.drop(columns=high_cat)
X_train_f, X_val_f, X_test_f = map(add_freq, (X_train, X_val, X_test))
num_cols_f = num_cols + [c + "_freq" for c in high_cat]

def make_pre(scaler="robust", onehot=True):
    sc = RobustScaler() if scaler == "robust" else StandardScaler()
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", sc)])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005, sparse_output=False))])
    return ColumnTransformer([("num", num, num_cols_f), ("cat", cat, low_cat)])
pre = make_pre().fit(X_train_f)
print("Số feature sau tiền xử lý:", pre.transform(X_train_f.head()).shape[1])
''')
md(r'''### 3.1 So sánh phân phối trước/sau scaling''')
code(r'''
cols = ["VehOdo", "WarrantyCost", "VehBCost"]
sub = X_train_f[cols].copy()
sub_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(sub), columns=cols)
rob = pd.DataFrame(RobustScaler().fit_transform(sub_imp), columns=cols)
std = pd.DataFrame(StandardScaler().fit_transform(sub_imp), columns=cols)
fig, axes = plt.subplots(3, 3, figsize=(14, 8))
for j, c in enumerate(cols):
    for i, (nm, dd) in enumerate([("Gốc", sub_imp), ("StandardScaler", std), ("RobustScaler", rob)]):
        sns.histplot(dd[c], bins=50, ax=axes[i, j], color=["#4c72b0", "#dd8452", "#55a868"][i])
        axes[i, j].set_title(f"{c} – {nm}")
plt.tight_layout(); plt.show()
w = sub_imp["WarrantyCost"]
sig_iqr = (w.quantile(.75) - w.quantile(.25)) / 1.349
note(f"**Kết luận:** với `WarrantyCost`, độ lệch chuẩn = **{w.std():.0f}** trong khi 'sigma theo IQR' = **{sig_iqr:.0f}** (tỉ lệ {w.std()/sig_iqr:.2f}). "
     f"Tỉ lệ > 1 nghĩa là outlier làm phồng độ lệch chuẩn nên StandardScaler nén phần lớn dữ liệu vào vùng hẹp hơn; RobustScaler chuẩn hóa theo median/IQR nên không bị outlier chi phối. Cả hai đều giữ nguyên hình dạng phân phối (lệch phải), chỉ đổi thang đo.")
''')
md(r'''
## 4. Mô hình hóa
Huấn luyện 6 cấu hình để so sánh có chủ đích:

| # | Mô hình | Mục đích |
|---|---|---|
| A | Logistic Regression (baseline) | Mốc tham chiếu |
| B | Logistic Regression + `class_weight=balanced` | Tác động của xử lý imbalance |
| C | Random Forest + `class_weight=balanced_subsample` | Mô hình phi tuyến |
| D | HistGradientBoosting (không xử lý imbalance) | Boosting |
| E | HistGradientBoosting + `class_weight=balanced` | Boosting + imbalance |
| F | E + **chỉnh ngưỡng** theo F1 trên validation | Ngưỡng 0.5 không phù hợp lớp hiếm |
''')
code(r'''
def evaluate(name, proba, y_true, thr=0.5):
    pred = (proba >= thr).astype(int)
    return dict(model=name, thr=round(thr, 3), accuracy=(pred == y_true).mean(),
                balanced_acc=balanced_accuracy_score(y_true, pred),
                precision=precision_score(y_true, pred, zero_division=0),
                recall=recall_score(y_true, pred), F1=f1_score(y_true, pred),
                ROC_AUC=roc_auc_score(y_true, proba), PR_AUC=average_precision_score(y_true, proba))

models = {
    "A. LogReg (baseline)": Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))]),
    "B. LogReg + balanced": Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "C. RandomForest + balanced": Pipeline([("pre", make_pre()), ("clf", RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE))]),
    "D. HistGB": Pipeline([("pre", make_pre()), ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE))]),
    "E. HistGB + balanced": Pipeline([("pre", make_pre()), ("clf", HistGradientBoostingClassifier(
        class_weight="balanced", random_state=RANDOM_STATE))]),
}
val_proba, test_proba, rows_val, rows_test = {}, {}, [], []
for n, m in models.items():
    m.fit(X_train_f, y_train)
    val_proba[n] = m.predict_proba(X_val_f)[:, 1]
    test_proba[n] = m.predict_proba(X_test_f)[:, 1]
    rows_val.append(evaluate(n, val_proba[n], y_val))
    rows_test.append(evaluate(n, test_proba[n], y_test))

# F: chọn ngưỡng tối ưu F1 trên VALIDATION, áp dụng lên TEST
n = "E. HistGB + balanced"
pr, rc, th = precision_recall_curve(y_val, val_proba[n])
f1s = 2 * pr * rc / (pr + rc + 1e-9)
best_thr = th[np.argmax(f1s[:-1])]
nF = "F. HistGB + balanced + tuned thr"
rows_val.append(evaluate(nF, val_proba[n], y_val, best_thr))
rows_test.append(evaluate(nF, test_proba[n], y_test, best_thr))
test_proba[nF] = test_proba[n]

val_df = pd.DataFrame(rows_val).set_index("model")
res = pd.DataFrame(rows_test).set_index("model")
print("== Validation =="); display(val_df.round(4))
print("== Test =="); display(res.round(4).style.background_gradient(cmap="Greens", subset=["F1", "ROC_AUC", "PR_AUC"]))
''')
md(r'''### 4.1 So sánh mô hình (baseline vs mô hình mở rộng)''')
code(r'''
m = res[["balanced_acc", "recall", "precision", "F1", "ROC_AUC", "PR_AUC"]].reset_index().melt("model")
fig = px.bar(m, x="variable", y="value", color="model", barmode="group", title="Metric trên tập TEST",
             labels={"variable": "Metric", "value": "Giá trị"}, height=450)
fig.show()
base, best = res.loc["A. LogReg (baseline)"], res.loc[nF]
best_name = res["F1"].idxmax()
note(f"**Kết luận:** (1) Baseline A có accuracy **{base.accuracy:.3f}** nhưng recall chỉ **{base.recall:.3f}** / F1 **{base.F1:.3f}** – bị lớp đa số chi phối. "
     f"(2) `class_weight=balanced` (B) tăng recall lên **{res.loc['B. LogReg + balanced'].recall:.3f}** nhưng precision giảm xuống **{res.loc['B. LogReg + balanced'].precision:.3f}**. "
     f"(3) Boosting D đạt PR-AUC **{res.loc['D. HistGB'].PR_AUC:.3f}** so với LogReg **{base.PR_AUC:.3f}**. "
     f"(4) Chỉnh ngưỡng (F, thr={best_thr:.2f}) cho F1 = **{best.F1:.3f}**. Mô hình F1 cao nhất trên test: **{best_name}** (F1={res['F1'].max():.3f}).")
''')
code(r'''
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for n in ["A. LogReg (baseline)", "C. RandomForest + balanced", "E. HistGB + balanced"]:
    fpr, tpr, _ = roc_curve(y_test, test_proba[n]); ax[0].plot(fpr, tpr, label=f"{n} (AUC={res.loc[n,'ROC_AUC']:.3f})")
    p, r, _ = precision_recall_curve(y_test, test_proba[n]); ax[1].plot(r, p, label=f"{n} (AP={res.loc[n,'PR_AUC']:.3f})")
ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_title("ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend(fontsize=8)
ax[1].axhline(y_test.mean(), color="k", ls="--"); ax[1].set_title("Precision–Recall (nét đứt = tỉ lệ lớp dương)")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
''')
md(r'''### 4.2 Confusion matrix: baseline vs mô hình cuối''')
code(r'''
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, (n, thr) in zip(ax, [("A. LogReg (baseline)", 0.5), (nF, best_thr)]):
    pred = (test_proba[n] >= thr).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Tốt", "Lemon"]).plot(ax=a, cmap="Blues", colorbar=False)
    a.set_title(f"{n}\nthr={thr:.2f}")
plt.tight_layout(); plt.show()
cmA = confusion_matrix(y_test, (test_proba["A. LogReg (baseline)"] >= .5).astype(int))
cmF = confusion_matrix(y_test, (test_proba[nF] >= best_thr).astype(int))
note(f"**Kết luận:** số lemon bắt được tăng từ **{cmA[1,1]}** (baseline) lên **{cmF[1,1]}** (mô hình cuối) trên tổng **{cmF[1].sum()}** lemon thật; "
     f"đổi lại số xe tốt bị cảnh báo nhầm tăng từ **{cmA[0,1]}** lên **{cmF[0,1]}**. Đây là trade-off: chi phí bỏ sót lemon thường cao hơn chi phí kiểm tra thêm xe.")
''')
md(r'''### 4.3 Feature importance (permutation importance, mô hình E trên test)''')
code(r'''
mE = models["E. HistGB + balanced"]
idx = X_test_f.sample(6000, random_state=RANDOM_STATE).index
pi = permutation_importance(mE, X_test_f.loc[idx], y_test.loc[idx], scoring="average_precision",
                            n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X_test_f.columns).sort_values(ascending=False).head(12)
plt.figure(figsize=(8, 5)); sns.barplot(x=imp.values, y=imp.index, color="#4c72b0")
plt.xlabel("Giảm PR-AUC khi xáo trộn feature"); plt.title("Top 12 feature quan trọng"); plt.show()
note(f"**Kết luận:** 3 feature quan trọng nhất là **{', '.join(imp.index[:3])}** (giảm PR-AUC {imp.iloc[0]:.3f}, {imp.iloc[1]:.3f}, {imp.iloc[2]:.3f}). "
     "Lưu ý `WheelTypeID` (thiếu giá trị ở một nhóm xe) là tín hiệu rất mạnh; cần cẩn trọng vì nếu cách thu thập ID này thay đổi khi triển khai thật thì mô hình sẽ lệch.")
''')
md(r'''
## 5. Tổng kết và mở rộng
Phần này được điền tự động từ bảng metric để kết luận luôn khớp số liệu trong report.
''')
code(r'''
best_by_pr = res["PR_AUC"].idxmax()
note(f"""
### Kết quả chính (tập test, {len(y_test):,} mẫu)
| Mô hình | F1 | Recall | PR-AUC | ROC-AUC |
|---|---|---|---|---|
""" + "\n".join(f"| {i} | {r.F1:.3f} | {r.recall:.3f} | {r.PR_AUC:.3f} | {r.ROC_AUC:.3f} |" for i, r in res.iterrows()) + f"""

- **Mô hình tốt nhất theo PR-AUC:** {best_by_pr} ({res.loc[best_by_pr,'PR_AUC']:.3f}); theo F1: {res['F1'].idxmax()} ({res['F1'].max():.3f}).
- So với baseline: F1 **{base.F1:.3f} → {res['F1'].max():.3f}** (gấp {res['F1'].max()/max(base.F1,1e-9):.1f} lần), recall **{base.recall:.3f} → {res.loc[res['F1'].idxmax()].recall:.3f}**.
- **Accuracy gây hiểu lầm:** baseline accuracy {base.accuracy:.3f} cao nhưng F1 thấp, do dữ liệu lệch lớp.

### Hạn chế
- ROC-AUC/PR-AUC vẫn ở mức vừa phải -> dữ liệu có nhiễu, thông tin trong bảng không đủ phân biệt hoàn toàn (không có thông tin kiểm định cơ khí).
- Frequency encoding làm mất thông tin về *danh tính* của Model/Trim; chưa tinh chỉnh siêu tham số (chỉ dùng mặc định).
- Chia ngẫu nhiên, chưa kiểm tra drift theo thời gian (`PurchDate`).

### Hướng mở rộng
- Target encoding có cross-fitting; tối ưu siêu tham số (RandomizedSearchCV); SMOTE / cost-sensitive threshold theo chi phí kinh doanh thật.
- Chia theo thời gian để kiểm tra mô hình có bền khi triển khai; SHAP để giải thích từng dự đoán.
""")
''')
build("01_tabular_kick_lemon.ipynb", T)

# ---------------------------------------------------------------- TIME SERIES
T = []
md = lambda s: T.append(("md", s))
code = lambda s: T.append(("code", s))

md(r'''
# Bài 2 – Time series: Dự báo nồng độ CO₂ khí quyển (Mauna Loa)
**Môn:** Nền tảng lập trình cho phân tích và trực quan dữ liệu – HK 261 (2026–2027) · **GV:** Lê Thành Sách
**Nhóm:** BugIsFeature · **Sinh viên:** Nguyễn Thị Cẩm Vân · **MSSV:** 2570803
**Hình thức:** cá nhân · **Loại dữ liệu:** Time series (tự chọn)

## 1. Mô tả bài toán và tập dữ liệu
- **Nguồn:** `statsmodels.datasets.co2` – đo CO₂ hàng tuần tại Mauna Loa (Scripps), 03/1958 – 12/2001, **2284 điểm**, có **59 giá trị thiếu**.
- **Đặc điểm:** *xu hướng tăng dài hạn* + *mùa vụ hàng năm* (chu kỳ 52 tuần).
- **Bài toán:** dự báo **12 tuần tới** (horizon h=12) – hồi quy.
- **Nguyên tắc:** chia train/val/test **theo thời gian** (không shuffle), mọi biến đổi chỉ dùng thông tin quá khứ để tránh data leakage.
- **Metric:** MAE, RMSE, MAPE.
''')
code(SETUP + r'''
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
H = 12   # horizon (tuần)
''')
code(r'''
co2 = sm.datasets.co2.load_pandas().data["co2"]
print(co2.shape, "| thiếu:", co2.isna().sum(), "| tần suất:", pd.infer_freq(co2.index))
co2.head()
''')
md(r'''## 2. Khám phá dữ liệu (EDA)
### 2.1 Chuỗi gốc và missing''')
code(r'''
miss_idx = co2[co2.isna()].index
fig = go.Figure()
fig.add_trace(go.Scatter(x=co2.index, y=co2.values, name="CO₂ (ppm)", line=dict(width=1.2)))
fig.add_trace(go.Scatter(x=miss_idx, y=[co2.min()]*len(miss_idx), mode="markers", name="Ngày thiếu", marker=dict(color="red", size=4)))
fig.update_layout(title="CO₂ hàng tuần tại Mauna Loa", yaxis_title="ppm", height=420)
fig.show()
gaps = (pd.Series(miss_idx).diff() != pd.Timedelta(weeks=1)).sum()
note(f"**Kết luận:** CO₂ tăng từ **{co2.dropna().iloc[0]:.1f}** lên **{co2.dropna().iloc[-1]:.1f} ppm** (+{co2.dropna().iloc[-1]-co2.dropna().iloc[0]:.1f}) trong 44 năm, kèm dao động mùa vụ đều đặn. "
     f"Có **{len(miss_idx)}** tuần thiếu ({len(miss_idx)/len(co2):.1%}) tạo thành **{gaps}** khoảng trống.")
''')
md(r'''### 2.2 Phân rã xu hướng – mùa vụ – phần dư''')
code(r'''
co2_ff = co2.ffill().bfill()          # chỉ để phân rã/EDA; xử lý chính thức ở mục 3
dec = seasonal_decompose(co2_ff, model="additive", period=52)
fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
for a, (t, s) in zip(ax, [("Quan sát", dec.observed), ("Xu hướng", dec.trend), ("Mùa vụ", dec.seasonal), ("Phần dư", dec.resid)]):
    a.plot(s, lw=1); a.set_ylabel(t)
plt.tight_layout(); plt.show()
amp = dec.seasonal.max() - dec.seasonal.min()
trend_per_year = (dec.trend.dropna().iloc[-1] - dec.trend.dropna().iloc[0]) / (len(dec.trend.dropna()) / 52)
var_share = dec.seasonal.var() / co2_ff.var()
note(f"**Kết luận:** biên độ mùa vụ ≈ **{amp:.1f} ppm**, xu hướng tăng ≈ **{trend_per_year:.2f} ppm/năm**; mùa vụ chỉ giải thích **{var_share:.1%}** phương sai so với xu hướng. "
     "=> Mô hình phải *ngoại suy được xu hướng tăng* thì mới dự báo tốt.")
''')
md(r'''### 2.3 Phát hiện outlier trên phần dư mùa vụ''')
code(r'''
resid = dec.resid.dropna()
q1, q3 = resid.quantile([.25, .75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = resid[(resid < lower) | (resid > upper)]
plt.figure(figsize=(12, 3.8))
plt.plot(resid, color="#4c72b0", lw=.9, label="Residual")
plt.scatter(outliers.index, outliers, color="#c44e52", s=18, label="Outlier (IQR)", zorder=3)
plt.axhline(lower, color="#dd8452", ls="--", lw=1)
plt.axhline(upper, color="#dd8452", ls="--", lw=1)
plt.title("Outlier sau khi loại xu hướng và mùa vụ")
plt.ylabel("Residual (ppm)"); plt.legend(); plt.tight_layout(); plt.show()
note(f"**Kết luận:** phát hiện **{len(outliers)}** điểm bất thường ({len(outliers)/len(resid):.2%}) ngoài khoảng IQR [{lower:.2f}, {upper:.2f}] ppm. "
    "Không xóa các điểm này vì có thể là biến động khí quyển thật; mô hình cây và metric MAE tương đối bền với outlier. Phần missing vẫn được forward-fill ở bước chuẩn bị dữ liệu.")
''')
md(r'''### 2.4 Mùa vụ theo tháng và tự tương quan''')
code(r'''
dev = co2_ff - co2_ff.groupby(co2_ff.index.year).transform("mean")
m = dev.groupby(dev.index.month).mean()
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].bar(m.index, m.values, color=np.where(m.values > 0, "#c44e52", "#4c72b0")); ax[0].set_title("Độ lệch trung bình so với trung bình năm theo tháng"); ax[0].set_xlabel("Tháng")
plot_acf(co2_ff.diff().dropna(), lags=110, ax=ax[1]); ax[1].set_title("ACF của chuỗi sai phân bậc 1")
plt.tight_layout(); plt.show()
note(f"**Kết luận:** CO₂ đạt đỉnh vào tháng **{m.idxmax()}** (+{m.max():.2f} ppm) và đáy vào tháng **{m.idxmin()}** ({m.min():.2f} ppm) – phù hợp chu kỳ quang hợp của thực vật bán cầu Bắc. ACF có đỉnh quanh lag 52 xác nhận chu kỳ năm -> nên đưa **lag 52** và **sin/cos tuần trong năm** vào feature.")
''')
md(r'''
## 3. Chuẩn bị dữ liệu
1. **Missing:** *forward-fill* (chỉ dùng quá khứ, không rò rỉ tương lai; nội suy tuyến tính sẽ dùng giá trị tương lai).
2. **Feature:** lag 1,2,3,4,8,12,26,52; trung bình trượt 4/12/52 tuần (dịch 0, tính trên quá khứ đến thời điểm t); sin/cos tuần trong năm; chỉ số thời gian `t` (xu hướng).
3. **Target:** giá trị tại **t+12**.
4. **Split theo thời gian:** 70% train / 15% val / 15% test (không shuffle). Với horizon h, hàng cuối cùng của mỗi tập được loại để target không chồng sang tập sau.
5. **Scaling:** StandardScaler fit trên train (cho mô hình tuyến tính).
''')
code(r'''
y = co2.ffill().bfill()
feat = pd.DataFrame(index=y.index)
for l in [1, 2, 3, 4, 8, 12, 26, 52]:
    feat[f"lag_{l}"] = y.shift(l - 1)           # lag_1 = giá trị hiện tại (tại t)
for w in [4, 12, 52]:
    feat[f"roll_mean_{w}"] = y.rolling(w).mean()
woy = y.index.isocalendar().week.astype(int).clip(upper=52).values
feat["sin52"] = np.sin(2 * np.pi * woy / 52); feat["cos52"] = np.cos(2 * np.pi * woy / 52)
feat["t"] = np.arange(len(y))
target = y.shift(-H)                            # giá trị 12 tuần sau
data = feat.assign(target=target, current=y).dropna()
n = len(data); i1, i2 = int(n * .70), int(n * .85)
train = data.iloc[:i1 - H]; val = data.iloc[i1:i2 - H]; test = data.iloc[i2:]
print({k: (len(v), str(v.index[0].date()), str(v.index[-1].date())) for k, v in dict(train=train, val=val, test=test).items()})
FEATS = [c for c in data.columns if c not in ("target", "current")]
''')
code(r'''
fig = go.Figure()
for nm, d, col in [("Train", train, "#4c72b0"), ("Validation", val, "#dd8452"), ("Test", test, "#c44e52")]:
    fig.add_trace(go.Scatter(x=d.index, y=d["current"], name=nm, line=dict(color=col)))
fig.update_layout(title="Chia dữ liệu theo thời gian (không shuffle)", yaxis_title="ppm", height=380); fig.show()
note(f"**Kết luận:** test ({test.index[0].year}–{test.index[-1].year}) nằm hoàn toàn *sau* train, nên đánh giá mô phỏng đúng tình huống dự báo thực tế. "
     f"Giá trị CO₂ trên test (trung bình {test['current'].mean():.1f}) cao hơn hẳn train ({train['current'].mean():.1f}) -> mô hình bị thử thách khi **ngoại suy xu hướng**.")
''')
md(r'''
## 4. Mô hình hóa
| Mô hình | Ý tưởng |
|---|---|
| Naive | dự báo = giá trị hiện tại |
| Seasonal naive | = giá trị cùng thời điểm năm trước của mốc dự báo (t+12−52) |
| Ridge (level) | hồi quy tuyến tính dự báo trực tiếp *mức* CO₂ |
| RandomForest (level) | cây quyết định dự báo *mức* – không ngoại suy được |
| HistGB (level) | tương tự với boosting |
| **RandomForest (delta)** | dự báo **mức thay đổi** `y[t+12] − y[t]` rồi cộng lại giá trị hiện tại |
| **HistGB (delta)** | tương tự |
| **Ridge (delta)** | tương tự |
''')
code(r'''
def metrics(name, yt, yp):
    return dict(model=name, MAE=mean_absolute_error(yt, yp),
                RMSE=np.sqrt(mean_squared_error(yt, yp)), MAPE_pct=100 * mean_absolute_percentage_error(yt, yp))

# Chọn alpha của Ridge trên VALIDATION
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
best_a = min([0.01, 0.1, 1, 10, 100],
             key=lambda a: mean_absolute_error(val["target"], make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(train[FEATS], train["target"]).predict(val[FEATS])))
print("alpha Ridge chọn từ validation:", best_a)

def fit_predict(make, delta):
    Xtr, Xte = train[FEATS], test[FEATS]
    ytr = train["target"] - train["current"] if delta else train["target"]
    p = make().fit(Xtr, ytr).predict(Xte)
    return p + test["current"].values if delta else p

makers = {
    "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=best_a)),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE),
    "HistGB": lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}
preds = {"Naive": test["current"].values}
sn = y.shift(52 - H).reindex(test.index)       # giá trị tại (t+12)-52 = đã biết tại thời điểm t
preds["Seasonal naive"] = (sn.fillna(test["current"])).values + 0
# Seasonal naive + drift để công bằng: cộng tăng trưởng 52 tuần gần nhất
preds["Seasonal naive + drift"] = preds["Seasonal naive"] + (test["current"] - y.shift(52).reindex(test.index)).values
for nm, mk in makers.items():
    preds[f"{nm} (level)"] = fit_predict(mk, delta=False)
    preds[f"{nm} (delta)"] = fit_predict(mk, delta=True)
res = pd.DataFrame([metrics(k, test["target"], v) for k, v in preds.items()]).set_index("model").sort_values("MAE")
display(res.round(3).style.background_gradient(cmap="Greens_r", subset=["MAE", "RMSE", "MAPE_pct"]))
''')
md(r'''### 4.1 So sánh mô hình: dự báo *level* và *delta*''')
code(r'''
fig = px.bar(res.reset_index(), x="model", y="MAE", color="MAE", color_continuous_scale="RdYlGn_r",
             title="MAE trên tập test theo mô hình (horizon 12 tuần)", height=430)
fig.update_layout(xaxis_title="", yaxis_title="MAE (ppm)"); fig.show()
b, w = res.index[0], res.index[-1]
rf_l, rf_d = res.loc["RandomForest (level)", "MAE"], res.loc["RandomForest (delta)", "MAE"]
note(f"**Kết luận:** mô hình tốt nhất là **{b}** (MAE **{res.iloc[0].MAE:.3f}** ppm, MAPE {res.iloc[0].MAPE_pct:.3f}%); kém nhất: **{w}** (MAE {res.iloc[-1].MAE:.3f}). "
     f"Cùng một RandomForest, đổi mục tiêu từ *level* sang *delta* làm MAE giảm **{rf_l:.3f} → {rf_d:.3f}** (giảm {(1-rf_d/rf_l):.0%}) – vì cây không thể dự đoán giá trị lớn hơn mọi giá trị đã thấy trong train, còn dự báo *mức thay đổi* thì không bị giới hạn. "
     f"So với Naive (MAE {res.loc['Naive','MAE']:.3f}), mô hình tốt nhất giảm sai số {(1-res.iloc[0].MAE/res.loc['Naive','MAE']):.0%}.")
''')
md(r'''### 4.2 Dự báo so với thực tế (tập test)''')
code(r'''
best_nm = res.index[0]
fig = go.Figure()
fig.add_trace(go.Scatter(x=test.index, y=test["target"], name="Thực tế (t+12)", line=dict(color="black", width=2)))
for nm, col in [(best_nm, "#2ca02c"), ("RandomForest (level)", "#d62728"), ("Naive", "#7f7f7f")]:
    fig.add_trace(go.Scatter(x=test.index, y=preds[nm], name=nm, line=dict(color=col, dash="dot" if nm != best_nm else "solid")))
fig.update_layout(title="Dự báo 12 tuần tới vs thực tế", yaxis_title="ppm", height=440); fig.show()
err_rf = test["target"].values - preds["RandomForest (level)"]
note(f"**Kết luận:** RandomForest (level) dự báo **thấp hơn thực tế một cách hệ thống** (sai số trung bình {err_rf.mean():+.2f} ppm, nghĩa là luôn thấp hơn), "
     f"đường dự báo bị 'phẳng' ở mức tối đa đã thấy trong train ({train['target'].max():.1f} ppm). Mô hình **{best_nm}** bám sát đường thực tế cả xu hướng lẫn mùa vụ (sai số trung bình {(test['target'].values - preds[best_nm]).mean():+.3f} ppm).")
''')
code(r'''
# Phân tích sai số theo thời gian của mô hình tốt nhất
e = pd.Series(test["target"].values - preds[best_nm], index=test.index)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
ax[0].plot(e, lw=1); ax[0].axhline(0, color="k", lw=.8); ax[0].set_title(f"Sai số dự báo – {best_nm}")
sns.histplot(e, bins=30, ax=ax[1], kde=True); ax[1].set_title("Phân phối sai số")
plt.tight_layout(); plt.show()
by_season = e.abs().groupby(e.index.month).mean()
note(f"**Kết luận:** sai số khá cân quanh 0 (độ lệch chuẩn {e.std():.3f} ppm). Tháng khó dự báo nhất là **tháng {by_season.idxmax()}** (|sai số| TB {by_season.max():.3f}), "
     f"dễ nhất là **tháng {by_season.idxmin()}** ({by_season.min():.3f}) – thường là các tháng chuyển pha mùa vụ (đỉnh/đáy) nên khó hơn.")
''')
md(r'''### 4.3 Độ ổn định theo horizon (mở rộng)
Kiểm tra mô hình tốt nhất khi horizon dài hơn (4, 12, 26, 52 tuần) so với Naive.''')
code(r'''
def run_horizon(h):
    tg = y.shift(-h)
    d = feat.assign(target=tg, current=y).dropna()
    n = len(d); a, b2 = int(n * .70), int(n * .85)
    tr, te = d.iloc[:a - h], d.iloc[b2:]
    m = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)
    m.fit(tr[FEATS], tr["target"] - tr["current"])
    p = m.predict(te[FEATS]) + te["current"].values
    return dict(horizon_weeks=h, naive_MAE=mean_absolute_error(te["target"], te["current"]),
                RF_delta_MAE=mean_absolute_error(te["target"], p))
hz = pd.DataFrame([run_horizon(h) for h in [4, 12, 26, 52]]).set_index("horizon_weeks")
display(hz.round(3))
fig = px.line(hz.reset_index().melt("horizon_weeks"), x="horizon_weeks", y="value", color="variable", markers=True,
              title="MAE theo horizon: Naive vs RandomForest (delta)", labels={"value": "MAE (ppm)"}); fig.show()
note(f"**Kết luận:** sai số của cả hai tăng theo horizon, nhưng RF (delta) tăng chậm hơn: ở h=52, MAE **{hz.loc[52,'RF_delta_MAE']:.3f}** vs Naive **{hz.loc[52,'naive_MAE']:.3f}** "
     f"(giảm {(1-hz.loc[52,'RF_delta_MAE']/hz.loc[52,'naive_MAE']):.0%}). Mô hình tận dụng được mùa vụ nên lợi thế so với Naive lớn dần khi dự báo xa.")
''')
md(r'''
## 5. Tổng kết và mở rộng
''')
code(r'''
tab = "\n".join(f"| {i} | {r.MAE:.3f} | {r.RMSE:.3f} | {r.MAPE_pct:.3f} |" for i, r in res.iterrows())
note(f"""
### Kết quả (test, horizon {H} tuần, {len(test)} mẫu)
| Mô hình | MAE | RMSE | MAPE % |
|---|---|---|---|
{tab}

- **Mô hình tốt nhất:** {res.index[0]} – MAE {res.iloc[0].MAE:.3f} ppm, so với Naive {res.loc['Naive','MAE']:.3f}.
- **Bài học chính:** với chuỗi có xu hướng, mô hình cây dự báo *mức* sẽ thất bại khi ngoại suy (MAE {rf_l:.3f}); dự báo *mức thay đổi* (delta) giải quyết được (MAE {rf_d:.3f}).
- Split theo thời gian + forward-fill bảo đảm không rò rỉ tương lai.

### Hạn chế
- Chỉ một chuỗi đơn biến; chưa dùng mô hình thống kê chuyên dụng (SARIMA, ETS, Prophet) để đối chiếu.
- Một lần chia train/val/test; chưa dùng walk-forward CV đầy đủ. Siêu tham số cây để mặc định.
- Dữ liệu dừng ở 2001 – chưa phản ánh giai đoạn tăng tốc sau đó.

### Hướng mở rộng
- So sánh với SARIMA/ETS; walk-forward `TimeSeriesSplit`; khoảng tin cậy dự báo (quantile regression).
- Thêm biến ngoại sinh (ENSO, phát thải) và áp dụng cho chuỗi cập nhật mới nhất của NOAA.
""")
''')
build("02_timeseries_co2.ipynb", T)
