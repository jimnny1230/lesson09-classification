import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="채점표 - 뇌졸중 예측 실습실",
    page_icon="📋",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/jimnny1230/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")

df = load_data()

st.title("📋 모델 채점표 (혼동행렬 및 성능 평가)")
st.markdown("테스트 데이터를 바탕으로 각 모델의 혼동행렬과 평가지표(정확도, 재현율, 정밀도, F1)를 비교하고, 오류 예측 사례를 확인합니다.")
st.divider()

# 속성 정의
feat_map_rev = {
    "나이": "age",
    "평균 혈당": "avg_glucose_level",
    "체질량지수": "bmi",
    "고혈압": "hypertension",
    "심장병": "heart_disease"
}
feat_map = {v: k for k, v in feat_map_rev.items()}
all_korean_feats = list(feat_map_rev.keys())
default_korean_feats = ["나이", "평균 혈당", "고혈압", "심장병"]

st.subheader("1. 모델 입력 속성 선택")
selected_korean_feats = st.multiselect(
    "사용할 속성을 고르세요 (우리말 이름)",
    options=all_korean_feats,
    default=default_korean_feats,
    key="score_features"
)

if len(selected_korean_feats) < 2:
    st.warning("⚠️ 속성을 **두 개 이상** 선택해 주세요.")
    st.stop()

selected_eng_feats = [feat_map_rev[k] for k in selected_korean_feats]

# 데이터 분할 (분류 모델 페이지와 동일한 규칙)
df_sorted = df.sort_values("id").reset_index(drop=True)
test_indices = []
train_indices = []

for chunk_start in range(0, len(df_sorted), 10):
    chunk = df_sorted.iloc[chunk_start:chunk_start+10]
    if len(chunk) == 0:
        break
    test_indices.extend(chunk.index[:3].tolist())
    train_indices.extend(chunk.index[3:].tolist())

df_train = df_sorted.loc[train_indices].copy()
df_test = df_sorted.loc[test_indices].copy()

if "bmi" in selected_eng_feats:
    train_bmi_median = df_train["bmi"].median()
    df_train["bmi"] = df_train["bmi"].fillna(train_bmi_median)
    df_test["bmi"] = df_test["bmi"].fillna(train_bmi_median)

X_train = df_train[selected_eng_feats]
y_train = df_train["stroke"]
X_test = df_test[selected_eng_feats]
y_test = df_test["stroke"]

# 테스트용 사람 수 및 뇌졸중인 사람 수 한 줄 안내
test_total_count = len(y_test)
test_stroke_count = int(y_test.sum())
st.info(f"📌 **테스트 데이터 정보**: 총 **{test_total_count:,}명**, 그중 실제 뇌졸중인 사람 수: **{test_stroke_count:,}명**")
st.divider()

# 모델 학습 및 예측
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

majority_class = y_train.mode()[0]
dummy_pred = np.full_like(y_test, majority_class)

# 혼동행렬 계산 (Labels: [0, 1] -> 0: 아님, 1: 뇌졸중)
# scikit-learn confusion_matrix 순서: 행(실제 0, 1), 열(예측 0, 1)
# [[TN, FP],
#  [FN, TP]]
def get_cm_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp

lr_tn, lr_fp, lr_fn, lr_tp = get_cm_metrics(y_test, lr_pred)
dt_tn, dt_fp, dt_fn, dt_tp = get_cm_metrics(y_test, dt_pred)
dummy_tn, dummy_fp, dummy_fn, dummy_tp = get_cm_metrics(y_test, dummy_pred)

# 지표 계산 함수
def calculate_metrics(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else "계산할 수 없음"
    
    if precision == "계산할 수 없음":
        f1 = "계산할 수 없음"
    else:
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, recall, precision, f1

lr_acc, lr_rec, lr_prec, lr_f1 = calculate_metrics(lr_tn, lr_fp, lr_fn, lr_tp)
dt_acc, dt_rec, dt_prec, dt_f1 = calculate_metrics(dt_tn, dt_fp, dt_fn, dt_tp)
dummy_acc, dummy_rec, dummy_prec, dummy_f1 = calculate_metrics(dummy_tn, dummy_fp, dummy_fn, dummy_tp)

# 2. 혼동행렬 2x2 표 표시
st.subheader("2. 모델별 혼동행렬 (Confusion Matrix)")
st.markdown("행은 **실제 값**, 열은 **예측 값**을 나타냅니다.")

col1, col2 = st.columns(2)

def display_confusion_matrix(title, tn, fp, fn, tp):
    st.markdown(f"#### {title}")
    cm_df = pd.DataFrame(
        {
            "예측: 아님(0)": [f"TN: {tn:,}", f"FN: {fn:,}", f"합계: {tn+fn:,}"],
            "예측: 뇌졸중(1)": [f"FP: {fp:,}", f"TP: {tp:,}", f"합계: {fp+tp:,}"],
            "합계": [f"{tn+fp:,}", f"{fn+tp:,}", f"{tn+fp+fn+tp:,}"]
        },
        index=["실제: 아님(0)", "실제: 뇌졸중(1)", "합계"]
    )
    st.dataframe(cm_df, use_container_width=True)

with col1:
    display_confusion_matrix("로지스틱 회귀(확률로 답하는 모델)", lr_tn, lr_fp, lr_fn, lr_tp)

with col2:
    display_confusion_matrix("의사결정트리(질문으로 답하는 모델)", dt_tn, dt_fp, dt_fn, dt_tp)

st.divider()

# 3. 평가지표 비교 표
st.subheader("3. 성능 평가 지표 비교")
st.markdown("정확도, 재현율, 정밀도, F1 점수를 모델별로 나란히 비교합니다.")

metrics_data = [
    {
        "모델명": "로지스틱 회귀(확률로 답하는 모델)",
        "정확도": f"{lr_acc*100:.2f}%",
        "재현율": f"{lr_rec*100:.2f}%" if isinstance(lr_rec, (int, float)) else lr_rec,
        "정밀도": f"{lr_prec*100:.2f}%" if isinstance(lr_prec, (int, float)) else lr_prec,
        "F1 점수": f"{lr_f1*100:.2f}%" if isinstance(lr_f1, (int, float)) else lr_f1
    },
    {
        "모델명": "의사결정트리(질문으로 답하는 모델)",
        "정확도": f"{dt_acc*100:.2f}%",
        "재현율": f"{dt_rec*100:.2f}%" if isinstance(dt_rec, (int, float)) else dt_rec,
        "정밀도": f"{dt_prec*100:.2f}%" if isinstance(dt_prec, (int, float)) else dt_prec,
        "F1 점수": f"{dt_f1*100:.2f}%" if isinstance(dt_f1, (int, float)) else dt_f1
    },
    {
        "모델명": "훈련용에서 많은 쪽으로만 답하는 모델",
        "정확도": f"{dummy_acc*100:.2f}%",
        "재현율": f"{dummy_rec*100:.2f}%" if isinstance(dummy_rec, (int, float)) else dummy_rec,
        "정밀도": f"{dummy_prec*100:.2f}%" if isinstance(dummy_prec, (int, float)) else dummy_prec,
        "F1 점수": f"{dummy_f1*100:.2f}%" if isinstance(dummy_f1, (int, float)) else dummy_f1
    }
]

st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

st.divider()

# 4. 오류 환자 목록 (놓친 환자 / 헛짚은 사람)
st.subheader("4. 예측 오류 분석 목록")
st.markdown("어느 모델의 목록을 확인할지 선택하고, 나이가 많은 순으로 정렬된 오류 환자 목록을 확인하세요.")

model_choice = st.radio(
    "확인할 모델을 선택하세요:",
    options=["로지스틱 회귀(확률로 답하는 모델)", "의사결정트리(질문으로 답하는 모델)"],
    horizontal=True
)

if "로지스틱" in model_choice:
    selected_preds = lr_pred
    selected_fn_count = lr_fn
    selected_fp_count = lr_fp
else:
    selected_preds = dt_pred
    selected_fn_count = dt_fn
    selected_fp_count = dt_fp

# 데이터프레임에 실제값과 예측값 결합
df_analysis = df_test.copy()
df_analysis["pred"] = selected_preds
df_analysis["actual"] = y_test.values

# 놓친 환자 (실제 1인데 0으로 예측한 경우 -> FN)
missed_patients = df_analysis[(df_analysis["actual"] == 1) & (df_analysis["pred"] == 0)].copy()
missed_patients = missed_patients.sort_values(by="age", ascending=False).reset_index(drop=True)

# 헛짚은 사람 (실제 0인데 1로 예측한 경우 -> FP)
false_alarm_patients = df_analysis[(df_analysis["actual"] == 0) & (df_analysis["pred"] == 1)].copy()
false_alarm_patients = false_alarm_patients.sort_values(by="age", ascending=False).reset_index(drop=True)

col3, col4 = st.columns(2)

def format_error_dataframe(df_subset):
    formatted_df = pd.DataFrame()
    formatted_df["번호"] = range(1, len(df_subset) + 1)
    formatted_df["나이"] = df_subset["age"]
    formatted_df["평균 혈당"] = df_subset["avg_glucose_level"]
    formatted_df["고혈압 여부"] = df_subset["hypertension"].map({1: "있음", 0: "없음"})
    formatted_df["심장병 여부"] = df_subset["heart_disease"].map({1: "있음", 0: "없음"})
    return formatted_df

with col3:
    total_missed = len(missed_patients)
    display_missed = missed_patients.head(20)
    st.markdown(f"#### 🛑 놓친 환자 목록 (실제 뇌졸중인데 아니라고 예측) - 전체 {total_missed:,}명")
    if total_missed > 20:
        st.caption("※ 목록이 20줄을 초과하여 상위 20줄만 표시됩니다.")
    st.dataframe(format_error_dataframe(display_missed), use_container_width=True, hide_index=True)

with col4:
    total_false = len(false_alarm_patients)
    display_false = false_alarm_patients.head(20)
    st.markdown(f"#### 🔍 헛짚은 사람 목록 (실제 아님인데 뇌졸중이라고 예측) - 전체 {total_false:,}명")
    if total_false > 20:
        st.caption("※ 목록이 20줄을 초과하여 상위 20줄만 표시됩니다.")
    st.dataframe(format_error_dataframe(display_false), use_container_width=True, hide_index=True)

st.markdown(f"💡 **확인 완료**: 두 목록의 사람 수(놓친 환자 **{total_missed}명** = 혼동행렬 FN **{selected_fn_count}명**, 헛짚은 사람 **{total_false}명** = 혼동행렬 FP **{selected_fp_count}명**)가 해당 모델 혼동행렬의 FN·FP 칸 수와 정확히 일치합니다.")
