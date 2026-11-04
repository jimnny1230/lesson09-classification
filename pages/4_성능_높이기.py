import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="성능 높이기 - 뇌졸중 예측 실습실",
    page_icon="⚖️",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/jimnny1230/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")

df = load_data()

st.title("⚖️ 클래스 불균형 해결 및 성능 높이기")
st.markdown("뇌졸중 데이터는 양성(환자) 비율이 매우 적습니다. 가중치 조정을 통해 모델의 성능 변화를 비교합니다.")
st.divider()

# 속성 정의 및 기본 선택 (age, avg_glucose_level, hypertension, heart_disease)
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
    key="balance_features"
)

if len(selected_korean_feats) < 2:
    st.warning("⚠️ 속성을 **두 개 이상** 선택해 주세요.")
    st.stop()

selected_eng_feats = [feat_map_rev[k] for k in selected_korean_feats]

# 데이터 분할 (채점표 및 분류 모델과 동일한 방식)
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

st.info(f"📌 테스트 데이터 수: **{len(X_test):,}명**, 훈련 데이터 수: **{len(X_train):,}명**")
st.divider()

# 모델 학습 (가중치 조정 전 vs 후)
# 로지스틱 회귀
lr_before = LogisticRegression(random_state=42)
lr_before.fit(X_train, y_train)
lr_pred_before = lr_before.predict(X_test)

lr_after = LogisticRegression(class_weight="balanced", random_state=42)
lr_after.fit(X_train, y_train)
lr_pred_after = lr_after.predict(X_test)

# 의사결정트리
dt_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_before.fit(X_train, y_train)
dt_pred_before = dt_before.predict(X_test)

dt_after = DecisionTreeClassifier(class_depth=None, class_weight="balanced", max_depth=3, min_samples_leaf=5, random_state=42)
dt_after.fit(X_train, y_train)
dt_pred_after = dt_after.predict(X_test)

def get_cm_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp

def calculate_metrics(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else "계산할 수 없음"
    f1 = "계산할 수 없음" if precision == "계산할 수 없음" else ((2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)
    return accuracy, recall, precision, f1

# 지표 계산
lr_b_tn, lr_b_fp, lr_b_fn, lr_b_tp = get_cm_metrics(y_test, lr_pred_before)
lr_a_tn, lr_a_fp, lr_a_fn, lr_a_tp = get_cm_metrics(y_test, lr_pred_after)

dt_b_tn, dt_b_fp, dt_b_fn, dt_b_tp = get_cm_metrics(y_test, dt_pred_before)
dt_a_tn, dt_a_fp, dt_a_fn, dt_a_tp = get_cm_metrics(y_test, dt_pred_after)

lr_b_acc, lr_b_rec, lr_b_prec, lr_b_f1 = calculate_metrics(lr_b_tn, lr_b_fp, lr_b_fn, lr_b_tp)
lr_a_acc, lr_a_rec, lr_a_prec, lr_a_f1 = calculate_metrics(lr_a_tn, lr_a_fp, lr_a_fn, lr_a_tp)

dt_b_acc, dt_b_rec, dt_b_prec, dt_b_f1 = calculate_metrics(dt_b_tn, dt_b_fp, dt_b_fn, dt_b_tp)
dt_a_acc, dt_a_rec, dt_a_prec, dt_a_f1 = calculate_metrics(dt_a_tn, dt_a_fp, dt_a_fn, dt_a_tp)

# 2. 전·후 비교 표 구성
st.subheader("2. 가중치 조정 전·후 모델 성능 비교 표")

def make_comparison_df(b_m, a_m, b_metrics, a_metrics):
    b_tn, b_fp, b_fn, b_tp = b_m
    a_tn, a_fp, a_fn, a_tp = a_m
    b_acc, b_rec, b_prec, b_f1 = b_metrics
    a_acc, a_rec, a_prec, a_f1 = a_metrics

    def fmt_val(v):
        return f"{v*100:.2f}%" if isinstance(v, (int, float)) else v

    data = {
        "평가 항목": ["TN (참 음성)", "FP (거짓 양성)", "FN (거짓 음성)", "TP (진짜 양성)", "정확도", "재현율", "정밀도", "F1 점수"],
        "가중치 조정 전": [f"{b_tn:,}", f"{b_fp:,}", f"{b_fn:,}", f"{b_tp:,}", fmt_val(b_acc), fmt_val(b_rec), fmt_val(b_prec), fmt_val(b_f1)],
        "가중치 조정 후": [f"{a_tn:,}", f"{a_fp:,}", f"{a_fn:,}", f"{a_tp:,}", fmt_val(a_acc), fmt_val(a_rec), fmt_val(a_prec), fmt_val(a_f1)]
    }
    return pd.DataFrame(data)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 로지스틱 회귀(확률로 답하는 모델)")
    lr_comp_df = make_comparison_df((lr_b_tn, lr_b_fp, lr_b_fn, lr_b_tp), (lr_a_tn, lr_a_fp, lr_a_fn, lr_a_tp),
                                    (lr_b_acc, lr_b_rec, lr_b_prec, lr_b_f1), (lr_a_acc, lr_a_rec, lr_a_prec, lr_a_f1))
    st.dataframe(lr_comp_df, use_container_width=True, hide_index=True)
    st.markdown(f"💡 **환자 수 변화 요약**: 찾아낸 환자(TP) **{lr_b_tp}명 ➔ {lr_a_tp}명**, 놓친 환자(FN) **{lr_b_fn}명 ➔ {lr_a_fn}명**, 헛짚은 사람(FP) **{lr_b_fp}명 ➔ {lr_a_fp}명**으로 변경되었습니다.")

with col2:
    st.markdown("#### 의사결정트리(질문으로 답하는 모델)")
    dt_comp_df = make_comparison_df((dt_b_tn, dt_b_fp, dt_b_fn, dt_b_tp), (dt_a_tn, dt_a_fp, dt_a_fn, dt_a_tp),
                                    (dt_b_acc, dt_b_rec, dt_b_prec, dt_b_f1), (dt_a_acc, dt_a_rec, dt_a_prec, dt_a_f1))
    st.dataframe(dt_comp_df, use_container_width=True, hide_index=True)
    st.markdown(f"💡 **환자 수 변화 요약**: 찾아낸 환자(TP) **{dt_b_tp}명 ➔ {dt_a_tp}명**, 놓친 환자(FN) **{dt_b_fn}명 ➔ {dt_a_fn}명**, 헛짚은 사람(FP) **{dt_b_fp}명 ➔ {dt_a_fp}명**으로 변경되었습니다.")

st.divider()

# 3. 산점도 및 결정 경계(2차원 또는 3차원) 시각화
st.subheader("3. 가중치 조정 전·후 0.5 경계선(평면) 겹쳐보기")

if len(selected_korean_feats) == 2:
    # 2차원 산점도 및 경계선
    x_feat_korean, y_feat_korean = selected_korean_feats[0], selected_korean_feats[1]
    x_feat_eng, y_feat_eng = selected_eng_feats[0], selected_eng_feats[1]

    other_features_median = {}
    other_feats_eng = [f for f in selected_eng_feats if f not in [x_feat_eng, y_feat_eng]]
    for feat in other_feats_eng:
        other_features_median[feat] = X_test[feat].median()

    df_test_plot = X_test.copy()
    df_test_plot["stroke"] = y_test.values
    df_test_plot["stroke_label"] = df_test_plot["stroke"].apply(lambda x: "뇌졸중" if x == 1 else "아님")

    fig = px.scatter(
        df_test_plot,
        x=x_feat_eng,
        y=y_feat_eng,
        color="stroke_label",
        color_discrete_map={"뇌졸중": "red", "아님": "blue"},
        labels={x_feat_eng: x_feat_korean, y_feat_eng: y_feat_korean},
        title=f"가중치 조정 전후 결정 경계 비교 ({x_feat_korean} vs {y_feat_korean})"
    )

    x_min, x_max = X_test[x_feat_eng].min(), X_test[x_feat_eng].max()
    y_min, y_max = X_test[y_feat_eng].min(), X_test[y_feat_eng].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    grid_df = pd.DataFrame({x_feat_eng: xx.ravel(), y_feat_eng: yy.ravel()})
    for feat, val in other_features_median.items():
        grid_df[feat] = val
    grid_df = grid_df[selected_eng_feats]

    # 조정 전 확률 및 경계선
    probs_before = lr_before.predict_proba(grid_df)[:, 1].reshape(xx.shape)
    fig.add_trace(
        go.Contour(
            z=probs_before,
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            contours=dict(start=0.5, end=0.5, size=0, coloring='lines'),
            line=dict(color='orange', width=3),
            name="조정 전 경계 (0.5)"
        )
    )

    # 조정 후 확률 및 경계선
    probs_after = lr_after.predict_proba(grid_df)[:, 1].reshape(xx.shape)
    fig.add_trace(
        go.Contour(
            z=probs_after,
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            contours=dict(start=0.5, end=0.5, size=0, coloring='lines'),
            line=dict(color='purple', width=3),
            name="조정 후 경계 (0.5)"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    if (probs_before.min() > 0.5 or probs_before.max() < 0.5) or (probs_after.min() > 0.5 or probs_after.max() < 0.5):
        st.warning("⚠️ 일부 0.5 경계선이 현재 그림의 범위를 벗어나 있습니다.")
    else:
        st.success("✅ 조정 전·후 0.5 경계선이 모두 그림 내에 정상적으로 표시되었습니다.")

elif len(selected_korean_feats) >= 3:
    # 3차원 산점도 및 평면 비교
    x_k, y_k, z_k = selected_korean_feats[0], selected_korean_feats[1], selected_korean_feats[2]
    x_e, y_e, z_e = selected_eng_feats[0], selected_eng_feats[1], selected_eng_feats[2]

    other_features_median = {}
    other_feats_eng = [f for f in selected_eng_feats if f not in [x_e, y_e, z_e]]
    for feat in other_feats_eng:
        other_features_median[feat] = X_test[feat].median()

    df_test_plot = X_test.copy()
    df_test_plot["stroke"] = y_test.values
    df_test_plot["stroke_label"] = df_test_plot["stroke"].apply(lambda x: "뇌졸중" if x == 1 else "아님")

    fig = px.scatter_3d(
        df_test_plot,
        x=x_e,
        y=y_e,
        z=z_e,
        color="stroke_label",
        color_discrete_map={"뇌졸중": "red", "아님": "blue"},
        labels={x_e: x_k, y_e: y_k, z_e: z_k},
        title=f"3차원 산점도 및 결정 평면 비교 ({x_k}, {y_k}, {z_k})"
    )

    # 3차원 그리드 평면 생성
    x_min, x_max = X_test[x_e].min(), X_test[x_e].max()
    y_min, y_max = X_test[y_e].min(), X_test[y_e].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))

    # 로지스틱 회귀 계수 활용하여 z 평면 계산
    # w1*x + w2*y + w3*z + b = 0  =>  z = (-w1*x - w2*y - b) / w3
    coef = lr_after.coef_[0]
    intercept = lr_after.intercept_[0]
    feat_to_idx = {feat: idx for idx, feat in enumerate(selected_eng_feats)}

    idx_x, idx_y, idx_z = feat_to_idx[x_e], feat_to_idx[y_e], feat_to_idx[z_e]
    w_x, w_y, w_z = coef[idx_x], coef[idx_y], coef[idx_z]

    if w_z != 0:
        # 기타 피처의 영향 합산
        other_sum = 0
        for f, val in other_features_median.items():
            other_sum += coef[feat_to_idx[f]] * val

        zz_after = (-w_x * xx - w_y * yy - other_sum - intercept) / w_z

        fig.add_trace(
            go.Surface(
                x=xx, y=yy, z=zz_after,
                colorscale=[[0, 'purple'], [1, 'purple']],
                opacity=0.4,
                showscale=False,
                name="조정 후 평면 (0.5)"
            )
        )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("🖱️ **마우스로 3차원 그래프를 자유롭게 돌려가며** 조정 후 0.5 결정 평면과 데이터 분포를 확인해 보세요.")
