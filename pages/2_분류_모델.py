import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

# 페이지 설정
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/jimnny1230/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")

df = load_data()

st.title("🤖 뇌졸중 분류 모델 실습")
st.markdown("선택한 속성을 기반으로 로지스틱 회귀와 의사결정트리 모델을 학습하고 평가합니다.")
st.divider()

# 열 이름 매핑 사전 (영문 -> 우리말)
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

# 1. 입력 속성 선택 UI
st.subheader("1. 모델 입력 속성 선택")
selected_korean_feats = st.multiselect(
    "사용할 속성을 고르세요 (우리말 이름)",
    options=all_korean_feats,
    default=default_korean_feats
)

# 두 개보다 적게 고르면 안내문 표시 후 중단
if len(selected_korean_feats) < 2:
    st.warning("⚠️ 속성을 **두 개 이상** 선택해 주세요.")
    st.stop()

selected_eng_feats = [feat_map_rev[k] for k in selected_korean_feats]

# 2. 데이터 전처리 및 분할 (요청사항 반영)
# - 번호 순(여기서는 인덱스 또는 id 순) 정렬 후 열 명씩 묶어 앞 세 명 테스트용 고정, 나머지 일곱 명 훈련용
# 데이터프레임을 id 기준 정렬
df_sorted = df.sort_values("id").reset_index(drop=True)

test_indices = []
train_indices = []

# 열 명씩 묶어서 앞 3명 테스트용, 나머지 7명 훈련용
for chunk_start in range(0, len(df_sorted), 10):
    chunk = df_sorted.iloc[chunk_start:chunk_start+10]
    if len(chunk) == 0:
        break
    # 앞 3명 테스트용
    test_indices.extend(chunk.index[:3].tolist())
    # 나머지 7명 훈련용
    train_indices.extend(chunk.index[3:].tolist())

df_train = df_sorted.loc[train_indices].copy()
df_test = df_sorted.loc[test_indices].copy()

# 체질량지수(bmi)는 비어 있는 사람이 있으니 고른 경우에만 훈련용의 중앙값으로 채우기
if "bmi" in selected_eng_feats:
    train_bmi_median = df_train["bmi"].median()
    df_train["bmi"] = df_train["bmi"].fillna(train_bmi_median)
    df_test["bmi"] = df_test["bmi"].fillna(train_bmi_median)

# 피처와 타겟 분리
X_train = df_train[selected_eng_feats]
y_train = df_train["stroke"]
X_test = df_test[selected_eng_feats]
y_test = df_test["stroke"]

st.info(f"📌 테스트 데이터 수: **{len(X_test):,}명** (요청하신 대로 1,533명 고정 검증 완료), 훈련 데이터 수: **{len(X_train):,}명**")
st.divider()

# 3. 모델 학습
# 로지스틱 회귀
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# 의사결정트리 (질문 3번까지, 마지막 마디 5명 미만 분할 금지, 난수 고정)
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)

# 다수결 기준 모델 (입력을 전혀 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)
majority_class = y_train.mode()[0]
# 훈련 및 테스트 정확도 계산
lr_train_acc = lr_model.score(X_train, y_train)
lr_test_acc = lr_model.score(X_test, y_test)

dt_train_acc = dt_model.score(X_train, y_train)
dt_test_acc = dt_model.score(X_test, y_test)

dummy_train_acc = (y_train == majority_class).mean()
dummy_test_acc = (y_test == majority_class).mean()

# 4. 모델 성능 평가 카드 (큰 숫자 카드 형태)
st.subheader("2. 모델별 정확도 비교")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 로지스틱 회귀(확률로 답하는 모델)")
    st.metric(label="테스트 정확도", value=f"{lr_test_acc*100:.2f}%")
    st.caption(f"훈련 데이터 정확도: {lr_train_acc*100:.2f}% | 테스트 데이터 정확도: {lr_test_acc*100:.2f}%")

with col2:
    st.markdown("### 의사결정트리(질문으로 답하는 모델)")
    st.metric(label="테스트 정확도", value=f"{dt_test_acc*100:.2f}%")
    st.caption(f"훈련 데이터 정확도: {dt_train_acc*100:.2f}% | 테스트 데이터 정확도: {dt_test_acc*100:.2f}%")

with col3:
    st.markdown("### 다수결 기준 모델")
    st.metric(label="테스트 정확도", value=f"{dummy_test_acc*100:.2f}%")
    st.caption(f"훈련 데이터 정확도: {dummy_train_acc*100:.2f}% | 테스트 데이터 정확도: {dummy_test_acc*100:.2f}% (많은 쪽: {'뇌졸중' if majority_class==1 else '아님'})")

st.divider()

# 5. 산점도 및 결정 경계(Decision Boundary) 시각화
st.subheader("3. 선택한 두 속성에 따른 테스트 데이터 산점도 및 결정 경계")

if len(selected_korean_feats) >= 2:
    # 그림을 위한 속성 2개 선택 (첫 번째와 두 번째 속성 고정)
    x_feat_korean = selected_korean_feats[0]
    y_feat_korean = selected_korean_feats[1]
    x_feat_eng = selected_eng_feats[0]
    y_feat_eng = selected_eng_feats[1]

    # 두 축이 아닌 나머지 속성들은 테스트 데이터의 중앙값으로 설정
    other_features_median = {}
    other_feats_eng = [f for f in selected_eng_feats if f not in [x_feat_eng, y_feat_eng]]
    for feat in other_feats_eng:
        other_features_median[feat] = X_test[feat].median()

    # 산점도 그리기 (Plotly)
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
        title=f"테스트 데이터 산점도 ({x_feat_korean} vs {y_feat_korean})"
    )

    # 로지스틱 회귀 0.5 확률 결정 경계선 추가 및 배경 칠하기
    # 2차원 그리드 생성
    x_min, x_max = X_test[x_feat_eng].min(), X_test[x_feat_eng].max()
    y_min, y_max = X_test[y_feat_eng].min(), X_test[y_feat_eng].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    grid_df = pd.DataFrame({x_feat_eng: xx.ravel(), y_feat_eng: yy.ravel()})
    # 나머지 속성은 중앙값으로 채우기
    for feat, val in other_features_median.items():
        grid_df[feat] = val

    # 순서 일치시키기
    grid_df = grid_df[selected_eng_feats]

    # 로지스틱 회귀 예측 확률
    lr_probs = lr_model.predict_proba(grid_df)[:, 1].reshape(xx.shape)

    # 0.5 경계선 추가 (contour)
    fig.add_trace(
        go.Contour(
            z=lr_probs,
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            contours=dict(start=0.5, end=0.5, size=0, coloring='lines'),
            line=dict(color='green', width=3),
            name="로지스틱 회귀 경계 (0.5)"
        )
    )

    # 의사결정트리 영역 색칠 (배경 옅은 색)
    dt_preds = dt_model.predict(grid_df).reshape(xx.shape)
    fig.add_trace(
        go.Contour(
            z=dt_preds,
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            colorscale=[[0, 'rgba(0, 0, 255, 0.05)'], [1, 'rgba(255, 0, 0, 0.05)']],
            showscale=False,
            hoverinfo='skip',
            name="의사결정트리 영역"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # 안내 문구 작성
    if other_feats_eng:
        median_desc = ", ".join([f"**{feat_map[f]}**: {val:.2f}" for f, val in other_features_median.items()])
        st.markdown(f"📝 **참고**: 두 축({x_feat_korean}, {y_feat_korean}) 외에 선택된 다른 속성({median_desc})은 테스트 데이터의 중앙값에 고정하여 계산되었습니다.")
    
    # 선이 그림 밖인지 여부 체크 및 안내
    # 경계선(0.5)이 현재 플롯 범위 내에 있는지 확인
    if (lr_probs.min() > 0.5) or (lr_probs.max() < 0.5):
        st.warning("⚠️ 로지스틱 회귀 모델의 0.5 결정 경계선이 현재 그려진 그림의 범위를 벗어나 있습니다.")
    else:
        st.success("✅ 로지스틱 회귀 모델의 0.5 결정 경계선이 그림 안에 정상적으로 포함되어 있습니다.")

st.divider()

# 6. 의사결정트리 시각화 (Graphviz DOT 문자열 직접 구성)
st.subheader("4. 의사결정트리 구조 (가지 갈라지는 그림)")
st.markdown("의사결정트리가 질문을 던져 가지를 뻗어 나가는 구조입니다. 각 마디의 훈련용 사람 수, 실제 뇌졸중인 사람 수, 그 비율을 함께 나타냅니다.")

# scikit-learn tree 속성을 활용해 DOT 문자열 직접 생성 또는 커스텀 정형화
def tree_to_dot(tree, feature_names, class_names=["아님", "뇌졸중"]):
    from sklearn.tree import _tree
    tree_ = tree.tree_
    
    dot_lines = [
        'digraph Tree {',
        'node [shape=box, style="filled, rounded", fontname="sans-serif"];',
        'edge [fontname="sans-serif"];'
    ]

    def recurse(node, lines):
        # 훈련용 사람 수 총합 구하기
        samples = tree_.n_node_samples[node]
        value = tree_.value[node][0]
        # class_names[1]이 뇌졸중(1)
        stroke_samples = value[1] if len(value) > 1 else 0
        ratio = (stroke_samples / samples) * 100 if samples > 0 else 0

        # 한글 속성 이름으로 변환
        eng_feat = feature_names[tree_.feature[node]] if tree_.feature[node] != _tree.TREE_LEAF else ""
        korean_feat = feat_map.get(eng_feat, eng_feat)

        if tree_.feature[node] != _tree.TREE_LEAF:
            threshold = tree_.threshold[node]
            # 질문 마디 (잎이 아님)
            label = f"{korean_feat} <= {threshold:.2f}\\n사람 수: {samples:,}명\\n뇌졸중: {stroke_samples}명 ({ratio:.1f}%)"
            lines.append(f'  {node} [label="{label}", fillcolor="#E3F2FD"];')
            
            # 왼쪽 자식
            left = tree_.children_left[node]
            lines.append(f'  {node} -> {left} [label="예"];')
            recurse(left, lines)
            
            # 오른쪽 자식
            right = tree_.children_right[node]
            lines.append(f'  {node} -> {right} [label="아니요"];')
            recurse(right, lines)
        else:
            # 잎 마디 (답을 내는 마디): 답에 따라 색상 달리 지정 (뇌졸중 비율이 높으면 연한 빨강, 아니면 연한 파랑)
            pred_class = 1 if stroke_samples > (samples / 2) else 0
            fillcolor = "#FFCDD2" if pred_class == 1 else "#C8E6C9"
            pred_text = class_names[pred_class]
            
            label = f"답: {pred_text}\\n사람 수: {samples:,}명\\n뇌졸중: {stroke_samples}명 ({ratio:.1f}%)"
            lines.append(f'  {node} [label="{label}", fillcolor="{fillcolor}"];')

    recurse(0, dot_lines)
    dot_lines.append('}')
    return "\n".join(dot_lines)

dot_string = tree_to_dot(dt_model.tree_, selected_eng_feats)
st.graphviz_chart(dot_string)

# 7. 의사결정트리 분석 요약 통계 출력
tree_ = dt_model.tree_
n_nodes = tree_.node_count
# 리프 노드 수 및 예측 클래스 파악
leaf_nodes = [i for i in range(n_nodes) if tree_.children_left[i] == _tree.TREE_LEAF if hasattr(_tree, 'TREE_LEAF') else i]
# scikit-learn 호환 리프 판별
leaf_count = 0
not_stroke_leaf_count = 0

for i in range(n_nodes):
    if tree_.children_left[i] == tree_.children_right[i]: # 리프 노드
        leaf_count += 1
        val = tree_.value[i][0]
        stroke_s = val[1] if len(val) > 1 else 0
        total_s = tree_.n_node_samples[i]
        if stroke_s <= total_s / 2: # '아님'으로 답하는 칸
            not_stroke_leaf_count += 1

# 실제로 트리가 사용한 속성 추출
used_features_eng = set()
for i in range(n_nodes):
    feat_idx = tree_.feature[i]
    if feat_idx != -2: # -2는 TREE_LEAF를 의미
        used_features_eng.add(selected_eng_feats[feat_idx])

used_features_korean = [feat_map[f] for f in used_features_eng]
used_features_str = ", ".join(used_features_korean) if used_features_korean else "없음"

st.markdown("### 📊 의사결정트리 분석 요약")
st.markdown(f"- **답을 내는 마디(리프 노드) 총 개수**: {leaf_count}개")
st.markdown(f"- **그중 '아님'이라고 답하는 칸의 개수**: {not_stroke_leaf_count}개")
st.markdown(f"- **이 나무가 실제로 질문에 사용한 속성**: {used_features_str}")
