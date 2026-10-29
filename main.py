import os
import json
import pandas as pd
import streamlit as st

# 1. 페이지 기본 설정 (앱 제목, 아이콘, 탭 제목 설정)
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 파일 경로 설정
DATA_URL = "https://raw.githubusercontent.com/jimnny1230/data/refs/heads/main/stroke.csv"
SAVE_DIR = "내_기록"
SAVE_FILE = os.path.join(SAVE_DIR, "user_records.json")

# 2. 저장용 폴더 및 초기 상태 설정
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

def load_user_records():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"meanings": {}, "source_text": ""}
    return {"meanings": {}, "source_text": ""}

def save_user_records(records):
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

# 사용자 기록 불러오기 및 세션 상태 초기화
saved_data = load_user_records()
if "meanings" not in st.session_state:
    st.session_state["meanings"] = saved_data.get("meanings", {})
if "source_text" not in st.session_state:
    st.session_state["source_text"] = saved_data.get("source_text", "")

# 3. 데이터 로드 (캐싱 처리)
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

# 4. 앱 헤더
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("이 앱은 뇌졸중 관련 데이터를 확인하고 분석하는 실습 공간입니다.")

st.divider()

# 5. 핵심 지표 (큰 숫자 카드 4개)
total_count = len(df)
column_count = len(df.columns)
stroke_count = int(df["stroke"].sum())
stroke_ratio = (stroke_count / total_count) * 100 if total_count > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 사람 수", f"{total_count:,}명")
col2.metric("열 개수", f"{column_count}개")
col3.metric("뇌졸중 환자 수 (stroke=1)", f"{stroke_count:,}명")
col4.metric("뇌졸중 비율", f"{stroke_ratio:.2f}%")

st.divider()

# 6. 데이터열 정보 메타데이터 및 교재 직접 입력 표
st.subheader("📋 데이터 열 정보 및 정보 입력")
st.caption("우리말 뜻 칸을 직접 채워보세요. 작성한 내용은 자동으로 저장됩니다.")

# 열 정보 데이터프레임 구축
cols_info = []
for col in df.columns:
    col_type = str(df[col].dtype)
    null_count = int(df[col].isnull().sum())
    cols_info.append({
        "열 이름": col,
        "값의 종류": col_type,
        "빈 값 개수": null_count
    })

info_df = pd.DataFrame(cols_info)

# 사용자가 입력할 '우리말 뜻' 입력 폼 구현
updated_meanings = {}
with st.form("meanings_form"):
    # 표 헤더
    h1, h2, h3, h4 = st.columns([2, 3, 2, 2])
    h1.markdown("**열 이름**")
    h2.markdown("**우리말 뜻 (직접 입력)**")
    h3.markdown("**값의 종류**")
    h4.markdown("**빈 값 개수**")
    st.divider()

    for idx, row in info_df.iterrows():
        c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
        col_name = row["열 이름"]
        c1.write(f"`{col_name}`")
        
        # 이전 입력값 불러오기
        default_val = st.session_state["meanings"].get(col_name, "")
        user_input = c2.text_input(
            label=f"{col_name} 뜻",
            value=default_val,
            key=f"meaning_{col_name}",
            label_visibility="collapsed"
        )
        updated_meanings[col_name] = user_input
        
        c3.write(row["값의 종류"])
        c4.write(str(row["빈 값 개수"]))

    save_meanings_btn = st.form_submit_button("우리말 뜻 저장하기")
    if save_meanings_btn:
        st.session_state["meanings"] = updated_meanings
        current_records = load_user_records()
        current_records["meanings"] = updated_meanings
        save_user_records(current_records)
        st.success("우리말 뜻이 '내_기록' 폴더에 저장되었습니다.")

st.divider()

# 7. 데이터 상위 5줄 미리보기
st.subheader("🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# 8. 데이터 출처 작성란
st.subheader("📌 데이터 출처")
st.caption("교재에 있는 데이터 출처 내용을 입력해 주세요. 작성 내용은 자동 저장됩니다.")

with st.form("source_form"):
    source_input = st.text_area(
        "출처 내용 입력",
        value=st.session_state["source_text"],
        height=100,
        placeholder="교재에 있는 데이터 출처 관련 설명을 입력하세요."
    )
    save_source_btn = st.form_submit_button("출처 저장하기")
    
    if save_source_btn:
        st.session_state["source_text"] = source_input
        current_records = load_user_records()
        current_records["source_text"] = source_input
        save_user_records(current_records)
        st.success("출처 정보가 '내_기록' 폴더에 저장되었습니다.")
