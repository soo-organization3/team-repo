import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# 구조
# D:\~\CLOUD
#   ├─ .env
#   └─ rag
#       ├─ origin\company_manual01.csv
#       ├─ embedding\em01.csv
#       ├─ embedding\em01.npy
#       ├─ app.py
#       ├─ app_web.py
#       └─ requirements.txt
# =========================================================

# -----------------------------
# 0) 경로/환경 설정
# -----------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))       # ...\rag
PROJECTS_DIR = os.path.dirname(APP_DIR)                    # ...\CLOUD
ENV_PATH = os.path.join(PROJECTS_DIR, ".env")

ORIGIN_DIR = os.path.join(APP_DIR, "origin")
EMB_DIR = os.path.join(APP_DIR, "embedding")

ORIGIN_CSV = os.path.join(ORIGIN_DIR, "company_manual01.csv")  # 원본
CACHE_DOCS = os.path.join(EMB_DIR, "em01.csv")                 # 캐시 복사본
CACHE_EMB = os.path.join(EMB_DIR, "em01.npy")                  # 임베딩 결과

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# .env 로드
load_dotenv(ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 Projects 폴더의 .env 파일에 설정되어 있지 않습니다.")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# 1) 유틸
# -----------------------------
def ensure_dirs():
    os.makedirs(ORIGIN_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)

def cosine_similarity_matrix(emb_matrix: np.ndarray, q_vec: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(q_vec) + 1e-12
    e_norm = np.linalg.norm(emb_matrix, axis=1) + 1e-12
    return (emb_matrix @ q_vec) / (e_norm * q_norm)

def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

def build_doc_embeddings_once(df: pd.DataFrame) -> np.ndarray:
    vectors = []
    total = len(df)
    for i, text in enumerate(df["content"].tolist(), start=1):
        vec = get_embedding(text)  # 쿼터 부족이면 여기서 예외
        vectors.append(vec)
        # Streamlit 진행 표시용: 너무 잦은 rerun을 막기 위해 print만
        print(f"[embedding] {i}/{total} done")
    return np.vstack(vectors).astype(np.float32)

def load_or_create_index() -> tuple[pd.DataFrame, np.ndarray]:
    """
    - embedding/em01.csv + embedding/em01.npy가 있으면 로드
    - 없으면 origin/company_manual01.csv 읽고 임베딩 생성 후 캐시 저장
    """
    ensure_dirs()

    if os.path.exists(CACHE_DOCS) and os.path.exists(CACHE_EMB):
        docs_df = pd.read_csv(CACHE_DOCS)
        emb = np.load(CACHE_EMB)
        if "content" not in docs_df.columns:
            raise ValueError("캐시된 em01.csv에 'content' 컬럼이 없습니다.")
        if emb.shape[0] != len(docs_df):
            raise ValueError("em01.npy 행 수와 em01.csv 문서 수가 일치하지 않습니다.")
        return docs_df, emb

    if not os.path.exists(ORIGIN_CSV):
        raise FileNotFoundError(f"원본 CSV를 찾을 수 없습니다: {ORIGIN_CSV}")

    df = pd.read_csv(ORIGIN_CSV)
    if "content" not in df.columns:
        raise ValueError("company_manual01.csv에는 최소 'content' 컬럼이 필요합니다.")
    df["content"] = df["content"].astype(str)

    emb = build_doc_embeddings_once(df)

    df.to_csv(CACHE_DOCS, index=False)
    np.save(CACHE_EMB, emb)

    return df, emb

def search_manual(docs_df: pd.DataFrame, doc_emb: np.ndarray, question: str, top_k: int = 3) -> pd.DataFrame:
    q_vec = get_embedding(question)  # 질문당 1회 임베딩
    sims = cosine_similarity_matrix(doc_emb, q_vec)
    result = docs_df.copy()
    result["similarity"] = sims
    return result.sort_values("similarity", ascending=False).head(top_k)

def ask_ai(question: str, snippets: list[str]) -> str:
    context = "\n".join([f"- {s}" for s in snippets])

    prompt = f"""
당신은 회사 규정을 잘 아는 인사/총무 담당자입니다.
아래 [회사 규정] 근거만 사용해 간결하고 명확하게 답변하세요.
근거에 없는 내용이면 "규정에 명시되어 있지 않습니다."라고 답하세요.

[회사 규정]
{context}

[질문]
{question}

[답변 형식]
- 결론: (한 문장)
- 근거: (규정 문장 인용/요약 1~2개)
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

# Streamlit 캐시: 문서 임베딩은 최초 1회 생성 후 파일 캐시를 사용하지만,
# 앱 리런(re-run)마다 파일을 다시 읽지 않도록 st.cache_resource로 감싼다.
@st.cache_resource(show_spinner=False)
def get_index_cached():
    return load_or_create_index()

# -----------------------------
# 2) Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG 매뉴얼 Q&A (Streamlit)", layout="wide")
st.title("사내 매뉴얼 Q&A (with RAG)")
st.caption("origin/company_manual01.csv → (최초 1회) → embedding/em01.csv + em01.npy 캐시 생성 후 재사용")

with st.sidebar:
    st.subheader("설정")
    top_k = st.slider("검색 Top-K", 1, 5, 3)
    show_similarity = st.toggle("유사도 표시", value=True)
    # st.divider()
    # st.subheader("파일 경로")
    # st.code(
    #     f".env: {ENV_PATH}\n"
    #     f"origin: {ORIGIN_CSV}\n"
    #     f"cache: {CACHE_DOCS}\n"
    #     f"emb: {CACHE_EMB}"
    # )
    st.divider()
    st.subheader("추천 질문")
    st.code(
        "국내 출장 식비 기준 알려줘\n"
        "연차는 반차로 쓸 수 있어?\n"
        "야근하면 보상은 어떻게 돼?\n"
        "회사 문서를 개인 메일로 보내도 돼?\n"
        "명예퇴직시 퇴직금 정산 방법은?"
    )

# 인덱스 로드/생성
try:
    with st.spinner("문서 인덱스를 준비 중입니다... (캐시가 없으면 최초 1회 임베딩 생성)"):
        docs_df, doc_emb = get_index_cached()
    st.success(f"인덱스 준비 완료! (문장 수: {len(docs_df)})")
except Exception as e:
    st.error(f"인덱스 준비 실패: {type(e).__name__}: {e}")
    st.stop()

question = st.text_input("질문을 입력하세요", placeholder="예: 국내 출장 식비 기준 알려줘")
colA, colB = st.columns([1, 1])

ask = st.button("질문하기", type="primary", disabled=(not question.strip()))

if ask:
    try:
        with st.spinner("관련 규정 검색 중..."):
            hits = search_manual(docs_df, doc_emb, question, top_k=top_k)
            snippets = hits["content"].astype(str).tolist()

        with st.spinner("답변 생성 중..."):
            answer = ask_ai(question, snippets)

        with colA:
            st.subheader("답변")
            st.write(answer)

        with colB:
            st.subheader("참조 규정")
            view_cols = ["content"]
            if "category" in hits.columns:
                view_cols = ["category", "content"]
            if show_similarity:
                view_cols.append("similarity")

            # 보기 좋게 정렬
            view = hits[view_cols].copy()
            if "similarity" in view.columns:
                view["similarity"] = view["similarity"].astype(float).round(3)

            st.dataframe(view, use_container_width=True)

    except Exception as e:
        st.error(f"처리 실패: {type(e).__name__}: {e}")
        st.info("429 (insufficient_quota)이면 OpenAI 결제/쿼터 문제입니다.")
