import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# 프로젝트 구조(요청사항)
# D:\~\CLOUD
#   ├─ .env
#   └─ rag
#       ├─ origin\company_manual01.csv
#       ├─ embedding\em01.csv
#       ├─ embedding\em01.npy
#       └─ app.py
# =========================================================


# 0) 경로/환경 설정
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


# 1) 유틸 함수
def cosine_similarity_matrix(emb_matrix: np.ndarray, q_vec: np.ndarray) -> np.ndarray:
    """emb_matrix: (N, D), q_vec: (D,) -> (N,)"""
    q_norm = np.linalg.norm(q_vec) + 1e-12
    e_norm = np.linalg.norm(emb_matrix, axis=1) + 1e-12
    return (emb_matrix @ q_vec) / (e_norm * q_norm)

def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def build_doc_embeddings_once(df: pd.DataFrame) -> np.ndarray:
    vectors = []
    total = len(df)
    for i, text in enumerate(df["content"].tolist(), start=1):
        vec = get_embedding(text)
        vectors.append(vec)
        print(f"[embedding] {i}/{total} done")
    return np.vstack(vectors).astype(np.float32)

def ensure_dirs():
    os.makedirs(ORIGIN_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)

def load_or_create_index():
    """
    - embedding/em01.csv + embedding/em01.npy가 있으면 로드
    - 없으면 origin/company_manual01.csv 읽고 임베딩 생성 후 캐시 저장
    """
    ensure_dirs()

    # 캐시가 존재하면 로드
    if os.path.exists(CACHE_DOCS) and os.path.exists(CACHE_EMB):
        docs_df = pd.read_csv(CACHE_DOCS)
        emb = np.load(CACHE_EMB)
        if "content" not in docs_df.columns:
            raise ValueError("캐시된 em01.csv에 'content' 컬럼이 없습니다.")
        if emb.shape[0] != len(docs_df):
            raise ValueError("em01.npy 행 수와 em01.csv 문서 수가 일치하지 않습니다.")
        print("[OK] Loaded cached index (embedding/em01.csv + embedding/em01.npy).")
        return docs_df, emb

    # 캐시가 없으면 원본으로부터 생성
    if not os.path.exists(ORIGIN_CSV):
        raise FileNotFoundError(f"원본 CSV를 찾을 수 없습니다: {ORIGIN_CSV}")

    df = pd.read_csv(ORIGIN_CSV)
    if "content" not in df.columns:
        raise ValueError("company_manual01.csv에는 최소 'content' 컬럼이 필요합니다.")

    df["content"] = df["content"].astype(str)

    print("[INFO] Cache not found. Building embeddings ONCE (first run)...")
    emb = build_doc_embeddings_once(df)

    # 캐시 저장
    df.to_csv(CACHE_DOCS, index=False)
    np.save(CACHE_EMB, emb)
    print("[OK] Cache saved:")
    print(f" - {CACHE_DOCS}")
    print(f" - {CACHE_EMB}")

    return df, emb

def search_manual(docs_df: pd.DataFrame, doc_emb: np.ndarray, question: str, top_k: int = 3) -> pd.DataFrame:
    q_vec = get_embedding(question)  # 질문은 매번 1회 임베딩 호출
    sims = cosine_similarity_matrix(doc_emb, q_vec)
    result = docs_df.copy()
    result["similarity"] = sims
    return result.sort_values("similarity", ascending=False).head(top_k)

def ask_ai(question: str, snippets: list[str]) -> str:
    context = "\n".join([f"- {s}" for s in snippets])

    prompt = f"""
당신은 회사 규정을 잘 아는 인사/총무 담당자입니다.
아래 [회사 규정] 근거만 사용해 간결하고 명확하게 답변하세요.
근거에 없는 내용이면 "(신유리2)규정에 명시되어 있지 않습니다."라고 답하세요.


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


# 2) 메인: 질문 반복 입력
def main():
    try:
        docs_df, doc_emb = load_or_create_index()
    except Exception as e:
        print(f"\n[ERROR] 인덱스 준비 실패: {type(e).__name__}: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("RAG CLI Demo (cached doc embeddings)")
    print(f"- ENV:     {ENV_PATH}")
    print(f"- ORIGIN:  {ORIGIN_CSV}")
    print(f"- CACHE:   {CACHE_DOCS}")
    print(f"- EMB:     {CACHE_EMB}")
    print("종료: exit / quit / 나가기")
    print("=" * 70)

    while True:
        question = input("\n질문> ").strip()
        if question.lower() in ("exit", "quit") or question == "나가기":
            print("종료합니다.")
            break
        if not question:
            continue

        try:
            hits = search_manual(docs_df, doc_emb, question, top_k=3)
            snippets = hits["content"].astype(str).tolist()

            answer = ask_ai(question, snippets)

            print("\n[답변]")
            print(answer)

            print("\n[참조 규정 Top-3]")
            for i, row in enumerate(hits.itertuples(index=False), start=1):
                sim = getattr(row, "similarity")
                content = getattr(row, "content")
                print(f"{i}. (sim={sim:.3f}) {content}")

        except Exception as e:
            print(f"\n[ERROR] 처리 실패: {type(e).__name__}: {e}")
            print("- (힌트) 429면 OpenAI 결제/쿼터 문제입니다.")

if __name__ == "__main__":
    main()
