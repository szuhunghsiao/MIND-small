# src/models/build_tfidf_index.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

PROCESSED = Path("data/processed")

def build_text(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("")
    abstract = df["abstract"].fillna("")
    return (title + " " + abstract).str.strip()

def main():
    news_path = PROCESSED / "news.parquet"
    assert news_path.exists(), f"Missing: {news_path}"

    news = pd.read_parquet(news_path)
    # dedupe
    news = news.drop_duplicates(subset=["news_id"]).reset_index(drop=True)

    texts = build_text(news)

    vectorizer = TfidfVectorizer(
        max_features=200_000,
        ngram_range=(1, 2),
        min_df=3,
        stop_words="english"
    )

    X = vectorizer.fit_transform(texts)  # sparse matrix [num_news, vocab]
    assert sparse.issparse(X)

    # model & matrix
    joblib.dump(vectorizer, PROCESSED / "tfidf_vectorizer.joblib")
    sparse.save_npz(PROCESSED / "news_tfidf_matrix.npz", X)

    # save index mapping（news row -> news_id）
    (PROCESSED / "news_id_list.txt").write_text("\n".join(news["news_id"].tolist()))

    print("TF-IDF index built")
    print(f"news rows: {len(news):,}")
    print(f"vocab size: {len(vectorizer.vocabulary_):,}")
    print(f"matrix shape: {X.shape}")

if __name__ == "__main__":
    main()