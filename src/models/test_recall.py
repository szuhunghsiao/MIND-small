# src/models/test_recall.py
import pandas as pd
from tfidf_retriever import TfidfRetriever

def main():
    samples = pd.read_parquet("data/processed/train_samples.parquet")

    # find a user with history
    row = samples[samples["history"].notna() & (samples["history"] != "")].iloc[0]
    history_ids = row["history"].split()

    retriever = TfidfRetriever()
    cands = retriever.get_candidates(history_ids, topk=20)

    print("user history (first 5):", history_ids[:5])
    print("candidates (top 20):", cands[:20])

if __name__ == "__main__":
    main()