# src/ingest/mind_to_samples.py
import pandas as pd
from pathlib import Path

def load_news(news_path: str) -> pd.DataFrame:
    cols = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    df = pd.read_csv(news_path, sep="\t", header=None)
    df.columns = cols[:df.shape[1]]
    return df[["news_id", "category", "subcategory", "title", "abstract"]]

def load_behaviors(behaviors_path: str) -> pd.DataFrame:
    cols = ["impression_id", "user_id", "time", "history", "impressions"]
    df = pd.read_csv(behaviors_path, sep="\t", header=None)
    df.columns = cols[:df.shape[1]]
    return df[["impression_id", "user_id", "time", "history", "impressions"]]

def explode_impressions(beh: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in beh.iterrows():
        history = r["history"] if isinstance(r["history"], str) else ""
        imps = str(r["impressions"]).split()
        for imp in imps:
            nid, label = imp.split("-")
            rows.append({
                "user_id": r["user_id"],
                "time": r["time"],
                "history": history,
                "candidate_news_id": nid,
                "label": int(label),
            })
    return pd.DataFrame(rows)

def main():
    raw = Path("data/raw")
    train_dir = raw / "MINDsmall_train"
    news = load_news(str(train_dir / "news.tsv"))
    beh = load_behaviors(str(train_dir / "behaviors.tsv"))
    samples = explode_impressions(beh)

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(out / "train_samples.parquet", index=False)
    news.to_parquet(out / "news.parquet", index=False)

if __name__ == "__main__":
    main()