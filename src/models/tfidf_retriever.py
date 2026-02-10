# src/models/tfidf_retriever.py
from pathlib import Path
import numpy as np
import joblib
from scipy import sparse
from sklearn.preprocessing import normalize

PROCESSED = Path("data/processed")

class TfidfRetriever:
    def __init__(self):
        self.vectorizer = joblib.load(PROCESSED / "tfidf_vectorizer.joblib")
        self.X = sparse.load_npz(PROCESSED / "news_tfidf_matrix.npz")  # [N, V]
        self.news_ids = (PROCESSED / "news_id_list.txt").read_text().splitlines()

        # normalize for dot product for cosine similarity
        self.X = normalize(self.X)

        # build mapping: news_id -> row index
        self.id2idx = {nid: i for i, nid in enumerate(self.news_ids)}

    def user_profile(self, history_news_ids):
        """history_news_ids: list[str]"""
        idxs = [self.id2idx[nid] for nid in history_news_ids if nid in self.id2idx]
        if not idxs:
            return None
        # get history vectors average
        profile = self.X[idxs].mean(axis=0)
        profile = sparse.csr_matrix(profile)
        profile = normalize(profile)
        return profile

    def get_candidates(self, history_news_ids, topk=1000, exclude_history=True):
        profile = self.user_profile(history_news_ids)
        if profile is None:
            return []  # cold start for fallback later

        # cosine sim: (1,V) dot (V,N) -> (1,N)
        scores = profile @ self.X.T
        scores = np.asarray(scores.todense()).ravel()

        # get topk
        top_idx = np.argpartition(-scores, kth=min(topk, len(scores)-1))[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        cands = [self.news_ids[i] for i in top_idx]

        if exclude_history:
            hist_set = set(history_news_ids)
            cands = [nid for nid in cands if nid not in hist_set]

        return cands