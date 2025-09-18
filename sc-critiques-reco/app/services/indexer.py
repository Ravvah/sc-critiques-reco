from __future__ import annotations

import glob
import json
import os

import faiss
import joblib
import pandas as pd
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from app.utils import clean_text

OUT = os.getenv("OUT_DIR", "./indices")
DATA = os.getenv("DATA_DIR", "./data")
MODEL = os.getenv(
    "EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
os.makedirs(OUT, exist_ok=True)


def infer_film_id(path: str) -> str:
    b = os.path.basename(path).lower()
    if "fight" in b and "club" in b:
        return "fight_club"
    if "interstellar" in b:
        return "interstellar"
    return os.path.splitext(b)[0]


def build_indices():
    paths = glob.glob(os.path.join(DATA, "*.csv"))
    frames = []
    for p in paths:
        raw = pd.read_csv(p)
        df = pd.DataFrame()
        df["review_id"] = raw["id"]
        df["text"] = raw["review_content"].astype(str).map(clean_text)
        df["upvotes"] = raw.get("gen_review_like_count", 0).fillna(0).astype(int)
        df["film_id"] = infer_film_id(p)
        frames.append(df)
    if not frames:
        raise SystemExit("no data")
    df = pd.concat(frames, ignore_index=True)
    model = SentenceTransformer(MODEL)
    for film_id, g in df.groupby("film_id"):
        film_dir = os.path.join(OUT, str(film_id))
        os.makedirs(film_dir, exist_ok=True)
        texts = g["text"].tolist()
        emb = model.encode(texts, batch_size=128, normalize_embeddings=True)
        d = emb.shape[1]
        idx = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 200
        idx.add(emb.astype("float32"))
        faiss.write_index(idx, os.path.join(film_dir, "hnsw.faiss"))
        g[["review_id", "film_id", "text", "upvotes"]].reset_index(drop=True).to_csv(
            os.path.join(film_dir, "meta.csv"), index=False
        )
        up_max = int(g["upvotes"].max())
        open(os.path.join(film_dir, "stats.json"), "w").write(
            json.dumps({"upvotes_max": up_max})
        )
        vec = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 1), stop_words="english"
        )
        X = vec.fit_transform(texts)
        joblib.dump(vec, os.path.join(film_dir, "tfidf_vectorizer.joblib"))
        sparse.save_npz(os.path.join(film_dir, "tfidf_X.npz"), X)
        open(os.path.join(film_dir, "model.json"), "w").write(
            json.dumps({"mode": "dense_hnsw", "emb_model": MODEL})
        )


if __name__ == "__main__":
    build_indices()
