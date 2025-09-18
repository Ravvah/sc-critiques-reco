from __future__ import annotations

import json
import os

import faiss
import joblib
import numpy as np
import pandas as pd
import logging
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

logger = logging.getLogger("sc_recommandation")

class ReviewRecommender:
    def __init__(self, indices_dir: str = "./indices"):
        self.indices_dir = indices_dir
        self.cache = {}
        self.model = None
        self.sentiment_model = None
    def get_sentiment(self, text: str) -> float:
        """Retourne le score de sentiment [-1, 1] pour un texte en français via HuggingFace."""
        try:
            if self.sentiment_model is None:
                self.sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
            result = self.sentiment_model(text[:512])[0]
            # Mapping: 1 étoile = très négatif (-1), 5 étoiles = très positif (+1)
            stars = int(result["label"][0])
            return (stars - 3) / 2  # -1 (1 étoile), 0 (3 étoiles), +1 (5 étoiles)
        except Exception as e:
            logger.warning(f"[SENTIMENT] Erreur modèle HuggingFace: {e}")
            return 0.0

    def load_film(self, film_id: str):
        logger.info(f"[LOAD] Chargement index pour film_id={film_id}")
        key = str(film_id)
        if key in self.cache:
            return self.cache[key]
        d = os.path.join(self.indices_dir, key)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"no index for film {film_id}")
        idx = faiss.read_index(os.path.join(d, "hnsw.faiss"))
        meta = pd.read_csv(os.path.join(d, "meta.csv"))
        stats = json.loads(open(os.path.join(d, "stats.json")).read())
        vec = joblib.load(os.path.join(d, "tfidf_vectorizer.joblib"))
        X = sparse.load_npz(os.path.join(d, "tfidf_X.npz"))
        info = json.loads(open(os.path.join(d, "model.json")).read())
        b = {
            "index": idx,
            "meta": meta,
            "stats": stats,
            "vec": vec,
            "X": X,
            "info": info,
        }
        self.cache[key] = b
        return b

    def _model(self, name: str):
        if self.model is None:
            self.model = SentenceTransformer(name)
        return self.model

    def dense_topk(self, film_id: str, text: str, k: int, ef_search: int):
        logger.info(f"[DENSE] Recherche dense pour film_id={film_id} k={k}")
        b = self.load_film(film_id)
        m = self._model(b["info"]["emb_model"])
        q = m.encode([text], normalize_embeddings=True).astype("float32")
        b["index"].hnsw.efSearch = int(ef_search)
        D, I = b["index"].search(q, k)
        rows = []
        for s, i in zip(D[0], I[0]):
            if i < 0:
                continue
            r = b["meta"].iloc[int(i)]
            rows.append(
                {
                    "review_id": r["review_id"],
                    "film_id": r["film_id"],
                    "text": r["text"],
                    "cos": float(s),
                    "upvotes": int(r.get("upvotes", 0)),
                }
            )
        return rows

    def fs_topm(self, film_id: str, text: str, m: int):
        logger.info(f"[FS] Recherche lexicale (TF-IDF) pour film_id={film_id} m={m}")
        b = self.load_film(film_id)
        qv = b["vec"].transform([text])
        sims = cosine_similarity(qv, b["X"]).ravel()
        order = np.argsort(-sims)[:m]
        out = []
        for idx in order:
            r = b["meta"].iloc[int(idx)]
            out.append(
                {
                    "review_id": r["review_id"],
                    "film_id": r["film_id"],
                    "text": r["text"],
                    "lex": float(sims[int(idx)]),
                    "upvotes": int(r.get("upvotes", 0)),
                }
            )
        return out

    def fuse(self, dens, spar, k: int, lambda_up: float):
        logger.info(f"[FUSION] Fusion des scores sémantique et lexical (FS)")
        up_max = 1 + max([x["upvotes"] for x in dens + spar] + [0])
        best = {}
        for x in dens:
            sem = (x["cos"] + 1.0) / 2.0
            up = np.log1p(x["upvotes"]) / np.log1p(up_max)
            score = sem * (1.0 + lambda_up * up)
            best[x["review_id"]] = {**x, "final": float(score)}
        for x in spar:
            key = x["review_id"]
            w = 0.30  
            if key in best:
                base = best[key]["final"]
                score = base + w * float(max(0.0, min(1.0, x["lex"])))
                best[key]["final"] = float(score)
            else:
                score = w * float(max(0.0, min(1.0, x["lex"])))
                best[key] = {
                    "review_id": x["review_id"],
                    "film_id": x["film_id"],
                    "text": x["text"],
                    "cos": 0.0,
                    "upvotes": x["upvotes"],
                    "final": float(score),
                }
        items = sorted(best.values(), key=lambda z: z["final"], reverse=True)
        return items[:k]

    def recommend(
        self,
        film_id: str,
        text: str,
        k: int = 10,
        mode: str = "dense",
        ef_search: int = 48,
        lambda_up: float = 0.10,
        exclude_review_id=None,
        use_sentiment=False,
    ):
        logger.info(f"[RECO] Démarrage recommandation pour film_id={film_id} mode={mode}")
        if mode == "dense":
            dens = self.dense_topk(film_id, text, k * 5, ef_search)
            out = self.fuse(dens, [], k, lambda_up)
        elif mode == "hybrid":
            dens = self.dense_topk(film_id, text, k * 5, ef_search)
            spar = self.fs_topm(film_id, text, 50)
            out = self.fuse(dens, spar, k, lambda_up)
        else:
            raise ValueError("Mode doit être 'dense' ou 'hybrid'")
        out = [x for x in out if str(x["review_id"]) != str(exclude_review_id)]
        # Rerank sur le sentiment si demandé dans la requête
        if use_sentiment:
            try:
                query_sentiment = self.get_sentiment(text)
                logger.info(f"[SENTIMENT] Sentiment requête (HuggingFace): {query_sentiment}")
                for item in out:
                    item_sentiment = self.get_sentiment(item["text"])
                    # Si le sentiment de la requête et de la critique sont alignés, on ajoute un bonus
                    if (query_sentiment * item_sentiment) > 0.15:
                        item["final"] += 0.05
                out = sorted(out, key=lambda x: x["final"], reverse=True)
                logger.info(f"[RERANK] Reranking sur le sentiment effectué (HuggingFace)")
            except Exception as e:
                logger.warning(f"[SENTIMENT] Erreur analyse sentiment (HuggingFace): {e}")
        logger.info(f"[SUCCESS] {len(out[:k])} recommandations retournées pour film_id={film_id}")
        return out[:k]
