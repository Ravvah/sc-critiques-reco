
from __future__ import annotations

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

from fastapi import FastAPI

from app.routers.recommendations import router

app = FastAPI(
    title="sc_recommandation",
    version="1.0.0",
    description="API de recommandation de critiques de films (HNSW + TFIDF)",
    docs_url="/",
    redoc_url="/redoc",
)

app.include_router(router, prefix="/v1")
