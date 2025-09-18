from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, status

from app.schemas.query import Query
from app.schemas.recommendations import RecommendationResponse
from app.services.recommender import ReviewRecommender

logger = logging.getLogger("sc_recommandation")
router = APIRouter()


@router.get("/readyz")
def readyz():
    """Ready check endpoint (index disponible)"""
    logger.info("[READYZ] Vérification disponibilité index...")
    ok = os.path.isdir("./indices") and any(os.scandir("./indices"))
    logger.info(f"[READYZ] Index disponible: {ok}")
    return {"ready": ok}


@router.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(query: Query):
    """Get recommendations based on text or review ID"""
    logger.info(f"[REQUEST] film_id={query.film_id} mode={query.mode} k={query.k}")
    recommender = ReviewRecommender(indices_dir="./indices")
    try:
        logger.info(f"[INDEX] Chargement index pour film_id={query.film_id}")
        text = query.text
        if query.mode not in ["dense", "hybrid"]:
            logger.warning(f"Mode {query.mode} non supporté")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mode doit être 'dense' ou 'hybrid'",
            )
        logger.info(
            f"[RECO] Démarrage recommandation pour film_id={query.film_id} mode={query.mode}"
        )
        results = recommender.recommend(
            query.film_id,
            text,
            k=query.k,
            mode=query.mode,
            ef_search=query.ef_search,
            lambda_up=query.lambda_up,
            exclude_review_id=query.review_id,
            use_sentiment=query.use_sentiment,
        )
        logger.info(
            f"[SUCCESS] {len(results)} recommandations retournées pour film_id={query.film_id}"
        )
        return {"items": results}
    except FileNotFoundError as e:
        logger.error(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"[ERROR] {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
