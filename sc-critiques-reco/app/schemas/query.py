from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Query(BaseModel):
    use_sentiment: bool = Field(False, description="Activer le reranking par sentiment (optionnel)")
    """Request model for recommendation queries"""

    film_id: str = Field(..., description="Film identifier", examples=["fight_club"])
    text: Optional[str] = Field(None, description="Text to find similar reviews for")
    review_id: Optional[str | int] = Field(
        None, description="Review ID to find similar reviews for"
    )
    k: int = Field(10, ge=1, le=50, description="Number of recommendations to return")
    mode: str = Field(
        "dense",
        pattern="^(dense|hybrid)$",
        description="Recommendation mode ('dense' ou 'hybrid')",
    )
    ef_search: int = Field(48, ge=8, le=256, description="HNSW search parameter")
    lambda_up: float = Field(
        0.10, ge=0.0, le=1.0, description="Upvotes weighting factor"
    )
