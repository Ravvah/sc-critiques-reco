from __future__ import annotations

from pydantic import BaseModel


class RecommendationItem(BaseModel):
    """A single recommendation item returned by the API"""

    review_id: str | int
    film_id: str | int
    text: str
    cos: float | None = None
    upvotes: int
    final: float


class RecommendationResponse(BaseModel):
    """Response containing a list of recommendation items"""

    items: list[RecommendationItem]
