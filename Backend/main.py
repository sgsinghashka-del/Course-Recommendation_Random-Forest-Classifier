from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from Backend.ab_testing import ab_predict
from Backend.feedback import store_feedback
from Backend.recommender import recommend_courses

app = FastAPI(
    title="Course Recommendation API",
    version="1.0.0",
    description="API for recommending courses based on learner profile.",
)


class UserInput(BaseModel):
    user_id: int | None = Field(default=None, ge=1)
    age: int = Field(..., ge=18, le=80)
    experience: int = Field(..., ge=0, le=40)
    interest_level: int = Field(..., ge=1, le=10)
    preferred_domain: str = Field(..., min_length=2, max_length=50)


class FeedbackPayload(BaseModel):
    user_input: UserInput
    recommended_course: str = Field(..., min_length=1)
    feedback: str = Field(default="neutral", min_length=1, max_length=20)


@app.get("/")
def root():
    return {"status": "ok", "message": "Course Recommendation API running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/recommend")
def recommend(user: UserInput):
    try:
        return recommend_courses(user.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(exc)}") from exc


@app.post("/feedback")
def feedback(payload: FeedbackPayload):
    try:
        store_feedback(
            payload.user_input.dict(),
            payload.recommended_course,
            payload.feedback,
        )
        return {"status": "feedback recorded"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(exc)}") from exc


@app.post("/recommend-ab")
def recommend_ab(user: UserInput):
    try:
        return ab_predict(user.dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"A/B recommendation failed: {str(exc)}") from exc
