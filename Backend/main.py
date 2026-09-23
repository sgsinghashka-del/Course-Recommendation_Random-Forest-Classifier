# Backend/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from Backend.recommender import recommend_courses
from Backend.feedback import store_feedback
from Backend.ab_testing import ab_predict

app = FastAPI(title="Course Recommendation API")


class UserInput(BaseModel):
    age: int
    experience: int
    interest_level: int
    preferred_domain: str


@app.get("/")
def root():
    return {"message": "Course Recommendation API running"}


@app.post("/recommend")
def recommend(user: UserInput):
    return recommend_courses(user.dict())

@app.post("/feedback")
def feedback(payload: dict):
    store_feedback(
        payload["user_input"],
        payload["recommended_course"],
        payload["feedback"]
    )
    return {"status": "feedback recorded"}

@app.post("/recommend-ab")
def recommend_ab(user: UserInput):
    return ab_predict(user.dict())