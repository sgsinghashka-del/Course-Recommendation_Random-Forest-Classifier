import pandas as pd
from datetime import datetime
import os

FEEDBACK_FILE = "data/feedback.csv"

def store_feedback(user_input, recommended_course, feedback):
    row = {
        **user_input,
        "recommended_course": recommended_course,
        "feedback": feedback,
        "timestamp": datetime.utcnow()
    }

    df = pd.DataFrame([row])

    if os.path.exists(FEEDBACK_FILE):
        df.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(FEEDBACK_FILE, index=False)