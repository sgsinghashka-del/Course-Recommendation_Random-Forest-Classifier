import streamlit as st
import requests
import pandas as pd

st.title("🎓 Course Recommendation System")

user_id = st.number_input("User ID", min_value=1)

age = st.slider("Age", 18, 60)
experience = st.slider("Experience (Years)", 0, 30)
interest = st.slider("Interest Level", 1, 10)
domain = st.selectbox("Preferred Domain", ["Data Science","Web Development","Backend","Cloud"])

payload = {
    "user_id": user_id,
    "features": {
        "age": age,
        "experience": experience,
        "interest_level": interest,
        "preferred_domain": domain
    }
}

if st.button("Get Recommendations"):
    res = requests.post("http://localhost:8000/recommend", json=payload)
    st.write(pd.DataFrame(res.json()["recommendations"], columns=["Course","Score"]))