import os

import requests
import streamlit as st

API_URL = os.getenv("COURSE_API_URL", "http://localhost:8000")
DOMAINS = ["Data Science", "Web Development", "Backend", "Cloud"]

st.set_page_config(page_title="Course Recommendation System", page_icon="🎓")
st.title("🎓 Course Recommendation System")

with st.form("recommendation_form"):
    col1, col2 = st.columns(2)

    with col1:
        user_id = st.number_input("User ID", min_value=1, value=1, step=1)
        age = st.slider("Age", 18, 60, 25)
        experience = st.slider("Experience (Years)", 0, 20, 2)

    with col2:
        interest = st.slider("Interest Level", 1, 10, 8)
        domain = st.selectbox("Preferred Domain", DOMAINS)

    submitted = st.form_submit_button("Get Recommendation", use_container_width=True)

if submitted:
    payload = {
        "user_id": user_id,
        "age": age,
        "experience": experience,
        "interest_level": interest,
        "preferred_domain": domain,
    }

    try:
        with st.spinner("Getting your recommendation..."):
            response = requests.post(f"{API_URL}/recommend", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

        course = result.get("recommended_course", "Not available")
        st.success(f"Recommended Course: {course}")
    except requests.RequestException as exc:
        st.error(f"Failed to connect to the API: {exc}")
