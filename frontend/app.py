import json
from urllib import error, request

import pandas as pd
import streamlit as st


API_URL_DEFAULT = "http://127.0.0.1:8000/predict"

EMOJI_MAP = {
    "Low": "😊",
    "Moderate": "😐",
    "High": "😟",
}

COLOR_MAP = {
    "Low": "green",
    "Moderate": "#d4a017",
    "High": "red",
}

TIP_MAP = {
    "Low": "Keep up your preparation and maintain your confidence.",
    "Moderate": "Take short breaks and practice relaxation techniques.",
    "High": "Consider talking with a counselor or teacher and practice breathing exercises.",
}


def call_api(api_url: str, text: str):
    payload = json.dumps({"text": text}).encode("utf-8")
    req = request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=20) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


st.set_page_config(page_title="AI Exam Anxiety Detector", page_icon="🧠", layout="centered")
st.title("AI Exam Anxiety Detector")
st.write("Enter your exam-related thoughts, and the model will estimate your anxiety level.")

api_url = st.sidebar.text_input("FastAPI /predict URL", value=API_URL_DEFAULT)
user_text = st.text_area("How are you feeling about your exams?", height=180)

if st.button("Predict"):
    if not user_text.strip():
        st.error("Please enter some text before prediction.")
    else:
        try:
            result = call_api(api_url, user_text)
            level = result.get("anxiety_level", "Unknown")
            confidence = float(result.get("confidence", 0.0))
            scores = result.get("scores", {})

            emoji = EMOJI_MAP.get(level, "🤔")
            color = COLOR_MAP.get(level, "gray")
            tip = TIP_MAP.get(level, "Take care and keep preparing steadily.")

            st.markdown(
                f"<h3 style='color:{color};'>Prediction: {level} {emoji}</h3>",
                unsafe_allow_html=True,
            )
            st.write(f"Confidence: **{confidence:.2%}**")
            st.info(f"Calming tip: {tip}")

            if scores:
                chart_df = pd.DataFrame(
                    {
                        "Anxiety Level": list(scores.keys()),
                        "Score": list(scores.values()),
                    }
                ).set_index("Anxiety Level")
                st.subheader("Anxiety Score Distribution")
                st.bar_chart(chart_df)

        except error.HTTPError as ex:
            detail = ex.read().decode("utf-8")
            st.error(f"API error ({ex.code}): {detail}")
        except error.URLError as ex:
            st.error(f"Unable to connect to API: {ex.reason}")
        except Exception as ex:
            st.error(f"Unexpected error: {ex}")
