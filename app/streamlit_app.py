"""
Streamlit UI. Run with: streamlit run app/streamlit_app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.predict import predict_text

st.set_page_config(page_title="SMS Spam Detector", page_icon="📵")
st.title("SMS Spam Detection")
st.write("Enter an SMS message below and the model will classify it as spam or ham.")

user_text = st.text_area("Message", value="", height=150)

if st.button("Predict", type="primary"):
    if not user_text.strip():
        st.warning("Please enter a message first.")
    else:
        try:
            result = predict_text(user_text)
        except FileNotFoundError:
            st.error("No trained model found. Run `python -m src.train` first.")
        else:
            if result["error"]:
                st.warning(result["error"])
            else:
                label = result["label_name"].upper()
                if label == "SPAM":
                    st.error(f"Prediction: {label}")
                else:
                    st.success(f"Prediction: {label}")
                if result["spam_probability"] is not None:
                    st.write(f"Estimated spam probability: {result['spam_probability']:.3f}")
                    st.caption(
                        "Note: this is the model's raw predicted probability, not a "
                        "calibrated confidence score."
                    )
