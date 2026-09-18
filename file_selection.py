import streamlit as st
import pandas as pd


def uploaded_data():
    file = st.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx", "xls"])

    if file is None:
        return None

    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"I CAN'T READ THIS FILE: {e}")
        return None

    if df.empty:
        st.warning("YOU UPLOADED FILE IS EMPTY")
        return None

    return df