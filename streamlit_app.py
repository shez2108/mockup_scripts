import streamlit as st
import json
from openai import OpenAI
import pandas as pd
from collections import Counter

st.set_page_config(
    page_title="Query-based Matching for Chat SEO",
    layout="centered"
)

api_key = st.secrets["OPENAI_API_KEY"]

# Main title with icon
st.title("QMatch")

# Footer note in an info box
st.info("""
    ℹ️ **Purpose**: This tool summarises brand mentions and product sentiment in ChatGPT. 
""")

client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)
