import streamlit as st
import json
from openai import OpenAI
import pandas as pd
from collections import Counter

st.set_page_config(
    page_title="Query-based Matching for Chat SEO",
    layout="centered"
)

# Main title with icon
st.title("QMatch")
