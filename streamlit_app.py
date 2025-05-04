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

# take a query input
query = st.text_input('Type your primary LLM Query here:')

num_queries = st.number_input('How many queries do you want to check brand mentions across? (Up to 100)')


if num_queries > 100:
    st.write('Sorry, you can't look at more than 100 queries')
    num_queries = st.text_input('How many queries do you want to check brand mentions across? (Up to 100)')
    num_queries = int(num_queries)

st.write(f'Getting query results for {num_queries} queries.')

