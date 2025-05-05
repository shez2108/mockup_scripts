import streamlit as st
import json
from openai import OpenAI
import pandas as pd
from collections import Counter
import json
from json import JSONDecodeError
from google.cloud import language_v1
import os 
from google.oauth2 import service_account
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt


api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit setup
st.set_page_config(page_title="Query-based Matching for Chat Engine SEO", layout="centered")
st.title("QMatch")
st.info("ℹ️ **Purpose**: This tool summarises brand mentions and product sentiment in ChatGPT.")

client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)

# take a query input
query = st.text_input('Type your primary LLM Query here:')

num_queries = st.number_input('How many queries? (Up to 100)', min_value=1, max_value=100, value=5)


if num_queries > 100:
    st.write('Error: Query Count exceeds maximum value.')
    num_queries = st.text_input('How many queries do you want to check brand mentions across? (Up to 100)')
    num_queries = int(num_queries)


json_format = {
    "type": "json_schema",
    "name": "query_response",
    "schema": {
        "type": "object",
        "properties": {
            "serps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "result": {"type": "string"},
                        "source": {"type": "string"},
                        "brand": {"type": "string"},
                        "product name" : {"type": "string"},
                        "urls": {"type": "string"}
                    },
                    "required": ["query", "result", "source", "brand", "product name", "urls"],
                    "additionalProperties": False
                }
            },
            "final_resolution": {"type": "string"}
        },
        "required": ["serps", "final_resolution"],
        "additionalProperties": False
    },
    "strict": True
}


#print(type(agg))

def get_query_response(query, num_queries):
    try:
        agg = num_queries - 1
        response = client.responses.create(
            model="gpt-4o-2024-08-06",
            tools=[{'type': 'web_search_preview'}],
            input=[
                {
                    'role': 'system',
                    'content': (
                        'You are part of a tool that generates related queries/prompts and their results based on '
                        'a user-inputted LLM prompt/query, so that we can see what the top product recommendations, deals and results are for each query.'
                    )
                },
                {
                    'role': 'user',
                    'content': (
                        f"Return the results of {num_queries} queries including the original query {query} and {agg} closely related search queries, including relevant brand mentions.\n"
                        "You should return an entry/row for each product recommendation that shows up as a result of a query. One query therefore may have mutliple rows."
                        "For example, if a query for cheap phone deals under 500 dollars returns 5 different product recommendations, each product should have its own row."
                        "and the 'results' property should resemble the typical output of searchGPT after a user prompt "
                        "including the brand mentions with the sources and urls you got the information from. "
                        "The other properties break down the results. For instance "
                        " - 'query' should be the query that returned this particular query result / product recommendation."
                        " - 'sources' show which sources chatgpt got the info for "
                        " - 'brands' show which brands are mentioned in the results "
                        " - 'products' shows which product names are being mentioned in the results for those brands "
                        " - 'urls' gives the specific urls of the sources. "
                        " if you are returning the results of 10 queries, and there are 5 product recommendations per query. there should be a total of 50 entries "
                    )
                }
            ],
            text={
                "format": json_format
            }
        )
        event = json.loads(response.output_text)
        return event
    except JSONDecodeError:
        st.error("❌ JSON parsing failed. Try again.")
        return None


# Define a function to get sentiment
def get_sentiment(text, lang_client):
    if pd.isnull(text):
        return None, None
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = lang_client.analyze_sentiment(request={"document": document})
    sentiment = response.document_sentiment
    return sentiment.score, sentiment.magnitude
def safe_get_sentiment(text, lang_client):
    try:
        return get_sentiment(text, lang_client) 
    except Exception as e:
        st.warning(f"Sentiment error: {e}")
        return None, None 

if st.button("Search") and query:
    time.sleep(1)
    st.write(f'Getting query results for {num_queries} queries.')
    time.sleep(5)
    try:
        total_queries = get_query_response(query, num_queries)
        if total_queries:
            df = pd.json_normalize(total_queries['serps'])
            # Apply the function to the 'result' column
            time.sleep(5)
            df['brand_product'] = df['brand'] + ' ' + df['product name']
            time.sleep(1)
            st.dataframe(df)
            st.download_button("⬇️ Download CSV", df.to_csv(index=False), "qmatch_output.csv", "text/csv")
            # Brand mentions chart
            st.subheader("Brand Mentions")
            st.bar_chart(df["brand"].value_counts())
            # Word cloud of query text
            st.subheader("Query Word Cloud")
            combined_queries = " ".join(df["query"].dropna().unique())
            wc = WordCloud(width=800, height=400, background_color="white").generate(combined_queries)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            # Sentiment toggle
            if st.radio("Run sentiment analysis?", ["No", "Yes"]) == "Yes":
                creds = service_account.Credentials.from_service_account_info(dict(st.secrets["GOOGLE_CREDENTIALS"]))
                lang_client = language_v1.LanguageServiceClient(credentials=creds)
                df[["sentiment_score", "sentiment_magnitude"]] = df["result"].apply(
                    lambda x: pd.Series(safe_get_sentiment(x, lang_client))
                )
                st.dataframe(df[["result", "sentiment_score", "sentiment_magnitude"]])
        else:
            st.error('No output received from response. Try searching again or refreshing the page.')
            pass
    except JSONDecodeError as e:
        st.error("JSON parsing failed. Refresh the page and try again.")
        #st.text(f"Raw output: {response.output_text[:500]}")  # Optional: log or show snippet
        event = None
