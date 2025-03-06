import streamlit as st
import random
import string
import time


# Add email input
st.title("Tool Name")
email = st.text_input("Enter your email address:")
password = st.text_input("Enter password:", type="password")
if password != "password":
    st.error("Incorrect password. Access denied.")
    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Run Domain Analysis", "Merge Data for Reports", "Search Knowledge Base"])


    st.write(
        "Input the domain and competitors of the site you'd like to analyse:"
    )

    domain1 = st.text_input('Enter the domain you want to analyse:')
    competitor_status = st.selectbox('Would you like to get competitor data?', ['No', 'Yes'])

    competitor_data = False
    domains = [domain1]
    if competitor_status == 'Yes':
        competitor_data = True
        num_competitors = st.number_input('How many competitors would you like to analyze?', min_value=1, max_value=5, value=2, step=1)
        domains = [domain1]  # Start with the main domain
        competitor_domains = []
        for i in range(5):
            key = f'competitor_{i}'
            if i < num_competitors:
                competitor_domain = st.text_input(f'Enter the domain of competitor {i+1}:', key=key)
                if competitor_domain:
                    domains.append(competitor_domain)
            else:
                st.text_input(f'Competitor {i+1}', value='', key=key, disabled=True, label_visibility='hidden')
        domains.extend(competitor_domains)
        if len(domains) > 1:
            competitors_str = ', '.join(domains[1:])
            st.write(f'Getting relevance data for {domain1} and its competitors: {competitors_str}')
        else:
            st.write('No valid competitor domains have been entered.')
    else:
        domains = [domain1]
    

    if st.button('Start Analysis'):
        st.write('Running Crawl for', domain1, domains)
        
