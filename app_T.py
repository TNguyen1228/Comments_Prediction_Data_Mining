import joblib
import numpy as np
import streamlit as st
import re
from credentials import client_id, client_secret, username, password, user_agent
from crawl.crawl import crawl_and_add_binary_features
import praw
import sys #lead to model_path
from Models.random_forest import RandomForestRegressor

sys.path.append('Models')
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     username=username,
                     password=password,
                    user_agent= user_agent)

def get_subreddit_name(url):
    regex = r"https:\/\/www\.reddit\.com\/r\/([^/]+)"
    match = re.search(regex, url)
    return match.group(1) if match else None

# model list
models = {
    "Random Forest Regressor":"Random Forest Regressor",
    "Decision Tree Regressor":"Decision Tree Regressor",
    "Poisson regression":"Possion Regression",
    #add more models
}

#  Title
st.markdown("<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'>", unsafe_allow_html=True)
st.markdown("<h1><i class='fa fa-comment'></i> Comments Prediction</h1>", unsafe_allow_html=True)

# Sidebar options
st.sidebar.title('Options')
url = st.sidebar.text_input("Enter URL")

# sidebar actions
selected_model = st.sidebar.selectbox("Choose model", list(models.keys()))
selected_model_name = models[selected_model]

# Model config
@st.cache(allow_output_mutation=True)  # Cache (faster process)
def load_model(model_name):
    return joblib.load(f'Models/{model_name.lower().replace(" ", "_")}_model.joblib')

if url :
    sub_reddit_name=get_subreddit_name(url=url)
    subreddit = reddit.subreddit(sub_reddit_name)
    st.write("Subreddit name: " + sub_reddit_name)
    
    input_model = crawl_and_add_binary_features(url)

    st.write(f"Data:\n")
    
    st.write(input_model)

    model = load_model(selected_model)

    st.write("Number of comments in the next 24 hours: " + str(np.round(model.predict(input_model)).astype(int)))

