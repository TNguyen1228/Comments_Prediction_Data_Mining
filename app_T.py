import joblib
import numpy as np
import streamlit as st
from credentials import client_id, client_secret, username, password, user_agent
from crawl.crawl import crawl_and_add_binary_features
import praw
import prawcore
import sys #lead to model_path
from Models.random_forest import RandomForestRegressor

sys.path.append('Models')
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     username=username,
                     password=password,
                    user_agent= user_agent)

@st.cache_data(ttl=600)  # Do not re-crawl the same post on every rerun
def crawl_post(url):
    # praw raises InvalidURL if the URL is not a Reddit post URL
    sub_reddit_name = reddit.submission(url=url).subreddit.display_name
    return sub_reddit_name, crawl_and_add_binary_features(url)

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
@st.cache_resource  # Cache (faster process)
def load_model(model_name):
    return joblib.load(f'Models/{model_name.lower().replace(" ", "_")}_model.joblib')

if url:
    try:
        sub_reddit_name, input_model = crawl_post(url)

        st.write("Subreddit name: " + sub_reddit_name)
        st.write(f"Data:\n")
        st.write(input_model)

        model = load_model(selected_model)
        st.write("Number of comments in the next 24 hours: " + str(np.round(model.predict(input_model)).astype(int)))

    except praw.exceptions.InvalidURL:
        st.error("Invalid URL. Please enter a valid Reddit post URL.")
    except (praw.exceptions.PRAWException, prawcore.exceptions.PrawcoreException) as e:
        st.error("Error accessing the subreddit. Please check the URL and try again.")
