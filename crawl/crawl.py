from credentials import client_id, client_secret, username, password, user_agent
import praw
from datetime import datetime, timedelta
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     username=username,
                     password=password,
                    user_agent= user_agent)

def get_post_urls(subreddit_name, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    post_urls = [submission.url for submission in subreddit.new(limit=limit)]
    return post_urls



def get_weekday_binary_features(date):
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = date.strftime("%A")
    return {f"Basetime Weekday {i+1}": int(weekday == day) for i, day in enumerate(weekdays)}

def get_post_weekday_binary_features(date):
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = date.strftime("%A")
    return {f"Post Weekday {i+1}": int(weekday == day) for i, day in enumerate(weekdays)}

def crawl_and_add_binary_features(post_url):

    # Fetch the post by URL
    submission = reddit.submission(url=post_url)
    submission.comments.replace_more(limit=None)  # Load all comments

    # Current time as basetime
    basetime_dt = datetime.now()

    # Time calculations
    T1 = basetime_dt - timedelta(hours=48)
    T2 = basetime_dt - timedelta(hours=24)
    post_time = datetime.fromtimestamp(submission.created_utc)
    time_diff_hours = (basetime_dt - post_time).total_seconds() / 3600  # Time difference in hours

    # Filter and count comments based on time criteria
    total_comments_before_basetime = sum(1 for c in submission.comments.list() if datetime.fromtimestamp(c.created_utc) < basetime_dt)
    comments_last_24h = sum(1 for c in submission.comments.list() if T2 <= datetime.fromtimestamp(c.created_utc) < basetime_dt)
    comments_T1_T2 = sum(1 for c in submission.comments.list() if T1 <= datetime.fromtimestamp(c.created_utc) < T2)
    comments_first_24h_after_post = sum(1 for c in submission.comments.list() if post_time <= datetime.fromtimestamp(c.created_utc) < post_time + timedelta(hours=24))

    # Length of the post
    post_length = len(submission.selftext)

    # Binary features for weekdays
    basetime_binary_features = get_weekday_binary_features(basetime_dt)
    post_binary_features = get_post_weekday_binary_features(post_time)

    # Create a DataFrame from the computed values
    data = {
        "text": [submission.selftext],
        "Total Comments Before Basetime": [total_comments_before_basetime],
        "Comments in Last 24 Hours Before Basetime": [comments_last_24h],
        "Comments Between T1 and T2": [comments_T1_T2],
        "Comments in First 24 Hours After Post": [comments_first_24h_after_post],
        "Different between attr2 and attr3": [comments_last_24h - comments_T1_T2],
        "Time Difference (hours)": [time_diff_hours],
        "Post Length": [post_length],
        **basetime_binary_features,
        **post_binary_features
    }

    df = pd.DataFrame(data)

    # Extract text data
    texts = df['text'].fillna('')

    # Drop the 'text' column from the original DataFrame
    df = df.drop('text', axis=1)

    # Initialize the CountVectorizer to get the 200 most frequent words
    vectorizer = CountVectorizer(max_features=200, binary=True)
    X = vectorizer.fit_transform(texts)

    # Get the feature names (the 200 most frequent words)
    feature_names = vectorizer.get_feature_names_out()

    # Create a DataFrame with the binary features
    binary_features_df = pd.DataFrame(X.toarray(), columns=feature_names)

    # Ensure binary_features_df has exactly 200 columns
    num_features = binary_features_df.shape[1]
    if num_features < 200:
        for i in range(200 - num_features):
            binary_features_df[f'extra_feature_{i}'] = 0

    # Find the insertion point (after 'Post Length' column)
    insertion_index = df.columns.get_loc('Post Length') + 1

    # Split the original DataFrame into two parts and insert the new features
    df_part1 = df.iloc[:, :insertion_index]
    df_part2 = df.iloc[:, insertion_index:]

    # Combine the parts with the new binary features in between
    updated_df = pd.concat([df_part1, binary_features_df, df_part2], axis=1)

    # Return the DataFrame 
    return updated_df

