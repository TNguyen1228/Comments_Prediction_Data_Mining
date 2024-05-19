from credential import client_id, client_secret, username, password, user_agent
import praw
import csv

reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     username=username,
                     password=password,
                    user_agent= user_agent)

def get_post_urls(subreddit_name, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    post_urls = [submission.url for submission in subreddit.new(limit=limit)]
    return post_urls

subreddit_name = 'VietNam'
post_urls = get_post_urls(subreddit_name, limit=300)

from datetime import datetime, timedelta

def get_weekday_binary_features(date):
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = date.strftime("%A")
    return {f"Basetime Weekday {i+1}": int(weekday == day) for i, day in enumerate(weekdays)}

def get_post_weekday_binary_features(date):
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = date.strftime("%A")
    return {f"Post Weekday {i+1}": int(weekday == day) for i, day in enumerate(weekdays)}

def crawl_reddit_comments(post_url):
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

    return {
        "text": submission.selftext,
        "Total Comments Before Basetime": total_comments_before_basetime,
        "Comments in Last 24 Hours Before Basetime": comments_last_24h,
        "Comments Between T1 and T2": comments_T1_T2,
        "Comments in First 24 Hours After Post": comments_first_24h_after_post,
        "Different between attr2 and attr3": comments_last_24h - comments_T1_T2,
        "Time Difference (hours)": time_diff_hours,
        "Post Length": post_length,
        **basetime_binary_features,
        **post_binary_features
    }

def crawl_multiple_posts(post_urls):
    results = []
    for url in post_urls:
        try:
            result = crawl_reddit_comments(url)
            
            results.append(result)
        except Exception as e:
            print(f"Error crawling {url}: {e}")
    return results

def save_to_csv(data, filename='reddit_comments.csv'):
    keys = data[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def add_binary_features(file_path, output_file_path=None):
    # Load the dataset
    df = pd.read_csv(file_path)

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

    # Find the insertion point (after 'Post Length' column)
    insertion_index = df.columns.get_loc('Post Length') + 1

    # Split the original DataFrame into two parts and insert the new features
    df_part1 = df.iloc[:, :insertion_index]
    df_part2 = df.iloc[:, insertion_index:]

    # Combine the parts with the new binary features in between
    updated_df = pd.concat([df_part1, binary_features_df, df_part2], axis=1)

    # Save the updated dataset to a new CSV file without the header if an output file path is provided
    if output_file_path:
        updated_df.to_csv(output_file_path, index=False, header=False)
        print(f"Updated dataset saved to {output_file_path}")

    # Return the DataFrame without headers
    return updated_df.to_string(index=False, header=False)