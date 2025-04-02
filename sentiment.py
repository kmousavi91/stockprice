import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

# Step 1: Set API Key and Retrieve Financial News Headlines
api_key = "a2e7ea53e4f74f4aa1cc85c3c9ea662d"
url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={api_key}"

response = requests.get(url)

if response.status_code == 200:
    articles = response.json().get("articles", [])
    news_headlines = [article["title"] for article in articles if article["title"]]
else:
    news_headlines = []

# Step 2: Perform Sentiment Analysis
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity  # Returns a score between -1 and 1

news_df = pd.DataFrame(news_headlines, columns=['Headline'])
news_df['Sentiment'] = news_df['Headline'].apply(analyze_sentiment)

# Step 3: Visualize Sentiment Distribution
plt.figure(figsize=(10, 5))
sns.histplot(news_df['Sentiment'], bins=20, kde=True, color='blue')
plt.title("Sentiment Distribution of Financial News")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.show()

# Step 4: Generate a Word Cloud
if not news_headlines:
    print("No valid news headlines to display.")
else:
    text = " ".join(news_df['Headline'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Financial News Headlines")
    plt.show()
