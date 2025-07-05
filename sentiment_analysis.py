

import pandas as pd

# Load the training dataset
df = pd.read_csv("data/twitter_training.csv", header=None)

# View first few rows
print("✅ Dataset Loaded:")
print(df.head())
print("\n✅ Shape:", df.shape)
# Rename columns for clarity
df.columns = ["ID", "Entity", "Sentiment", "Text"]

# Check for missing values
print("\n✅ Missing values:\n", df.isnull().sum())

# View unique sentiment labels
print("\n✅ Unique sentiments:", df["Sentiment"].unique())

# Sentiment distribution
print("\n✅ Sentiment distribution:\n", df["Sentiment"].value_counts())
import matplotlib.pyplot as plt
import seaborn as sns

# Plot sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Sentiment", order=df["Sentiment"].value_counts().index, palette="pastel")
plt.title("Sentiment Distribution in Twitter Training Data")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.tight_layout()
plt.savefig("images/sentiment_distribution.png", dpi=300)
plt.show()
from wordcloud import WordCloud

# Combine all tweet text for each sentiment
sentiments = df["Sentiment"].unique()

for sentiment in sentiments:
    text = " ".join(df[df["Sentiment"] == sentiment]["Text"].astype(str))

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud - {sentiment} Tweets")
    plt.tight_layout()
    filename = f"images/wordcloud_{sentiment.lower()}.png"
    plt.savefig(filename, dpi=300)
    plt.show()
