import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

def analyze_sentiment_keywords(df, title_prefix="",exportName=""):
    """
    Analyze sentiment in the given DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'sentiment' and 'processed_review' columns.
        title_prefix (str): Optional prefix for plot titles (e.g., dataset name).
    """
    
    # Filter positive and negative reviews
    positive_reviews = df[df['sentiment'] == 'positive']['processed_review']
    negative_reviews = df[df['sentiment'] == 'negative']['processed_review']

    # Extract keywords from positive reviews
    vectorizer_pos = TfidfVectorizer(max_features=10)
    X_pos = vectorizer_pos.fit_transform(positive_reviews)
    print(f"Top Keywords in {title_prefix} Positive Reviews:", vectorizer_pos.get_feature_names_out())

    # Extract keywords from negative reviews
    vectorizer_neg = TfidfVectorizer(max_features=10)
    X_neg = vectorizer_neg.fit_transform(negative_reviews)
    print(f"Top Keywords in {title_prefix} Negative Reviews:", vectorizer_neg.get_feature_names_out())

    # Sentiment distribution
    df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title(f'{title_prefix} Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    # Export
    output_path=f"../output/{exportName}_Sentiment.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)

    plt.show()

    

    # Word cloud for positive reviews
    positive_text = ' '.join(positive_reviews)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{title_prefix} Word Cloud for Positive Reviews')

    # Export
    output_path=f"../output/{exportName}_Word_cloud.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)

    plt.show()

# Preprocessing function
def preprocess_text(text: str) -> str:
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# TextBlob sentiment
def get_sentiment(text: str) -> str:
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# VADER sentiment
sia = SentimentIntensityAnalyzer()

def get_vader_sentiment(text: str) -> str:
    scores = sia.polarity_scores(text)
    if scores['compound'] > 0.05:
        return 'positive'
    elif scores['compound'] < -0.05:
        return 'negative'
    else:
        return 'neutral'