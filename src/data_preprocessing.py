import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from logger import get_logger  

logger = get_logger(__name__)

def clean_Export(df: pd.DataFrame, app_name: str) -> pd.DataFrame:
    logger.info(f"Loading data for {app_name}")
    df = df.drop_duplicates()
    df = df.dropna(subset=['review_text'])
    # Convert to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].dt.normalize()  # Keeps datetime dtype but zeroes out time
    print(df.info())
    df.to_csv('../data/'+app_name+'.csv', index=False)
    return df

def combine_reviews(file_map: dict) -> pd.DataFrame:
    all_reviews = []
    for app_name, file_path in file_map.items():
        df = load_and_clean(file_path, app_name)
        all_reviews.append(df)
    combined_df = pd.concat(all_reviews, ignore_index=True)
    return combined_df

def inspect_reviews(df: pd.DataFrame):
    """
    Cleans the DataFrame by removing empty or missing review_text,
    shows duplicates, and visualizes duplicate counts.
    """
    # Show rows with missing or empty review text
    empty_rows = df[df['review_text'].isna() | (df['review_text'].str.strip() == '')]
    print(f"Empty or missing review_text rows:\n{empty_rows}")

    # Show and analyze duplicates
    duplicates = df[df.duplicated(subset=['review_text'], keep=False)]
    print(f"Duplicate rows based on review_text:\n{duplicates}")

    # Plot number of times each duplicate appears
    duplicates_counts = duplicates['review_text'].value_counts()

    if not duplicates_counts.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=duplicates_counts.values, y=duplicates_counts.index)
        plt.title('Number of Duplicates Count Per Duplicate Value')
        plt.xlabel('Duplicates Count')
        plt.ylabel('Duplicate Value')
        plt.tight_layout()
        plt.show()
    else:
        print("No duplicate review_text values to plot.")