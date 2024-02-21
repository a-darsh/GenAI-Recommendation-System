import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_preprocessed_data(filename):
    """Load the preprocessed dataset."""
    return pd.read_csv(filename)

def perform_eda(df):
    """Perform exploratory data analysis on the dataset."""
    # Setting up the visualisation settings
    sns.set(style="whitegrid")
    
    # Distribution of ratings
    plt.figure(figsize=(8, 6))
    sns.countplot(x='rating', data=df, palette='viridis')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()

    # Number of ratings per user
    plt.figure(figsize=(10, 6))
    ratings_per_user = df.groupby('user_id').size()
    sns.histplot(ratings_per_user, bins=30, kde=True, color='blue')
    plt.title('Number of Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    # Number of ratings per product
    plt.figure(figsize=(10, 6))
    ratings_per_product = df.groupby('product_id').size()
    sns.histplot(ratings_per_product, bins=30, kde=True, color='green')
    plt.title('Number of Ratings per Product')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Products')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    # Average rating per user
    plt.figure(figsize=(10, 6))
    avg_rating_per_user = df.groupby('user_id')['rating'].mean()
    sns.histplot(avg_rating_per_user, bins=30, kde=True, color='orange')
    plt.title('Average Rating per User')
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Users')
    plt.show()

    # Average rating per product
    plt.figure(figsize=(10, 6))
    avg_rating_per_product = df.groupby('product_id')['rating'].mean()
    sns.histplot(avg_rating_per_product, bins=30, kde=True, color='red')
    plt.title('Average Rating per Product')
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Products')
    plt.show()

if __name__ == "__main__":
    # Load the dataset (ensure you've updated the path to where your preprocessed dataset is located)
    df = load_preprocessed_data('preprocessed_data.csv')
    # Perform EDA
    perform_eda(df)
