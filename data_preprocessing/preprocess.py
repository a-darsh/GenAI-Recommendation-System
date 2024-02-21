import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_dataset(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """Basic cleaning including removing duplicates and handling missing values."""
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['userID', 'itemID', 'rating'], inplace=True)  # Assuming these columns must not be empty
    return df

def encode_categoricals(df, columns):
    """Encode categorical variables using Label Encoder."""
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def add_temporal_features(df, timestamp_col):
    """Add temporal features from a timestamp: hour of day and day of week."""
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['hour_of_day'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    # Convert to cyclic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    return df

def preprocess_text_data(df, text_column):
    """Placeholder for text preprocessing if needed. Could include:
       - Cleaning text data
       - Extracting features using NLP techniques like TF-IDF or embeddings from pre-trained models.
       This function will be expanded based on specific requirements."""
    # This is a placeholder. Implementation will depend on the specific NLP tasks and models used.
    return df

def preprocess_data(file_path, categorical_columns, timestamp_col, text_column=None):
    """Complete preprocessing pipeline."""
    df = load_dataset(file_path)
    df = clean_data(df)
    df, encoders = encode_categoricals(df, categorical_columns)
    df = add_temporal_features(df, timestamp_col)
    if text_column:
        df = preprocess_text_data(df, text_column)
    return df, encoders

if __name__ == "__main__":
    file_path = 'path_to_your_dataset.csv'  # Update with the actual path to your dataset
    categorical_columns = ['userID', 'itemID']  # Update as per your dataset
    timestamp_col = 'timestamp'  # Update if your timestamp column is named differently
    text_column = 'reviewText'  # Optional, use if you have textual data to preprocess

    processed_df, encoders = preprocess_data(file_path, categorical_columns, timestamp_col, text_column)
    print(processed_df.head())
    # Save the processed dataframe for further use
    processed_df.to_csv('processed_dataset.csv', index=False)
