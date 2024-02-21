import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
# For image features, assuming you have pre-extracted features or use a pre-trained model
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def extract_basic_features(df):
    """Extract basic user and item features."""
    # User features
    df['user_avg_rating'] = df.groupby('userID')['rating'].transform('mean')
    df['user_rating_count'] = df.groupby('userID')['rating'].transform('count')
    
    # Item features
    df['item_avg_rating'] = df.groupby('itemID')['rating'].transform('mean')
    df['item_rating_count'] = df.groupby('itemID')['rating'].transform('count')
    
    return df

def extract_text_features(df, text_column):
    """Extract features from textual data using TF-IDF and sentiment analysis."""
    # TF-IDF for text column
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_result = tfidf.fit_transform(df[text_column].fillna('')).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns=[f'tfidf_{i}' for i in range(tfidf_result.shape[1])])
    df = pd.concat([df, tfidf_df], axis=1)
    
    # Sentiment analysis
    df['text_sentiment'] = df[text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    
    return df

# Placeholder function for image feature extraction
# def extract_image_features(df, image_column):
#     """Extract features from image data using a pre-trained CNN."""
#     # Load pre-trained ResNet50 model
#     model = ResNet50(weights='imagenet')
#     
#     # Assuming you have a way to load images from `image_column`
#     # and that they are stored as file paths
#     def get_image_features(img_path):
#         img = image.load_img(img_path, target_size=(224, 224))
#         img_array = image.img_to_array(img)
#         expanded_img_array = np.expand_dims(img_array, axis=0)
#         preprocessed_img = preprocess_input(expanded_img_array)
#         features = model.predict(preprocessed_img)
#         return features.flatten()
#     
#     # Apply feature extraction to each image
#     df['image_features'] = df[image_column].apply(get_image_features)
#     
#     return df

def perform_feature_engineering(df, text_column=None, image_column=None):
    """Orchestrate feature extraction process."""
    df = extract_basic_features(df)
    
    if text_column:
        df = extract_text_features(df, text_column)
    
    # Uncomment and customize if you need to extract image features
    # if image_column:
    #     df = extract_image_features(df, image_column)
    
    return df

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('path_to_processed_dataset.csv')  # Update with the actual path
    text_column = 'reviewText'  # Update based on your dataset
    # image_column = 'imagePath'  # Uncomment and update if you have image data
    
    df = perform_feature_engineering(df, text_column=text_column)
    # Save the dataframe with new features
    df.to_csv('features_enhanced_dataset.csv', index=False)
