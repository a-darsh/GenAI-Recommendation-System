import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path, usecols=['userID', 'itemID', 'rating'])
    return df

# Preprocessing
def preprocess_data(df):
    user_ids = df['userID'].unique().tolist()
    user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_ids = df['itemID'].unique().tolist()
    item_id_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
    
    df['user'] = df['userID'].map(user_id_map)
    df['item'] = df['itemID'].map(item_id_map)
    
    num_users, num_items = len(user_id_map), len(item_id_map)
    return df, num_users, num_items

# Model definition
def build_autoencoder(num_items, encoding_dim=128):
    # Input layer
    input_layer = Input(shape=(num_items,))
    
    # Encoder
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    encoded = Dropout(0.2)(encoded)
    
    # Decoder
    decoded = Dense(num_items, activation='sigmoid')(encoded)
    
    # Autoencoder model
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
    
    return autoencoder

if __name__ == "__main__":
    file_path = 'amazon_ratings.csv'  
    df = load_dataset(file_path)
    df, num_users, num_items = preprocess_data(df)
    
    # Convert the ratings to a user-item matrix
    user_item_matrix = df.pivot(index='user', columns='item', values='rating').fillna(0)
    
    X_train, X_test = train_test_split(user_item_matrix, test_size=0.2, random_state=42)
    
    model = build_autoencoder(num_items)
    model.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_test, X_test))
