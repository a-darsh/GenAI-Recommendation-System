import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
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
def build_ncf_model(num_users, num_items, embedding_size=50):
    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    
    # Embedding layers
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_size, name='item_embedding')(item_input)
    
    # Flatten the embeddings
    user_vector = Flatten(name='flatten_users')(user_embedding)
    item_vector = Flatten(name='flatten_items')(item_embedding)
    
    # Concatenate the embeddings
    concat = Concatenate()([user_vector, item_vector])
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu')(concat)
    dropout1 = Dropout(0.2)(fc1)
    fc2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(fc2)
    output = Dense(1)(dropout2)
    
    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    
    return model

if __name__ == "__main__":
    file_path = 'amazon_ratings.csv'  
    df = load_dataset(file_path)
    df, num_users, num_items = preprocess_data(df)
    
    X = [df['user'].values, df['item'].values]
    y = df['rating'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_ncf_model(num_users, num_items)
    model.fit([X_train[0], X_train[1]], y_train, epochs=5, batch_size=64, validation_split=0.1)
