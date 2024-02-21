import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load dataset from Kaggle
data = pd.read_csv('amazon_reviews.csv')

# Data Preprocessing
data.drop_duplicates(subset=['user_id', 'item_id'], inplace=True)
data.dropna(subset=['user_id', 'item_id'], inplace=True)

# Train-test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Collaborative Filtering
def collaborative_filtering(train_data):
    user_item_matrix = train_data.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)
    user_sim_matrix = cosine_similarity(user_item_matrix)
    return user_sim_matrix

user_sim_matrix = collaborative_filtering(train_data)

# Load pre-trained GPT2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Fine-tune GPT2 model on the dataset
def fine_tune_gpt2(train_texts):
    inputs = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    labels = inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # Ignore loss on padding tokens
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for _ in range(3):  # Adjust the number of epochs
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Convert user-item interactions to text sequences
train_texts = []
for user_id, group in train_data.groupby('user_id'):
    items_interacted = group['item_id'].tolist()
    train_texts.append('User {} interacted with items: {}'.format(user_id, ', '.join(items_interacted)))

# Fine-tune GPT2 on the training data
fine_tune_gpt2(train_texts)

# Generate recommendations
def generate_recommendations(user_id, num_recommendations=5):
    user_text = 'User {} interacted with items:'.format(user_id)
    inputs = tokenizer(user_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(input_ids=inputs.input_ids, max_length=512, num_return_sequences=num_recommendations, early_stopping=True)
    recommended_items = []
    for output in outputs:
        recommended_items.append(tokenizer.decode(output, skip_special_tokens=True).split(':')[-1].strip())
    return recommended_items

# Model Evaluation
def evaluate_model(test_data):
    # Assuming 'user_id', 'item_id', and 'rating' columns in test_data
    true_ratings = test_data['rating']
    predicted_ratings = []
    for _, row in test_data.iterrows():
        recommendations = generate_recommendations(row['user_id'])
        if row['item_id'] in recommendations:
            predicted_ratings.append(5)  # Assuming the model recommends highly rated items
        else:
            predicted_ratings.append(1)  # Assuming the model does not recommend irrelevant items

    precision = precision_score(true_ratings, predicted_ratings, average='binary', pos_label=5)
    recall = recall_score(true_ratings, predicted_ratings, average='binary', pos_label=5)
    f1 = f1_score(true_ratings, predicted_ratings, average='binary', pos_label=5)
    return precision, recall, f1

# Example of generating recommendations for a user
user_id = 123
recommendations = generate_recommendations(user_id)
print("Recommended items for User {}: {}".format(user_id, recommendations))

# Example of evaluating the model
precision, recall, f1 = evaluate_model(test_data)
print("Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}".format(precision, recall, f1))
