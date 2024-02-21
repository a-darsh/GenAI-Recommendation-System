from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

import pandas as pd

# Load the dataset
def load_data(file_path):
    # Define the format
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    data = Dataset.load_from_file(file_path, reader=reader)
    return data

# User-Based Collaborative Filtering
def user_based_cf(trainset, testset):
    # Configure the algorithm to use user-based cosine similarity
    sim_options = {
        'name': 'cosine',
        'user_based': True  # compute similarities between users
    }
    algo = KNNBasic(sim_options=sim_options)

    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    rmse(predictions)
    return predictions

# Item-Based Collaborative Filtering
def item_based_cf(trainset, testset):
    # Configure the algorithm to use item-based cosine similarity
    sim_options = {
        'name': 'cosine',
        'user_based': False  # compute similarities between items
    }
    algo = KNNBasic(sim_options=sim_options)

    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    rmse(predictions)
    return predictions

if __name__ == "__main__":
    file_path = 'amazon_ratings.csv'  
    data = load_data(file_path)

    # Split the dataset into the trainset and the testset
    trainset, testset = train_test_split(data, test_size=0.25)

    print("User-Based Collaborative Filtering:")
    user_based_predictions = user_based_cf(trainset, testset)

    print("\nItem-Based Collaborative Filtering:")
    item_based_predictions = item_based_cf(trainset, testset)
