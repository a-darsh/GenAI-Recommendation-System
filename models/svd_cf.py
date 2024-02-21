from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split
from surprise.accuracy import rmse

def load_data(file_path):
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    data = Dataset.load_from_file(file_path, reader=reader)
    return data

if __name__ == "__main__":
    file_path = 'amazon_ratings.csv'  
    data = load_data(file_path)
    
    # Split the dataset for evaluation
    trainset, testset = train_test_split(data, test_size=0.25)

    # Use the SVD algorithm.
    algo = SVD()

    # Train and test reporting the RMSE
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse(predictions)

    # Optionally, cross-validate (to evaluate and compare performance)
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
