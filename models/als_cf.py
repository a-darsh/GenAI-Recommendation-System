from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def create_spark_session():
    return SparkSession.builder.appName("ALSExample").getOrCreate()

def load_data(spark, file_path):
    return spark.read.csv(file_path, header=True, inferSchema=True)

if __name__ == "__main__":
    spark = create_spark_session()
    file_path = 'amazon_ratings.csv'  
    
    # Load and prepare dataset
    df = load_data(spark, file_path)
    df = df.selectExpr("userID as user", "itemID as item", "rating as rating")
    
    # Split the data into training and test sets
    (training, test) = df.randomSplit([0.8, 0.2])

    # Build the recommendation model using ALS on the training data
    als = ALS(maxIter=5, regParam=0.01, userCol="user", itemCol="item", ratingCol="rating",
              coldStartStrategy="drop", nonnegative=True)
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))
