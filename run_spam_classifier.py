# -*- coding: utf-8 -*-
"""
run_spam_classifier.py

This script automates the process of training a spam classification model using PySpark.
It performs the following steps:
1.  Initializes a SparkSession.
2.  Loads and combines multiple raw email datasets from an S3 bucket.
3.  Cleans and preprocesses the data.
4.  Performs Exploratory Data Analysis (EDA) and saves plots to S3.
5.  Writes the cleaned data to S3 in Parquet format and creates a Glue table.
6.  Trains a Naive Bayes classifier using a Spark ML Pipeline.
7.  Evaluates the model's performance and saves metrics/plots to S3.
8.  Saves the trained pipeline model to S3.
9.  Runs a final test prediction on sample texts.
"""
import os
import boto3
from botocore.exceptions import ClientError
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, length, trim, when, explode
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# --- Configuration ---
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET")
RAW_BASE_PATH = f"s3a://{S3_BUCKET_NAME}/spam-dataset/"
OUTPUT_PATH = f"s3a://{S3_BUCKET_NAME}/outputs/"

# --- Helper Functions ---

def upload_to_s3(local_path, s3_key):
    """Uploads a local file to a specified S3 key."""
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    try:
        s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        print(f"Successfully uploaded {local_path} to s3://{S3_BUCKET_NAME}/{s3_key}")
    except ClientError as e:
        print(f"Failed to upload {local_path} to S3: {e}")

def main():
    """Main function to run the entire pipeline."""
    if not S3_BUCKET_NAME:
        raise ValueError("S3_BUCKET_NAME environment variable not set. Please configure it.")

    # Corrected Code
    spark = (
        SparkSession.builder
        .appName("SpamClassifierPipeline")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .enableHiveSupport()
        .getOrCreate()
    )
    print("SparkSession created successfully.")

    # --- 1. Load and Combine Datasets ---
    datasets_config = [
        ("email-spam-dataset", None, {"Body": "text", "Label": "label"}),
        # Add other dataset configs here if needed, e.g.:
        # ("spam-or-not-spam-dataset", "spam_or_not_spam.csv", {"email": "text", "label": "label"}),
    ]

    all_dfs = []
    for folder, filename, col_map in datasets_config:
        s3_path = f"{RAW_BASE_PATH}{folder}/{filename or '*.csv'}"
        try:
            df = spark.read.option("header", "true").option("inferSchema", "true").csv(s3_path)
            for src_col, tgt_col in col_map.items():
                if src_col in df.columns:
                    df = df.withColumnRenamed(src_col, tgt_col)
            
            if "text" in df.columns and "label" in df.columns:
                all_dfs.append(df.select("text", "label"))
                print(f"Successfully loaded and mapped data from: {s3_path}")
            else:
                print(f"⚠️ SKIPPING '{s3_path}': missing 'text' or 'label' column after mapping.")

        except Exception as e:
            print(f"⚠️ FAILED to load from '{s3_path}': {e}")
    
    if not all_dfs:
        raise RuntimeError("No valid DataFrames were loaded. Check S3 paths and column mappings.")

    raw_df = all_dfs[0]
    for df in all_dfs[1:]:
        raw_df = raw_df.unionByName(df)

    print(f"Total raw rows combined: {raw_df.count()}")

    # --- 2. Data Cleaning and Transformation ---
    cleaned_df = (
        raw_df
        .withColumn("text", lower(col("text")))
        .withColumn("text", regexp_replace(col("text"), r"[^a-zA-Z0-9\s]", " "))
        .withColumn("text", regexp_replace(col("text"), r"\s+", " "))
        .withColumn("text", trim(col("text")))
        .filter((col("text").isNotNull()) & (length(col("text")) > 3))
        .withColumn("label", col("label").cast("integer"))
        .filter(col("label").isin([0, 1]))
        .na.drop(subset=["label"])
    )
    
    cleaned_df.cache()
    cleaned_count = cleaned_df.count()
    print(f"Total rows after cleaning and filtering for labels {{0, 1}}: {cleaned_count}")

    # --- 3. Exploratory Data Analysis (EDA) ---
    print("Starting Exploratory Data Analysis (EDA)...")
    eda_output_key = "outputs/eda/"

    # Class Distribution
    class_dist_pd = cleaned_df.groupBy("label").count().orderBy("label").toPandas()
    class_dist_pd["label_name"] = class_dist_pd["label"].map({0: "Ham", 1: "Spam"})
    plt.figure(figsize=(8, 5))
    plt.bar(class_dist_pd["label_name"], class_dist_pd["count"], color=['#1f77b4', '#ff7f0e'])
    plt.title('Email Class Distribution')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    class_dist_path = "/tmp/class_distribution.png"
    plt.savefig(class_dist_path)
    plt.close()
    upload_to_s3(class_dist_path, f"{eda_output_key}class_distribution.png")

    # --- 4. Write Cleaned Data to S3 and Create Glue Table ---
    transformed_path = f"{OUTPUT_PATH}transformed/raw_text_label/"
    print(f"Writing cleaned data to Parquet at {transformed_path}")
    cleaned_df.write.mode("overwrite").parquet(transformed_path)

    db_name = "spam_ml_catalog"
    table_name = "spam_emails"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    spark.sql(f"USE {db_name}")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    spark.sql(f"""
        CREATE EXTERNAL TABLE {table_name} (text STRING, label INT)
        STORED AS PARQUET LOCATION '{transformed_path}'
    """)
    print(f"Glue table `{db_name}.{table_name}` created successfully.")

    # --- 5. Model Training ---
    athena_df = spark.table(f"{db_name}.{table_name}")
    train_df, test_df = athena_df.randomSplit([0.8, 0.2], seed=42)

    print(f"Training data count: {train_df.count()}, Test data count: {test_df.count()}")

    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    stop_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    hashingTF = HashingTF(inputCol="filtered_tokens", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    nb_classifier = NaiveBayes(featuresCol="features", labelCol="label", smoothing=1.0)

    pipeline = Pipeline(stages=[tokenizer, stop_remover, hashingTF, idf, nb_classifier])

    print("Training the NaiveBayes pipeline model...")
    model = pipeline.fit(train_df)
    print("Model training complete.")

    # --- 6. Model Evaluation ---
    print("Evaluating the model...")
    predictions = model.transform(test_df)
    eval_output_key = "outputs/evaluation/"

    # Calculate metrics
    accuracy = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(metricName="f1").evaluate(predictions)
    auc = BinaryClassificationEvaluator().evaluate(predictions)
    print(f"Test Set Metrics: Accuracy = {accuracy:.4f}, F1-Score = {f1:.4f}, ROC AUC = {auc:.4f}")

    # Confusion Matrix
    conf_matrix_pd = predictions.crosstab("label", "prediction").toPandas()
    print("Confusion Matrix:\n", conf_matrix_pd)
    
    # Plot and save confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix_pd.iloc[:, 1:].values, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Ham (0)', 'Spam (1)'])
    plt.yticks([0, 1], ['Ham (0)', 'Spam (1)'])
    conf_matrix_path = "/tmp/confusion_matrix.png"
    plt.savefig(conf_matrix_path)
    plt.close()
    upload_to_s3(conf_matrix_path, f"{eval_output_key}confusion_matrix.png")

    # --- 7. Save Model ---
    model_s3_path = f"{OUTPUT_PATH}models/spam_nb_model"
    print(f"Saving pipeline model to {model_s3_path}")
    model.write().overwrite().save(model_s3_path)
    print("Model saved successfully.")

    # --- 8. Final Test with Sample Texts ---
    print("Running prediction on sample texts...")
    sample_texts = [
        ("Congratulations! You've won a free iPhone. Click here to claim your prize.",),
        ("Hi Sarah, I attached the Q2 report. Let me know if you have questions.",),
        ("URGENT: Your bank account has been compromised. Reply with your details.",),
        ("Meeting reminder: Project review at 3 PM today in Conference Room B",)
    ]
    sample_df = spark.createDataFrame(sample_texts, ["text"])
    sample_predictions = model.transform(sample_df)
    print("Sample Predictions (0=Ham, 1=Spam):")
    sample_predictions.select("text", "prediction").show(truncate=False)

    spark.stop()
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
