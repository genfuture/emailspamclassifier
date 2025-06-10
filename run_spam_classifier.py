# -*- coding: utf-8 -*-
"""
Kaggle Dataset Downloader and Spam Classifier Pipeline
"""
import os
import boto3
import kagglehub
import pandas as pd
from botocore.exceptions import ClientError
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, length, trim, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import tempfile

# --- Configuration ---
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
RAW_BASE_PATH = f"s3a://{S3_BUCKET_NAME}/spam-dataset/"
OUTPUT_PATH = f"s3a://{S3_BUCKET_NAME}/outputs/"

# --- Kaggle Dataset Configuration ---
KAGGLE_DATASETS = [
    {
        "handle": "rtatman/enron-email-dataset",
        "file": "emails.csv",
        "s3_folder": "enron-email-dataset",
        "col_map": {"message": "text"}
    },
    {
        "handle": "rtatman/deceptive-opinion-spam-corpus",
        "file": "deceptive-opinion.csv",
        "s3_folder": "deceptive-opinion-spam-corpus",
        "col_map": {"deceptive": "label", "text": "text"}
    },
    {
        "handle": "uciml/sms-spam-collection-dataset",
        "file": "spam.csv",
        "s3_folder": "spam-or-not-spam-dataset",
        "col_map": {"v2": "text", "v1": "label"}
    },
    {
        "handle": "balaka18/email-spam-classification-dataset",
        "file": "spam_ham_dataset.csv",
        "s3_folder": "spam-mails-dataset",
        "col_map": {"text": "text", "label_num": "label"}
    },
    {
        "handle": "karthickveerakumar/spam-filter",
        "file": "emails.csv",
        "s3_folder": "email-spam-dataset",
        "col_map": {"Body": "text", "Label": "label"}
    }
]

# --- Helper Functions ---
def upload_to_s3(local_path, s3_key):
    """Uploads a local file to a specified S3 key."""
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    try:
        s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        print(f"Uploaded {local_path} to s3://{S3_BUCKET_NAME}/{s3_key}")
    except ClientError as e:
        print(f"S3 upload failed: {e}")

def download_and_upload_datasets():
    """Downloads datasets from Kaggle Hub and uploads to S3"""
    print("Starting Kaggle dataset download and upload process...")
    
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    for dataset in KAGGLE_DATASETS:
        handle = dataset["handle"]
        file_name = dataset["file"]
        s3_folder = dataset["s3_folder"]
        s3_key = f"spam-dataset/{s3_folder}/{file_name}"
        
        print(f"\nProcessing dataset: {handle}")
        
        try:
            # Download from Kaggle Hub
            print(f"Downloading {file_name} from Kaggle Hub...")
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = kagglehub.model_download(
                    handle,
                    file_name,
                    username=KAGGLE_USERNAME,
                    key=KAGGLE_KEY,
                    path=tmpdir
                )
                print(f"Downloaded to: {local_path}")
                
                # Upload to S3
                s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
                print(f"Uploaded to s3://{S3_BUCKET_NAME}/{s3_key}")
                
        except Exception as e:
            print(f"Failed to process {handle}: {str(e)}")
    
    print("\nAll datasets uploaded to S3 successfully!")

def load_source_df(spark, folder, filename, col_map, sample_limit=None, sample_size=None):
    """Robust dataset loader with schema handling and sampling"""
    s3_path = f"{RAW_BASE_PATH}{folder}/{filename}"
    
    try:
        df = spark.read.option("header", "true").csv(s3_path)
        print(f"Loaded: {s3_path} | Columns: {df.columns}")

        # Handle column name conflicts
        target_cols = set(col_map.values())
        for tgt_col in target_cols:
            if tgt_col in df.columns and tgt_col not in col_map.keys():
                df = df.drop(tgt_col)
        
        # Apply column mappings
        for src_col, tgt_col in col_map.items():
            if src_col in df.columns:
                df = df.withColumnRenamed(src_col, tgt_col)
        
        # Verify required columns
        if "text" not in df.columns or "label" not in df.columns:
            missing = [col for col in ["text", "label"] if col not in df.columns]
            print(f"⚠️ Missing columns {missing} in {s3_path}")
            return None
        
        # Apply sampling
        if sample_limit and sample_limit > 0:
            df = df.limit(sample_limit)
            if sample_size and 0 < sample_size <= sample_limit:
                fraction = sample_size / sample_limit
                df = df.sample(False, fraction, seed=42)
        
        return df.select("text", "label")
    
    except Exception as e:
        print(f"🚨 Error loading {s3_path}: {str(e)}")
        return None

def run_training_pipeline():
    """Main training pipeline execution"""
    # Initialize Spark with enhanced configuration
    spark = (
        SparkSession.builder
        .appName("SpamClassifierPipeline")
        .config("spark.jars.packages", 
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "com.amazonaws:aws-java-sdk-bundle:1.12.262")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .enableHiveSupport()
        .getOrCreate()
    )
    
    # Set log level to WARN to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    print("SparkSession initialized with S3 support")

    # Dataset configuration with schema mappings
    datasets = [
        ("enron-email-dataset", "emails.csv", {"message": "text"}, 0, 50000),
        ("deceptive-opinion-spam-corpus", "deceptive-opinion.csv", {"deceptive": "label", "text": "text"}, None, None),
        ("spam-or-not-spam-dataset", "spam.csv", {"v2": "text", "v1": "label"}, None, None),
        ("spam-mails-dataset", "spam_ham_dataset.csv", {"text": "text", "label_num": "label"}, None, None),
        ("email-spam-dataset", "emails.csv", {"Body": "text", "Label": "label"}, None, None),
    ]

    # Load and combine datasets
    all_dfs = []
    for dataset in datasets:
        folder, filename, col_map, sample_limit, sample_size = dataset
        df = load_source_df(spark, folder, filename, col_map, sample_limit, sample_size)
        if df:
            row_count = df.count()
            all_dfs.append(df)
            print(f"✅ Added {row_count} rows from {folder}/{filename}")

    if not all_dfs:
        raise RuntimeError("No datasets loaded - check configurations and S3 paths")
    
    # Combine datasets with schema validation
    combined_df = all_dfs[0]
    for df in all_dfs[1:]:
        combined_df = combined_df.union(df)
    
    total_rows = combined_df.count()
    print(f"\n📊 TOTAL DATASET SIZE: {total_rows:,} emails")
    
    # --- Data Cleaning ---
    print("\n🧹 Cleaning data...")
    cleaned_df = (
        combined_df
        .withColumn("text", lower(trim(regexp_replace(col("text"), r"[^a-zA-Z0-9\s]", " "))))
        .withColumn("text", regexp_replace(col("text"), r"\s+", " "))
        .filter(length(col("text")) > 10)  # Remove empty/short texts
        .withColumn("label", 
                    when(col("label").cast("string").rlike("spam|1"), 1)
                    .otherwise(0))
        .filter(col("label").isin([0, 1]))
        .dropDuplicates(["text"])
    )
    
    cleaned_df.cache()
    final_count = cleaned_df.count()
    print(f"✅ Cleaned data: {final_count:,} rows ({total_rows - final_count:,} removed)")
    
    # --- EDA: Class Distribution ---
    class_dist = cleaned_df.groupBy("label").count().toPandas()
    plt.figure(figsize=(10, 6))
    bars = plt.bar(["Ham (0)", "Spam (1)"], class_dist["count"], color=["#4CAF50", "#F44336"])
    plt.title("Email Class Distribution", fontsize=14)
    plt.ylabel("Count", fontsize=12)
    
    # Add counts on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,}', ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3)
    dist_path = "/tmp/class_distribution.png"
    plt.savefig(dist_path)
    plt.close()
    upload_to_s3(dist_path, "outputs/eda/class_distribution.png")
    
    # --- Store Cleaned Data ---
    transformed_path = f"{OUTPUT_PATH}transformed/cleaned_emails/"
    print(f"\n💾 Saving cleaned data to: {transformed_path}")
    (cleaned_df.write
               .mode("overwrite")
               .parquet(transformed_path))
    
    # Create Glue Catalog
    db_name = "spam_ml_catalog"
    table_name = "cleaned_spam_emails"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    spark.sql(f"USE {db_name}")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    spark.sql(f"""
        CREATE EXTERNAL TABLE {table_name} (text STRING, label INT)
        STORED AS PARQUET LOCATION '{transformed_path}'
    """)
    print(f"📋 Glue table created: {db_name}.{table_name}")
    
    # --- Model Training ---
    print("\n🔨 Preparing model pipeline...")
    df = spark.table(f"{db_name}.{table_name}")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # ML Pipeline Components
    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    stop_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    nb = NaiveBayes(featuresCol="features", labelCol="label", smoothing=1.0)
    
    pipeline = Pipeline(stages=[tokenizer, stop_remover, hashing_tf, idf, nb])
    
    print("🏋️ Training model (this may take several minutes)...")
    model = pipeline.fit(train_df)
    print("🎉 Model training complete!")
    
    # --- Model Evaluation ---
    print("\n🧪 Evaluating model performance...")
    predictions = model.transform(test_df)
    
    # Calculate metrics
    accuracy_eval = MulticlassClassificationEvaluator(metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(metricName="f1")
    auc_eval = BinaryClassificationEvaluator()
    
    accuracy = accuracy_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)
    auc = auc_eval.evaluate(predictions)
    
    print(f"📊 Model Metrics:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    print(f"  - ROC AUC:  {auc:.4f}")
    
    # Confusion Matrix Visualization
    conf_matrix = (predictions
                  .groupBy("label", "prediction")
                  .count()
                  .orderBy("label", "prediction")
                  .toPandas())
    
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix.pivot("label", "prediction", "count").fillna(0),
               cmap="Blues", interpolation='nearest')
    plt.colorbar()
    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Ham", "Spam"])
    plt.yticks([0, 1], ["Ham", "Spam"])
    
    for i in range(2):
        for j in range(2):
            count = conf_matrix[(conf_matrix.label == i) & 
                              (conf_matrix.prediction == j)]["count"].values
            if count.size > 0:
                plt.text(j, i, f"{count[0]:,}", 
                        ha="center", va="center", color="black", fontsize=15)
    
    matrix_path = "/tmp/confusion_matrix.png"
    plt.savefig(matrix_path)
    plt.close()
    upload_to_s3(matrix_path, "outputs/evaluation/confusion_matrix.png")
    
    # --- Save Model ---
    model_path = f"{OUTPUT_PATH}models/spam_classifier_v1"
    print(f"\n💾 Saving model to: {model_path}")
    model.write().overwrite().save(model_path)
    
    # --- Sample Predictions ---
    print("\n🔮 Sample Predictions:")
    samples = [
        ("WINNER!! You've been selected for a $1000 Walmart gift card. Claim now!",),
        ("Hi Alex, the meeting is rescheduled to 3 PM tomorrow. Bring the Q2 reports.",),
        ("URGENT: Your bank account requires verification. Click here to secure it!",),
        ("Your Amazon order #42-665 has shipped. Track your package: [link]",)
    ]
    sample_df = spark.createDataFrame(samples, ["text"])
    model.transform(sample_df).select("text", "prediction").show(truncate=50)
    
    spark.stop()
    print("\n✅ Training pipeline completed successfully!")

def main():
    """Main execution flow"""
    if not all([S3_BUCKET_NAME, KAGGLE_USERNAME, KAGGLE_KEY]):
        raise ValueError("Required environment variables not set: "
                         "S3_BUCKET, KAGGLE_USERNAME, KAGGLE_KEY")
    
    # Step 1: Download datasets from Kaggle Hub and upload to S3
    download_and_upload_datasets()
    
    # Step 2: Run the training pipeline
    run_training_pipeline()
    
    print("\n🚀 All processes completed successfully!")

if __name__ == "__main__":
    main()
