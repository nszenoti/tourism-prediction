import pandas as pd
from sklearn.model_selection import train_test_split
import os
from src.utils.hf_utils import HFHandler
from src.utils.logger import get_logger

logger = get_logger(__name__)

def prepare_data():
    # Load dataset
    DATASET_PATH = "hf://datasets/nszfgfg/tourism-prediction/tourism.csv"
    df = pd.read_csv(DATASET_PATH)
    logger.info("Dataset loaded successfully")

    # Data cleaning
    to_drop = ['Unnamed: 0', 'CustomerID']
    for col_name in to_drop:
        if col_name in df.columns:
            df.drop(columns=[col_name], inplace=True)
            logger.info(f"Dropped column: {col_name}")

    # Split data
    target_col = 'ProdTaken'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    logger.info("Data split completed")

    # Save splits temporarily and upload
    splits = {
        "Xtrain.csv": X_train,
        "Xtest.csv": X_test,
        "ytrain.csv": y_train,
        "ytest.csv": y_test
    }

    hf = HFHandler()

    for filename, data in splits.items():
        # Save locally
        data.to_csv(filename, index=False)

        # Upload to HF
        hf.upload(
            path=filename,
            repo_name="tourism-prediction",
            repo_type="dataset"
        )
        logger.info(f"Uploaded {filename} to HuggingFace")

        # Cleanup
        os.remove(filename)

    logger.info("Data preparation completed successfully")

if __name__ == "__main__":
    prepare_data()
