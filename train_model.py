import os
import sys
import traceback

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the necessary modules
from src.data.download import download_california_housing
from src.data.preprocessing import DataPreprocessor
from src.models.train import train_and_save_model


def main():
    try:
        # Step 1: Ensure data is downloaded
        if not os.path.exists('data/raw/california_housing.csv'):
            print("Downloading dataset...")
            download_california_housing()

        # Step 2: Preprocess data if not already processed
        if not (os.path.exists('data/processed/X_train.npy') and
                os.path.exists('data/processed/X_test.npy')):
            print("Preprocessing dataset...")
            preprocessor = DataPreprocessor()
            preprocessor.preprocess()

        # Step 3: Train and save model
        print("Starting model training...")
        train_and_save_model()
        print("Model training completed successfully!")

    except Exception as e:
        print(f"Error during model training:")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
