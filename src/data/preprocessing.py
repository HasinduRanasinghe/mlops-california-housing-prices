import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, data_path='data/raw/california_housing.csv'):
        # Ensure absolute path
        self.data_path = os.path.abspath(data_path)

        # Check if file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}. Run download script first.")

        # Load data
        self.data = pd.read_csv(self.data_path)

    def preprocess(self, test_size=0.2, random_state=42):
        """
        Preprocess the data
        """
        # Ensure processed directory exists
        os.makedirs('data/processed', exist_ok=True)

        # Separate features and target
        X = self.data.drop('target', axis=1)
        y = self.data['target']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save processed data
        np.save('data/processed/X_train.npy', X_train_scaled)
        np.save('data/processed/X_test.npy', X_test_scaled)
        np.save('data/processed/y_train.npy', y_train)
        np.save('data/processed/y_test.npy', y_test)

        print("Data processed and saved in data/processed/")

        return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    preprocessor = DataPreprocessor()
    preprocessor.preprocess()


if __name__ == "__main__":
    main()
