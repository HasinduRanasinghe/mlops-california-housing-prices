import os
from sklearn.datasets import fetch_california_housing


def download_california_housing(data_path='data/raw'):
    """
    Download California Housing dataset
    """
    # Ensure the directory exists
    os.makedirs(data_path, exist_ok=True)

    # Fetch dataset
    california_housing = fetch_california_housing(as_frame=True)

    # Convert to DataFrame
    df = california_housing.data.copy()
    df['target'] = california_housing.target

    # Full path for saving
    full_path = os.path.join(data_path, 'california_housing.csv')

    # Save to CSV
    df.to_csv(full_path, index=False)

    print(f"Dataset saved to {full_path}")
    return df


def main():
    download_california_housing()


if __name__ == "__main__":
    main()
