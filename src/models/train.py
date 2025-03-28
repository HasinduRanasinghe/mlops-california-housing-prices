import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.preprocessing import DataPreprocessor
from src.models.model import HousingPriceModel


def train_model(X_train, y_train, X_test, y_test, epochs=100, learning_rate=0.001):
    """
    Train neural network model for housing price prediction
    """
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train.to_numpy()).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test.to_numpy()).unsqueeze(1)

    # Initialize model, loss, and optimizer
    model = HousingPriceModel(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # MLflow tracking
    mlflow.set_experiment("California Housing Prices")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training loss
            mlflow.log_metric("train_loss", loss.item(), step=epoch)

        # Evaluate on test set
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)

            # Log test metrics
            mlflow.log_metric("test_loss", test_loss.item())

        return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    X_test = torch.FloatTensor(X_test)
    y_test = y_test.to_numpy()

    with torch.no_grad():
        y_pred = model(X_test).numpy()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }

    return metrics


def train_and_save_model(config_path=None):
    """
    Full pipeline to train and save model
    """
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Load and preprocess data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess()

    # Train model
    model = train_model(X_train, y_train, X_test, y_test)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print("Model Evaluation Metrics:", metrics)

    # Save model with error handling
    try:
        model_path = 'models/housing_price_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved successfully to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise


def main():
    train_and_save_model()


if __name__ == "__main__":
    main()
