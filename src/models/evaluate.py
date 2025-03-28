from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).numpy()

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
