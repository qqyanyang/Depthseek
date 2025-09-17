import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import argparse
import warnings
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def load_and_preprocess_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def build_model(model_type):
    if model_type == 'lasso':
        model = Lasso(alpha=0.1, random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        model = XGBRegressor(random_state=42)
    elif model_type == 'svm':
        model = SVR(kernel='rbf')
    else:
        raise ValueError("Unsupported model type")
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train.ravel())
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_loss = mean_squared_error(y_train, train_pred)
    test_loss = mean_squared_error(y_test, test_pred)
    
    print(f"Train Loss: {train_loss}")
    print(f"Test Loss: {test_loss}")
    return train_loss, test_loss

def custom_function(x, a, m):
    return m - m * np.exp(-((1000000*x)**a)/m)

def estimate_parameters(X, y, y_max):
    try:
        initial_guess = [0.95, y_max*0.9]
        bounds = ([0.8, 0.7*y_max], [1.2, np.inf]) 
        popt, _ = curve_fit(custom_function,
                           X.ravel(),
                           y.ravel(),
                           p0=initial_guess,
                           bounds=bounds,
                           method='trf',
                           maxfev=10000)
        a_est, m_est = popt
    except RuntimeError:
        a_est, m_est = 1.0, y_max

    return a_est, m_est

def main():
    parser = argparse.ArgumentParser(description='Fit and predict using a dataset.')
    parser.add_argument('-i', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('-o', type=str, required=True, help='Output directory for results')
    args = parser.parse_args()
    
    os.makedirs(args.o, exist_ok=True)
    dataset_name = os.path.basename(args.i).replace('.txt', '')
    
    original_data = np.loadtxt(args.i)
    X_original = original_data[:, 0]
    y_original = original_data[:, 1]
    
    max_depth = int(np.max(X_original))
    y_max = np.max(y_original)
    
    a_est, m_est = estimate_parameters(X_original, y_original, y_max)
    
    predict_depths = np.arange(max_depth + 1, 401)
    
    predicted_values = custom_function(predict_depths, a_est, m_est)
    
    combined_depths = np.concatenate([X_original, predict_depths])
    combined_values = np.concatenate([y_original, predicted_values])
    
    output_data = np.column_stack((combined_depths, combined_values))
    
    output_file = os.path.join(args.o, f"{dataset_name}_prediction.txt")
    np.savetxt(output_file, output_data, fmt=['%.1f', '%.1f'], delimiter='\t')
    
    print(f"Prediction completed. Results saved to {output_file}")
    print(f"Original data points: {len(X_original)}")
    print(f"Predicted points: {len(predict_depths)}")
    print(f"Total points: {len(output_data)}")
    print(f"Fitted parameters: a={a_est:.3f}, m={m_est:.2f}")

if __name__ == "__main__":
    main()
