import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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

def predict(model, X_test, scaler_y):
    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    return y_pred

def custom_function_1(x, m):
    return m - m * np.exp(-1000000 * x / m)

def custom_function_2(x, a, m):
    return m - m * np.exp(-((1000000*x)**a) / m)

def estimate_parameters(X, y, y_max):
    try:
        initial_guess_1 = [y_max * 0.8]
        bounds_1 = ([0.5*y_max], [np.inf]) 
        popt_1, _ = curve_fit(custom_function_1, 
                             X.ravel(), 
                             y.ravel(),
                             p0=initial_guess_1,
                             bounds=bounds_1,
                             method='trf',
                             maxfev=10000)
        m_est_1 = popt_1[0]
    except RuntimeError:
        m_est_1 = y_max

    try:
        initial_guess_2 = [0.95, y_max*0.9]
        bounds_2 = ([0.8, 0.7*y_max],
                    [1.2, np.inf]) 
        popt_2, _ = curve_fit(custom_function_2,
                             X.ravel(),
                             y.ravel(),
                             p0=initial_guess_2,
                             bounds=bounds_2,
                             method='trf',
                             maxfev=10000)
        a_est_2, m_est_2 = popt_2
    except RuntimeError:
        a_est_2, m_est_2 = 1.0, y_max

    return m_est_1, a_est_2, m_est_2

def calculate_deviation(y_true, y_pred):
    deviation = np.mean(np.abs((y_true - y_pred) / y_true))
    return deviation

def plot_fitted_curve(X, y, m_est_1, a_est_2, m_est_2, scaler_X, scaler_y, label, color, ax1, ax2):
    X = scaler_X.inverse_transform(X)
    y = scaler_y.inverse_transform(y)
    
    x_fit = np.linspace(0, min(X.max(), 1200), 100)
    y_fit_1 = custom_function_1(x_fit, m_est_1)
    y_fit_2 = custom_function_2(x_fit, a_est_2, m_est_2)
    
    ax1.plot(x_fit, y_fit_1, color='#1f77b4', linestyle='--', label=f'LW Model')
    ax2.plot(x_fit, y_fit_2, color='#ff7f0e', linestyle='-.', label=f'Depthseek Model')
    
def main():
    plt.rcParams['font.family'] = 'Arial'
    
    parser = argparse.ArgumentParser(description='Fit and predict using a dataset.')
    parser.add_argument('-i', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('-o', type=str, required=True, help='Output directory for logs and plots')
    parser.add_argument('-n', type=float, required=True, help='Sequencing depth to predict')
    args = parser.parse_args()
    
    os.makedirs(args.o, exist_ok=True)
    dataset_name = os.path.basename(args.i).replace('.txt', '')
    
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_and_preprocess_data(args.i)
    
    y_train_inv = scaler_y.inverse_transform(y_train)
    y_max = np.max(y_train_inv)
    
    X_train_inv = scaler_X.inverse_transform(X_train)
    m_est_1, a_est_2, m_est_2 = estimate_parameters(X_train_inv, y_train_inv, y_max)
    
    model = build_model('xgboost')
    model = train_model(model, X_train, y_train)
    
    train_loss, test_loss = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    X_test_inv = scaler_X.inverse_transform(X_test)
    y_test_inv = scaler_y.inverse_transform(y_test)
    
    y_pred_1 = custom_function_1(X_test_inv.ravel(), m_est_1)
    y_pred_2 = custom_function_2(X_test_inv.ravel(), a_est_2, m_est_2)
    
    deviation_1 = calculate_deviation(y_test_inv.ravel(), y_pred_1)
    deviation_2 = calculate_deviation(y_test_inv.ravel(), y_pred_2)
    
    y_pred_n_1 = custom_function_1(args.n, m_est_1)
    y_pred_n_2 = custom_function_2(args.n, a_est_2, m_est_2)
    
    formula_1 = f"y = {m_est_1:.2f} - {m_est_1:.2f} * exp(-1e6*x/{m_est_1:.2f})"
    formula_2 = f"y = {m_est_2:.2f} - {m_est_2:.2f} * exp(-(1e6*x)^{a_est_2:.3f}/{m_est_2:.2f})"
    
    log_file = os.path.join(args.o, f"{dataset_name}_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Max observed y: {y_max:.2f}\n")
        f.write(f"Formula1 params: m={m_est_1:.2f}\n")
        f.write(f"Formula2 params: a={a_est_2:.3f}, m={m_est_2:.2f}\n")
        f.write(f"Predicted value at {args.n}M (Formula1): {y_pred_n_1:.2f}\n")
        f.write(f"Predicted value at {args.n}M (Formula2): {y_pred_n_2:.2f}\n")
        f.write(f"Deviation (Formula1): {deviation_1:.2%}\n")
        f.write(f"Deviation (Formula2): {deviation_2:.2%}\n")
    
    cm = 1/2.54
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12*cm, 6*cm))
    
    plot_fitted_curve(X_train, y_train, m_est_1, a_est_2, m_est_2, 
                     scaler_X, scaler_y, label=dataset_name, color='#EB5B25',
                     ax1=ax1, ax2=ax2)
    
    ax1.scatter(X_train_inv, y_train_inv, s=5, color='gray', alpha=0.3, label='Training Data')
    ax2.scatter(X_train_inv, y_train_inv, s=5, color='gray', alpha=0.3, label='Training Data')
    
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 800)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(False)
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.legend(loc='lower right', fontsize=4,edgecolor='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    ax1.set_xlabel('Sequencing Depth (Million Reads)', fontsize=6)
    ax1.set_ylabel('Library Complexity (Read Counts)', fontsize=6)
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    ax1.text(0.5, 1.075, f'LW Model (Deviation: {deviation_1:.2%})', 
         fontsize=6, ha='center', va='bottom', transform=ax1.transAxes)
    ax2.text(0.5, 1.075, f'Depthseek Model (Deviation: {deviation_2:.2%})', 
         fontsize=6, ha='center', va='bottom', transform=ax2.transAxes)

    plt.tight_layout()
    output_svg = os.path.join(args.o, f"{dataset_name}_fit_result.svg")
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
