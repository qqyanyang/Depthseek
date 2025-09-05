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

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 6

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

def calculate_deviation(y_true, y_pred):
    deviation = np.mean(np.abs((y_true - y_pred) / y_true))
    return deviation

def plot_fitted_curve(X, y, a_est, m_est, scaler_X, scaler_y, output_dir, dataset_name):
    X = scaler_X.inverse_transform(X)
    y = scaler_y.inverse_transform(y)
    
    fig, ax = plt.subplots(figsize=(5/2.54, 5/2.54))
    
    ax.scatter(X, y, s=5, color='#1E90FF', alpha=0.6, label='Observed Data')
    
    x_fit = np.linspace(0, min(X.max(), 1200), 100)
    y_fit = custom_function(x_fit, a_est, m_est)
    ax.plot(x_fit, y_fit, color='#EB5B25', linestyle='-', linewidth=1, label='Fitted Curve')
    
    ax.set_xlim(0, 800)
    ax.set_xlabel('Sequencing Depth (Million Reads)', fontsize=6)
    ax.set_ylabel('Unique Reads', fontsize=6)
    
    ax.tick_params(axis='both', which='major', direction='in', length=3, width=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    formula = r'$y = m - m \cdot \exp\left(-\frac{x^a}{m}\right)$'
    ax.text(0.95, 0.075, formula, 
        transform=ax.transAxes, 
        fontsize=5,
        horizontalalignment='right',
        verticalalignment='bottom'
        )

    ax.legend(loc='lower right', 
          fontsize=5,
          framealpha=0.8,
          bbox_to_anchor=(0.95, 0.125),
          facecolor='white',
          edgecolor='none',
          borderpad=0.3)
    
    output_svg = os.path.join(output_dir, f"{dataset_name}_fit_result.svg")
    plt.savefig(output_svg, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

def generate_html_report(output_dir, dataset_name, a_est, m_est, y_pred_n, deviation, y_max, n_value):  # 添加n_value参数
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Fitting Report - {dataset_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; font-size: 10px; }}
        .container {{ width: 80%; margin: 0 auto; }}
        .image-container {{ text-align: center; margin: 10px 0; }}
        .results {{ margin: 15px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 5px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Fitting Report - {dataset_name}</h2>
        
        <div class="image-container">
            <img src="{dataset_name}_fit_result.svg" alt="Fitting Result" style="width: 5cm; height: 5cm;">
        </div>
        
        <div class="results">
            <h3>Fitting Results</h3>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Max observed y</td>
                    <td>{y_max:.2f}</td>
                </tr>
                <tr>
                    <td>Parameter a</td>
                    <td>{a_est:.3f}</td>
                </tr>
                <tr>
                    <td>Parameter m</td>
                    <td>{m_est:.2f}</td>
                </tr>
                <tr>
                    <td>Predicted value at {n_value}M</td>  <!-- 这里显示传入的-n值 -->
                    <td>{y_pred_n:.2f}</td>
                </tr>
                <tr>
                    <td>Deviation</td>
                    <td>{deviation:.2%}</td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>
    """
    html_file = os.path.join(output_dir, f"{dataset_name}_report.html")
    with open(html_file, 'w') as f:
        f.write(html_content)
def main():
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
    a_est, m_est = estimate_parameters(X_train_inv, y_train_inv, y_max)
    
    model = build_model('xgboost')
    model = train_model(model, X_train, y_train)
    
    train_loss, test_loss = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    X_test_inv = scaler_X.inverse_transform(X_test)
    y_test_inv = scaler_y.inverse_transform(y_test)
    
    y_pred = custom_function(X_test_inv.ravel(), a_est, m_est)
    deviation = calculate_deviation(y_test_inv.ravel(), y_pred)
    
    y_pred_n = custom_function(args.n, a_est, m_est)
    
    log_file = os.path.join(args.o, f"{dataset_name}_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Max observed y: {y_max:.2f}\n")
        f.write(f"Fitted params: a={a_est:.3f}, m={m_est:.2f}\n")
        f.write(f"Predicted value at {args.n}M: {y_pred_n:.2f}\n")
        f.write(f"Deviation: {deviation:.2%}\n")
    
    plot_fitted_curve(X_train, y_train, a_est, m_est, scaler_X, scaler_y, args.o, dataset_name)
    
    generate_html_report(args.o, dataset_name, a_est, m_est, y_pred_n, deviation, y_max, args.n)

if __name__ == "__main__":
    main()
