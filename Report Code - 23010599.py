
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:59:04 2024

@author: Saman
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def load_data(file_path):
    """
    Load the sales dataset from the specified file path.

    Parameters:
    file_path (str): Path to the CSV file containing the sales dataset.

    Returns:
    pandas.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def preprocess_data(data):
    """
    Preprocess the dataset by calculating Total Sales and setting High Sales threshold.

    Parameters:
    data (pandas.DataFrame): Original dataset.

    Returns:
    pandas.DataFrame: Preprocessed dataset.
    """

    # Create a list of column names that start with 'W', presumably representing weekly sales data
    weekly_sales_columns = [col for col in data.columns if col.startswith('W')]

    # Calculate the total sales for each row by summing up the values in the weekly sales columns
    data['Total_Sales'] = data[weekly_sales_columns].sum(axis=1)

    # Calculate the median value of the total sales column to use as a threshold
    threshold = data['Total_Sales'].median()

    # Create a new column 'High_Sales' indicating whether the total sales are above the threshold
    # Assign 1 if total sales are above the threshold, 0 otherwise
    data['High_Sales'] = (data['Total_Sales'] > threshold).astype(int)

    return data


def train_test_split_data(X, y, test_size=0.2, random_state=0):
    """
    Split the dataset into training and testing sets.

    Parameters:
    X (pandas.DataFrame): Features.
    y (pandas.Series): Target variable.
    test_size (float): Size of the testing set (default is 0.2).
    random_state (int): Random state for reproducibility (default is 0).

    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_linear_regression(y_true, y_pred):
    """
    Evaluate the performance of Linear Regression model.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    tuple: RMSE and R-squared score.
    """

    # Calculate Root Mean Squared Error (RMSE) between true and predicted target values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate R-squared (coefficient of determination) between true and predicted target values
    r2 = r2_score(y_true, y_pred)

    return rmse, r2


def evaluate_logistic_regression(y_true, y_pred):
    """
    Evaluate the performance of Logistic Regression model.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    tuple: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Classification Report.
    """

    # Calculate accuracy score between true and predicted target values
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision score between true and predicted target values
    precision = precision_score(y_true, y_pred)

    # Calculate recall score between true and predicted target values
    recall = recall_score(y_true, y_pred)

    # Calculate F1 score between true and predicted target values
    f1 = f1_score(y_true, y_pred)

    # Generate confusion matrix between true and predicted target values
    cm = confusion_matrix(y_true, y_pred)

    # Generate classification report containing precision, recall, F1-score, and support
    class_report = classification_report(y_true, y_pred)

    return accuracy, precision, recall, f1, cm, class_report


def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plot Actual vs Predicted Sales from Linear Regression.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    """

    # Create a figure with specific size and resolution
    plt.figure(figsize=(8, 6), dpi=300)

    # Create a scatter plot of predicted vs. actual values
    scatter_handle = plt.scatter(
        y_true, y_pred, alpha=0.5, label='Predicted vs Actual')

    # Add a line representing perfect prediction (y_true = y_pred)
    line_handle, = plt.plot([y_true.min(), y_true.max()], [
                            y_true.min(), y_true.max()], 'k--', lw=2,
                            label='Perfect Prediction')

    # Set labels for x and y axes
    plt.xlabel('Actual Total Sales', fontsize=12)
    plt.ylabel('Predicted Total Sales', fontsize=12)

    # Set title for the plot
    plt.title('Linear Regression: Actual vs Predicted',
              fontsize=14, fontweight='bold', pad=10)

    # Add legend with scatter plot and line plot handles
    plt.legend(handles=[scatter_handle, line_handle],
               loc='upper left', prop={'size': 12})

    # Display the plot
    plt.show()


def plot_confusion_matrix(cm):
    """
    Plot Confusion Matrix from Logistic Regression.

    Parameters:
    cm (array-like): Confusion matrix.
    """

    # Create a figure and axis object for the plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    # Plot the confusion matrix using ConfusionMatrixDisplay
    # Display labels are set as ['Low', 'High'], and colormap is set to 'Blues'
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=['Low', 'High'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    
    # Manually add colorbar
    cbar = plt.colorbar(disp.im_, ax=ax, shrink=0.82)

    # Set title for the plot
    plt.title('Logistic Regression: Confusion Matrix',
              fontsize=14, fontweight='bold', pad=14)

    # Set labels for x and y axes
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # Display the plot
    plt.show()


def plot_performance_comparison(metrics, values):
    """
    Plot Performance Metrics Comparison.

    Parameters:
    metrics (list): List of metric names.
    values (list): List of metric values.
    """

    # Create a figure with specific size and resolution
    plt.figure(figsize=(8, 6), dpi=300)

    # Define colors for the bars
    colors = ['blue', 'purple', 'orange']

    # Create a bar plot with given metrics and corresponding values, using defined colors
    plt.bar(metrics, values, color=colors)

    # Set title for the plot
    plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold')

    # Set y-axis limits
    plt.ylim(0, 1.3)

    # Display the plot
    plt.show()


# Load the dataset from the CSV file using a custom function called load_data
data = load_data('SalesDataset.csv')

# Preprocess the dataset (e.g., handle missing values, encode categorical variables)
data = preprocess_data(data)

# Prepare data for linear regression
# Select features (X_lin) and target variable (y_lin)
# Features are columns starting with 'W'
X_lin = data[[col for col in data.columns if col.startswith('W')]]
y_lin = data['Total_Sales']  # Target variable is 'Total_Sales'

# Prepare data for logistic regression
# For logistic regression, we need a binary target variable
# Features (X_log) are the same as for linear regression
X_log = X_lin

# Target variable (y_log) is 'High_Sales', indicating whether sales are high or not
y_log = data['High_Sales']

# Split data for linear regression
# Split the data into training and testing sets for linear regression
# Features (X_lin) and target variable (y_lin) are split
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split_data(
    X_lin, y_lin)

# Split data for logistic regression
# Split the data into training and testing sets for logistic regression
# Features (X_log) and target variable (y_log) are split
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split_data(
    X_log, y_log)

# Linear Regression Model
# Create a linear regression model instance
linear_reg = LinearRegression()

# Train the linear regression model on the training data
linear_reg.fit(X_train_lin, y_train_lin)

# Make predictions on the testing set using the trained model
y_pred_lin = linear_reg.predict(X_test_lin)

# Logistic Regression Model
# Create a logistic regression model instance with a maximum number of iterations set to 1000
logistic_reg = LogisticRegression(max_iter=1000)

# Train the logistic regression model on the training data
logistic_reg.fit(X_train_log, y_train_log)

# Make predictions on the testing set using the trained logistic regression model
y_pred_log = logistic_reg.predict(X_test_log)

# Evaluate Linear Regression
# Calculate evaluation metrics for linear regression: RMSE and R^2
rmse_lin, r2_lin = evaluate_linear_regression(y_test_lin, y_pred_lin)

# Evaluate Logistic Regression
# Calculate evaluation metrics for logistic regression: accuracy, precision, 
# recall, F1-score, confusion matrix, and classification report
accuracy_log, precision_log, recall_log, f1_log, cm_log, class_report_log = evaluate_logistic_regression(
    y_test_log, y_pred_log)

# Plot Actual vs Predicted Sales from Linear Regression
# Plot a scatter plot of actual vs predicted sales from linear regression
plot_actual_vs_predicted(y_test_lin, y_pred_lin)

# Plot Confusion Matrix from Logistic Regression
plot_confusion_matrix(cm_log)

# Plot Performance Comparison
# Plot a bar chart comparing performance metrics between linear and logistic regression
metrics = ['RMSE', 'R^2', 'Accuracy']
values = [rmse_lin, r2_lin, accuracy_log]
plot_performance_comparison(metrics, values)

# Print metrics for Linear Regression
print("Linear Regression - Mean Squared Error (RMSE):", rmse_lin.round(15))
print("Linear Regression - R-squared (R^2):", r2_lin)

# Print metrics for Logistic Regression
print("Logistic Regression - Accuracy:", accuracy_log.round(4))
print("Logistic Regression - Precision:", precision_log.round(4))
print("Logistic Regression - Recall:", recall_log.round(4))
print("Logistic Regression - F1 Score:", f1_log.round(4))

# Print confusion matrix for Logistic Regression
print("Logistic Regression - Confusion Matrix:")
print(cm_log)

# Print classification report for Logistic Regression
print("Logistic Regression - Classification report")
print(class_report_log)
