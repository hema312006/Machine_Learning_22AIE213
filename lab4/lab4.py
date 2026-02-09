import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score
)

# A1: Function to calculate classification performance metrics

def get_classification_results(true_labels, predicted_labels):
    # Create confusion matrix
    conf_mat = confusion_matrix(true_labels, predicted_labels)

    # Calculate evaluation metrics
    precision_val = precision_score(true_labels, predicted_labels)
    recall_val = recall_score(true_labels, predicted_labels)
    f1_val = f1_score(true_labels, predicted_labels)

    # Accuracy computed manually
    accuracy_val = np.mean(true_labels == predicted_labels)

    return conf_mat, accuracy_val, precision_val, recall_val, f1_val

# A2: Function to compute regression metrics

def get_regression_results(actual_values, predicted_values):
    # Mean Squared Error
    mse_val = mean_squared_error(actual_values, predicted_values)

    # Root Mean Squared Error
    rmse_val = np.sqrt(mse_val)

    # Mean Absolute Percentage Error
    mape_val = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    # R-squared value
    r2_val = r2_score(actual_values, predicted_values)

    return mse_val, rmse_val, mape_val, r2_val

# A3: Generate synthetic training data

def create_synthetic_data(sample_count=20):
    # Generate random points between 1 and 10
    features = np.random.uniform(1, 10, size=(sample_count, 2))

    # Assign class based on rule
    labels = np.array([0 if f[0] + f[1] < 11 else 1 for f in features])

    return features, labels


# Plot synthetic data
def display_training_points(features, labels):
    # Plot class 0 points
    plt.scatter(features[labels == 0][:, 0], features[labels == 0][:, 1])

    # Plot class 1 points
    plt.scatter(features[labels == 1][:, 0], features[labels == 1][:, 1])

    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("Scatter Plot of Training Data")
    plt.show()

# A4 & A5: Create test grid and decision boundary

def create_prediction_grid(step_size=0.1):
    x_range = np.arange(0, 10, step_size)
    y_range = np.arange(0, 10, step_size)

    grid_x, grid_y = np.meshgrid(x_range, y_range)
    combined_grid = np.c_[grid_x.ravel(), grid_y.ravel()]

    return grid_x, grid_y, combined_grid


def show_knn_decision_boundary(train_features, train_labels, k_value):
    # Train KNN model
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(train_features, train_labels)

    # Generate prediction grid
    grid_x, grid_y, grid_points = create_prediction_grid()
    grid_predictions = knn_model.predict(grid_points)

    # Plot decision regions
    plt.scatter(grid_points[grid_predictions == 0][:, 0],
                grid_points[grid_predictions == 0][:, 1],
                alpha=0.3)

    plt.scatter(grid_points[grid_predictions == 1][:, 0],
                grid_points[grid_predictions == 1][:, 1],
                alpha=0.3)

    # Plot training points
    plt.scatter(train_features[train_labels == 0][:, 0],
                train_features[train_labels == 0][:, 1])

    plt.scatter(train_features[train_labels == 1][:, 0],
                train_features[train_labels == 1][:, 1])

    plt.title(f"KNN Decision Boundary (k = {k_value})")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.show()

# A7: Find best k using GridSearch

def find_optimal_k(train_features, train_labels):
    # Define search space
    search_params = {"n_neighbors": list(range(1, 21))}

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        search_params,
        cv=5,
        scoring="accuracy"
    )

    grid_search.fit(train_features, train_labels)

    return grid_search.best_params_, grid_search.best_score_

# Main program
# Load dataset
data_file = r"C:\Users\hema3\Downloads\machine learning\Conf_Text_Labels.xlsx"
data_frame = pd.read_excel(data_file, sheet_name="Conf Data", engine="openpyxl")

# Extract numeric features
feature_matrix = data_frame.select_dtypes(include=[np.number]).iloc[:, :-1].values
label_vector = data_frame.iloc[:, -1].values

# Select first two classes only
unique_classes = np.unique(label_vector)[:2]
class_mask = np.isin(label_vector, unique_classes)

feature_matrix = feature_matrix[class_mask]
label_vector = label_vector[class_mask]

# Convert labels into binary values
binary_map = {unique_classes[0]: 0, unique_classes[1]: 1}
label_vector = np.array([binary_map[val] for val in label_vector])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    feature_matrix, label_vector, test_size=0.3, random_state=42
)

# Train KNN model
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# Predictions
train_predictions = knn_classifier.predict(X_train)
test_predictions = knn_classifier.predict(X_test)

# Compute training metrics
train_cm, train_acc, train_prec, train_rec, train_f1 = get_classification_results(
    y_train, train_predictions
)

# Compute test metrics
test_cm, test_acc, test_prec, test_rec, test_f1 = get_classification_results(
    y_test, test_predictions
)

# Display results
print("Training Confusion Matrix:\n", train_cm)
print("Training Accuracy:", train_acc)
print("Training Precision:", train_prec)
print("Training Recall:", train_rec)
print("Training F1:", train_f1)

print("\nTest Confusion Matrix:\n", test_cm)
print("Test Accuracy:", test_acc)
print("Test Precision:", test_prec)
print("Test Recall:", test_rec)
print("Test F1:", test_f1)

# Generate and plot synthetic dataset
synthetic_X, synthetic_y = create_synthetic_data()
display_training_points(synthetic_X, synthetic_y)

# Plot decision boundaries for different k values
for k in [1, 3, 7]:
    show_knn_decision_boundary(synthetic_X, synthetic_y, k)

# Tune k value using grid search
optimal_k, optimal_score = find_optimal_k(X_train, y_train)
print("\nBest k from GridSearch:", optimal_k)
print("Best CV Accuracy:", optimal_score)
