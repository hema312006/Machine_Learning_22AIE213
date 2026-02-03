import pandas as pd
import numpy as np

#Basic statistical helper functions

def compute_mean(data_array):
    """Return the mean of the given values"""
    return np.mean(data_array)

def compute_variance(data_array):
    """Return the variance of the given values"""
    return np.var(data_array)

def compute_standard_deviation(data_array):
    """Return the standard deviation of the given values"""
    return np.std(data_array)


#Dataset-level statistics 

def calculate_dataset_statistics(feature_matrix):
    """
    Calculate mean, variance, and standard deviation
    for each feature column in the dataset
    """
    mean_values = np.mean(feature_matrix, axis=0)
    variance_values = np.var(feature_matrix, axis=0)
    std_values = np.std(feature_matrix, axis=0)

    return mean_values, variance_values, std_values


#Class-wise statistics 

def compute_class_center_and_dispersion(feature_matrix, class_labels, target_class):
    """
    Compute the center (mean vector) and spread (std deviation)
    for a specific class
    """
    class_samples = feature_matrix[class_labels == target_class]

    class_center = np.mean(class_samples, axis=0)
    class_spread = np.std(class_samples, axis=0)

    return class_center, class_spread


#Distance calculation 

def calculate_euclidean_distance(vector_one, vector_two):
    """Compute Euclidean distance between two vectors"""
    return np.linalg.norm(vector_one - vector_two)


def run_analysis():
    # Load the Excel dataset
    dataset = pd.read_excel(
        r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx",
        #"Lab Session Data.xlsx",
        sheet_name="Purchase data"
    )

    # Extract feature columns
    feature_data = dataset[
        ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
    ].values

    # Convert payment amount into class labels
    class_labels = dataset['Payment (Rs)'].apply(
        lambda payment: "RICH" if payment > 200 else "POOR"
    ).values

    # Compute overall dataset statistics
    overall_mean, overall_variance, overall_std = calculate_dataset_statistics(feature_data)

    # Compute statistics for each class
    rich_center, rich_spread = compute_class_center_and_dispersion(
        feature_data, class_labels, "RICH"
    )

    poor_center, poor_spread = compute_class_center_and_dispersion(
        feature_data, class_labels, "POOR"
    )

    # Calculate distance between class centers
    center_distance = calculate_euclidean_distance(rich_center, poor_center)

    # Display results
    print("Dataset Mean:", overall_mean)
    print("Dataset Variance:", overall_variance)
    print("Dataset Std Dev:", overall_std)

    print("\nCenter (RICH):", rich_center)
    print("Spread (RICH):", rich_spread)

    print("\nCenter (POOR):", poor_center)
    print("Spread (POOR):", poor_spread)

    print("\nDistance Between Classes:", center_distance)

if __name__ == "__main__":
    run_analysis()
