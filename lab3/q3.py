# A3 : Histogram, Mean and Variance of a Single Feature

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_candy_feature(excel_file):
    """
    Load the Excel file and extract the 'Candies (#)' column
    """
    purchase_df = pd.read_excel(excel_file, sheet_name="Purchase data")
    return purchase_df['Candies (#)'].values

def compute_average(values):
    """
    Calculate mean of the given data
    """
    return np.mean(values)


def compute_variance(values):
    """
    Calculate variance of the given data
    """
    return np.var(values)

def draw_histogram(values):
    """
    Plot histogram for the given feature values
    """
    # Compute histogram data (counts and bin edges)
    frequency, bin_edges = np.histogram(values, bins=10)

    # Plot histogram
    plt.hist(values, bins=10)
    plt.xlabel("Candies")
    plt.ylabel("Frequency")
    plt.title("Histogram of Candies Purchased")
    plt.show()

    return frequency, bin_edges

def main():
    # Path to the Excel dataset
    excel_file_path = r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx"

    # Load feature data
    candy_values = extract_candy_feature(excel_file_path)

    # Compute mean and variance
    mean_result = compute_average(candy_values)
    variance_result = compute_variance(candy_values)

    # Display results
    print("Mean:", mean_result)
    print("Variance:", variance_result)

    # Plot histogram
    draw_histogram(candy_values)


if __name__ == "__main__":
    main()
