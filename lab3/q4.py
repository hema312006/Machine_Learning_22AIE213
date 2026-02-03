#Minkowski Distance for Different p Values

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_minkowski(vec_a, vec_b, p_value):
    """
    Compute the Minkowski distance between two vectors
    for a given p value
    """
    distance_sum = 0

    for index in range(len(vec_a)):
        distance_sum += abs(vec_a[index] - vec_b[index]) ** p_value

    return distance_sum ** (1 / p_value)


def run_minkowski_analysis():
    # Load dataset from Excel file
    data_frame = pd.read_excel(
        r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx",
        sheet_name="Purchase data"
    )

    # Extract feature vectors
    feature_matrix = data_frame[
        ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
    ].values

    # Select two sample vectors
    first_vector = feature_matrix[0]
    second_vector = feature_matrix[1]

    # Compute Minkowski distances for p = 1 to 10
    distance_results = []
    for p in range(1, 11):
        distance_results.append(
            calculate_minkowski(first_vector, second_vector, p)
        )

    # Plot p values vs distance
    plt.plot(range(1, 11), distance_results, marker='o')
    plt.xlabel("p value")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski Distance vs p Value")
    plt.show()
    
if __name__ == "__main__":
    run_minkowski_analysis()
