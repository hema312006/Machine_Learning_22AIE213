# A5 : Comparison of Manual and SciPy Minkowski Distance

import pandas as pd
from scipy.spatial.distance import minkowski

def compute_minkowski_distance(vector_one, vector_two, p_value):
    """
    Manually compute the Minkowski distance between two vectors
    """
    distance_sum = 0

    for index in range(len(vector_one)):
        distance_sum += abs(vector_one[index] - vector_two[index]) ** p_value

    return distance_sum ** (1 / p_value)

def main():
    # Load the dataset from Excel
    data_frame = pd.read_excel(
        r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx",
        sheet_name="Purchase data"
    )

    # Extract feature vectors
    feature_vectors = data_frame[
        ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
    ].values

    # Compute Minkowski distance manually
    manual_distance = compute_minkowski_distance(
        feature_vectors[0],
        feature_vectors[1],
        3
    )

    # Compute Minkowski distance using SciPy
    scipy_distance = minkowski(
        feature_vectors[0],
        feature_vectors[1],
        3
    )

    # Display results
    print("Manual Minkowski:", manual_distance)
    print("SciPy Minkowski:", scipy_distance)

if __name__ == "__main__":
    main()
