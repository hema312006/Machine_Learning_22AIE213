# A6 : Train–Test Split of Purchase Dataset

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # Load the Excel dataset
    data_frame = pd.read_excel(
        r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx",
        sheet_name="Purchase data"
    )

    # Extract input features
    feature_data = data_frame[
        ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
    ].values

    # Create class labels based on payment amount
    class_labels = data_frame['Payment (Rs)'].apply(
        lambda amount: "RICH" if amount > 200 else "POOR"
    ).values

    # Split the dataset into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        feature_data,
        class_labels,
        test_size=0.3
    )

    # Display sizes of training and testing sets
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

if __name__ == "__main__":
    main()
