import numpy as np
import pandas as pd

# read thyroid dataset from excel
def read_dataset(path):
    return pd.read_excel(path, sheet_name="thyroid0387_UCI")

# calculate frequency counts for binary vectors
def calculate_frequencies(arr1, arr2):
    # initialize counters
    both_one = first_one = second_one = both_zero = 0

    for idx in range(len(arr1)):
        if arr1[idx] == 1 and arr2[idx] == 1:
            # both values are 1
            both_one += 1
        elif arr1[idx] == 1 and arr2[idx] == 0:
            # first is 1, second is 0
            first_one += 1
        elif arr1[idx] == 0 and arr2[idx] == 1:
            # first is 0, second is 1
            second_one += 1
        else:
            # both values are 0
            both_zero += 1

    return both_one, first_one, second_one, both_zero

def main():
    excel_path = r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx"
    df = read_dataset(excel_path)

    # convert true/false labels into binary format
    binary_df = df.replace({"t": 1, "f": 0})

    # extract only numeric (binary) attributes
    numeric_binary = binary_df.select_dtypes(include=[np.number])

    # select two sample records
    record_a = numeric_binary.iloc[0].values
    record_b = numeric_binary.iloc[1].values

    # compute frequency values
    f11, f10, f01, f00 = calculate_frequencies(record_a, record_b)

    # similarity measures
    jaccard_value = f11 / (f01 + f10 + f11)
    smc_value = (f11 + f00) / (f00 + f01 + f10 + f11)

    # display results
    print("Both 1 Count (f11):", f11)
    print("First 1 Only (f10):", f10)
    print("Second 1 Only (f01):", f01)
    print("Both 0 Count (f00):", f00)

    print("Jaccard Similarity:", jaccard_value)
    print("Simple Matching Similarity:", smc_value)
    # usually SMC ≥ Jaccard

if __name__ == "__main__":
    main()

