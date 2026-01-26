import numpy as np
import pandas as pd

# read thyroid dataset
def read_thyroid_data(path):
    return pd.read_excel(path, sheet_name="thyroid0387_UCI")

# perform min–max scaling
def scale_min_max(series):
    return (series - series.min()) / (series.max() - series.min())

def main():
    excel_path = r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx"
    df = read_thyroid_data(excel_path)

    # handle missing value symbols
    df = df.replace("?", np.nan)
    df = df.infer_objects(copy=False)

    # ensure selected columns are numeric
    num_features = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
    for feature in num_features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    # apply normalization to numeric attributes
    scaled_df = df.copy()
    for feature in num_features:
        scaled_df[feature] = scale_min_max(df[feature])

    # save normalized dataset
    save_path = r"C:\Users\hema3\Downloads\machine learning\thyroid_normalized.xlsx"
    scaled_df.to_excel(save_path, index=False)

    print("\nMin–Max normalized data saved successfully.")
    print("File location:", save_path)


if __name__ == "__main__":
    main()
