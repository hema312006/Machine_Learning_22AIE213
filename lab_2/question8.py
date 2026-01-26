import numpy as np
import pandas as pd

# read thyroid dataset from excel
def read_dataset(path):
    return pd.read_excel(path, sheet_name="thyroid0387_UCI")

def main():
    excel_path = r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx"
    df = read_dataset(excel_path)

    # mark missing entries as NaN
    df.replace("?", np.nan, inplace=True)
    df = df.infer_objects(copy=False)

    # impute categorical attributes using mode
    cat_features = ["sex", "referral source", "Condition"]
    for feature in cat_features:
        if feature in df.columns:
            most_common = df[feature].mode()[0]
            df[feature] = df[feature].fillna(most_common)

    # impute age using mean value
    df["age"] = df["age"].fillna(df["age"].mean())

    # impute numeric attributes using median (robust to outliers)
    numeric_features = ["TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
    for feature in numeric_features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
            df[feature] = df[feature].fillna(df[feature].median())

    # display remaining missing values
    print("Null value count after preprocessing:\n")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()

