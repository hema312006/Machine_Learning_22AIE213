import numpy as np
import pandas as pd

# function to read thyroid dataset
def read_data(path):
    return pd.read_excel(path, sheet_name="thyroid0387_UCI")

def main():
    file_path = r"C:\Users\hema3\Downloads\machine learning\Lab Session Data.xlsx"
    df = read_data(file_path)

    # replace missing value symbols with NaN
    df.replace("?", np.nan, inplace=True)

    # rename medical attributes for clarity
    df.rename(columns={
        "TSH": "TSH_Level",
        "T3": "T3_Hormone",
        "TT4": "Total_T4",
        "T4U": "T4_Uptake",
        "FTI": "Free_Thyroxine_Index",
        "TBG": "Thyroxine_Binding_Globulin"
    }, inplace=True)

    # convert selected columns to numeric
    numeric_cols = [
        "TSH_Level",
        "T3_Hormone",
        "Total_T4",
        "T4_Uptake",
        "Free_Thyroxine_Index",
        "Thyroxine_Binding_Globulin"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # select numeric data only
    numeric_df = df.select_dtypes(include=[np.number])

    # min–max range
    print("\nRange of Thyroid Parameters:\n")
    for col in numeric_df.columns:
        print(
            col,
            "Min:", numeric_df[col].min(),
            "Max:", numeric_df[col].max()
        )

    # mean and variance
    print("\nMean and Variance of Thyroid Parameters:\n")
    for col in numeric_df.columns:
        print(
            col,
            "\n Mean:", numeric_df[col].mean(),
            "\n Variance:", numeric_df[col].var(),
            "\n"
        )


if __name__ == "__main__":
    main()

