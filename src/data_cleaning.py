import pandas as pd
import numpy as np

def clean_data(input_path, output_path="data/cleaned_diabetic_data.csv"):
    
    # Load dataset
    df = pd.read_csv(input_path)

    print("Original shape:", df.shape)

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Remove invalid gender rows
    df = df[df["gender"] != "Unknown/Invalid"]

    # Drop ID columns (not useful for prediction)
    df.drop(columns=["encounter_id", "patient_nbr"], inplace=True, errors="ignore")

    # Drop columns with >50% missing values
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.5].index
    df.drop(columns=cols_to_drop, inplace=True)

    print("Columns dropped due to missing values:", list(cols_to_drop))

    # Create binary target variable
    # 1 = readmitted within 30 days
    # 0 = not readmitted within 30 days
    df["readmit_30"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

    # Keep only your final variables
    selected_columns = [
        "age",
        "gender",
        "insulin",
        "diabetesMed",
        "number_inpatient",
        "number_emergency",
        "time_in_hospital",
        "num_medications",
        "number_diagnoses",
        "admission_type_id",
        "readmit_30"
    ]

    df = df[selected_columns].copy()

    # Drop rows with missing values in selected columns
    df.dropna(inplace=True)

    print("Cleaned shape:", df.shape)

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print(f"Cleaned dataset saved to {output_path}")

    return df


# Run directly if file is executed
if __name__ == "__main__":
    clean_data("data/diabetic_data.csv")


    