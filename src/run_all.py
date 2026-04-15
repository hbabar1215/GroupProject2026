from data_cleaning import clean_data
from analysis import run_analysis

df = clean_data("data/diabetic_data.csv")
run_analysis(df)

print("Analysis complete. Check images folder.")