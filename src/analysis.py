import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_analysis(df):

    features = [
        'number_inpatient',
        'num_medications',
        'time_in_hospital',
        'number_diagnoses'
    ]

    df_model = df[features + ['readmitted_binary']].dropna()

    # Correlation heatmap
    corr = df_model.corr()
    sns.heatmap(corr, annot=True)
    plt.title("Correlation Matrix")
    plt.savefig("images/correlation.png")
    plt.close()

    # Feature importance
    importance = corr['readmitted_binary'].drop('readmitted_binary')
    importance.sort_values().plot(kind='barh')
    plt.title("Feature Importance")
    plt.savefig("images/importance.png")
    plt.close()
    