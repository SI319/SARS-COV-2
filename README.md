## Intrinsically disordered regions in SARS-COV-2
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, chi2_contingency, mannwhitneyu, kruskal, stats
file_path = "/content/stat-MR.xlsx"

### Load the Excel file into a DataFrame called 'df'
df = pd.read_excel(file_path)

# Replace infinite values with NaN (if any)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values in the specified columns
df.dropna(subset=['Experiment', 'flDPnn', 'VLXT', 'VSL2', 'ADOPT'], inplace=True)

# Define exp_column here
exp_column = "Experiment"

# --- Analysis for FlPDnn, VLXT, VSL2, Adopt ---
for column in ['flDPnn', 'VLXT', 'VSL2', 'ADOPT']:
    # Point-Biserial Correlation (for binary and continuous data)
    point_biserial_corr, pb_p_value = pointbiserialr(df[exp_column], df[column])
    print(f"Point-Biserial Correlation (Exp vs. {column}): r = {point_biserial_corr:.4f}, p-value = {pb_p_value:.4f}")

    # Chi-squared test (for categorical data - since FlPDnn, VLXT, VSL2, Adopt are 0/1)
    contingency_table = pd.crosstab(df[exp_column], df[column])
    chi2, p_value_chi2, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-squared Test (Exp vs. {column}): chi2 = {chi2:.4f}, p-value = {p_value_chi2:.4f}")

    # Mann-Whitney U test (non-parametric for independent groups)
    group_0 = df[df[exp_column] == 0][column]
    group_1 = df[df[exp_column] == 1][column]
    u_stat, p_value_u = mannwhitneyu(group_0, group_1)
    print(f"Mann-Whitney U Test (Exp vs. {column}): U-stat = {u_stat:.4f}, p-value = {p_value_u:.4f}")

# Kruskal-Wallis H test (comparing all 4 columns with Exp) - can still be used for ordinal data (0/1)
h_stat, p_value_h = kruskal(df[df[exp_column] == 0]['flDPnn'], df[df[exp_column] == 1]['flDPnn'],
                           df[df[exp_column] == 0]['VLXT'], df[df[exp_column] == 1]['VLXT'],
                           df[df[exp_column] == 0]['VSL2'], df[df[exp_column] == 1]['VSL2'],
                           df[df[exp_column] == 0]['ADOPT'], df[df[exp_column] == 1]['ADOPT'])
print(f"Kruskal-Wallis H Test (Exp vs. flDPnn, VLXT, VSL2, Adopt): H-stat = {h_stat:.4f}, p-value = {p_value_h:.4f}")
