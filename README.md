#Intrinsically disordered regions in SARS-COV-2
#Python code for Chi-squared Test and ROC curve
file_path = "/content/stat.xlsx"
#Load the Excel file into a DataFrame called 'df'
df = pd.read_excel(file_path)
#Replace infinite values with NaN (if any)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
#Drop rows with NaN values in the specified columns
df.dropna(subset=['Experiment', 'flDPnn', 'VLXT', 'VSL2', 'ADOPT'], inplace=True)
#Define exp_column here
exp_column = "Experiment"
#Chi-squared test (for categorical data - since FlPDnn, VLXT, VSL2, Adopt are 0/1)
    contingency_table = pd.crosstab(df[exp_column], df[column])
    chi2, p_value_chi2, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-squared Test (Exp vs. {column}): chi2 = {chi2:.4f}, p-value = {p_value_chi2:.4f}")
