## Intrinsically disordered regions in SARS-COV-2
### Python code for Chi-squared Test and ROC curve
file_path = "/content/stat.xlsx"
### Load the Excel file into a DataFrame called 'df'
df = pd.read_excel(file_path)
### Replace infinite values with NaN (if any)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
### Drop rows with NaN values in the specified columns
df.dropna(subset=['Experiment', 'flDPnn', 'VLXT', 'VSL2', 'ADOPT'], inplace=True)
### Define exp_column here
exp_column = "Experiment"
### Chi-squared test (for categorical data - since FlPDnn, VLXT, VSL2, Adopt are 0/1)
    contingency_table = pd.crosstab(df[exp_column], df[column])
    chi2, p_value_chi2, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-squared Test (Exp vs. {column}): chi2 = {chi2:.4f}, p-value = {p_value_chi2:.4f}")
### import libraries 
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
### Logistic Regression
y = df['Experiment']
X = df[['flDPnn', 'VLXT', 'VSL2', 'ADOPT']]
X = sm.add_constant(X)  # Add a constant (intercept)

model = sm.Logit(y, X)
result = model.fit()
print(result.summary())

# Confusion Matrix and Classification Metrics (without VLXT)
models = ['flDPnn', 'VLXT', 'VSL2', 'ADOPT']
for model_name in models:
    y_true = df['Experiment']
    y_pred = df[model_name]
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\nModel: {model_name}")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

### ROC Curve Analysis
plt.figure(figsize=(8, 6))
for model_name in models:
    fpr, tpr, thresholds = roc_curve(df['Experiment'], df[model_name])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

### Chi-squared odds ratio and risk ratio
from scipy.stats import chi2_contingency
for model_name in models:
    contingency_table = pd.crosstab(df['Experiment'], df[model_name])
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    odds_ratio = (contingency_table.iloc[1, 1] * contingency_table.iloc[0, 0]) / (
        contingency_table.iloc[1, 0] * contingency_table.iloc[0, 1])
    risk_ratio = (contingency_table.iloc[1, 1] / (
        contingency_table.iloc[1, 1] + contingency_table.iloc[1, 0])) / (
            contingency_table.iloc[0, 1] / (
                contingency_table.iloc[0, 1] + contingency_table.iloc[0, 0]))
    print(f"\n{model_name} Chi-squared analysis")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"Risk Ratio: {risk_ratio:.4f}")
