import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_excel(r"C:\Users\46473\OneDrive\Desktop\CSE5104\Project\1\concretecompressivestrength\Concrete_Data.xls")
col_names = ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age', 'Strength']
df.columns = col_names

feature_names = col_names[:-1]

test_indices = list(range(500, 630))
train_indices = [i for i in range(len(df)) if i not in test_indices]

df_train = df.iloc[train_indices].reset_index(drop=True)

X_train_raw = df_train[feature_names].values
y_train = df_train['Strength'].values

# Set 1 Standardized

print("=" * 80)
print("SET 1: STANDARDIZED FEATURES")
print("=" * 80)

mean = X_train_raw.mean(axis=0)
std = X_train_raw.std(axis=0)
X_train_std = (X_train_raw - mean) / std

X_std_with_const = sm.add_constant(X_train_std)

model_std = sm.OLS(y_train, X_std_with_const).fit()
print(model_std.summary())

print("\n--- P-values (Standardized) ---")
for name, pval in zip(feature_names, model_std.pvalues[1:]):  
    print(f"  {name:25s}: p = {pval:.6e}")

# Set 2 Untransformed


print("\n" + "=" * 80)
print("SET 2: RAW (UNTRANSFORMED) FEATURES")
print("=" * 80)

X_raw_with_const = sm.add_constant(X_train_raw)

model_raw = sm.OLS(y_train, X_raw_with_const).fit()
print(model_raw.summary())

print("\n--- P-values (Raw) ---")
for name, pval in zip(feature_names, model_raw.pvalues[1:]):
    print(f"  {name:25s}: p = {pval:.6e}")

y_pred_train = model_raw.predict(X_raw_with_const)
train_mse = np.mean((y_train - y_pred_train) ** 2)
print(f"Train MSE: {train_mse:.6f}")
print(f"Train R²: {model_raw.rsquared:.6f}")

df_test = df.iloc[test_indices].reset_index(drop=True)
X_test_raw = df_test[feature_names].values
y_test = df_test['Strength'].values

X_test_with_const = sm.add_constant(X_test_raw)
y_pred_test = model_raw.predict(X_test_with_const)

test_mse = np.mean((y_test - y_pred_test) ** 2)
test_r2 = 1 - test_mse / np.var(y_test)

print(f"Test MSE: {test_mse:.6f}")
print(f"Test R²:  {test_r2:.6f}")

# Set 3 Log Transformed

print("\n" + "=" * 80)
print("SET 3: LOG TRANSFORMED FEATURES (log(x + 1))")
print("=" * 80)

X_train_log = np.log(X_train_raw + 1)

X_log_with_const = sm.add_constant(X_train_log)

model_log = sm.OLS(y_train, X_log_with_const).fit()
print(model_log.summary())

print("\n--- P-values (Log Transformed) ---")
for name, pval in zip(feature_names, model_log.pvalues[1:]):
    print(f"  {name:25s}: p = {pval:.6e}")


