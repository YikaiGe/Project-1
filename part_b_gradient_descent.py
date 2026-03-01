import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_excel(r"C:\Users\46473\OneDrive\Desktop\CSE5104\Project\1\concretecompressivestrength\Concrete_Data.xls")
col_names = ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age', 'Strength']
df.columns = col_names

feature_names = col_names[:-1]  
target_name = 'Strength'

# Test Set
test_indices = list(range(500, 630)) 
# Train set
train_indices = [i for i in range(len(df)) if i not in test_indices]  

df_train = df.iloc[train_indices].reset_index(drop=True)
df_test = df.iloc[test_indices].reset_index(drop=True)

X_train_raw = df_train[feature_names].values  
y_train = df_train[target_name].values        
X_test_raw = df_test[feature_names].values     
y_test = df_test[target_name].values            

def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 0.000000001  
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    return X_train_std, X_test_std, mean, std

X_train_std, X_test_std, train_mean, train_std = standardize(X_train_raw, X_test_raw)

def compute_mse(X, y, weights, bias):
    mse = np.sum((X @ weights + bias - y) ** 2) / len(y)
    return mse

def compute_r_squared(X, y, weights, bias):
    mse = np.mean((X @ weights + bias - y) ** 2)
    variance = np.var(y)
    r_squared = 1 - (mse / variance)
    return r_squared

def gradient_descent(X, y, learning_rate=0.01, max_iterations=100000, 
                     tolerance=1e-9, verbose=False, init_weights=None, init_bias=None):
    n_samples, n_features = X.shape    
    weights = np.zeros(n_features) if init_weights is None else init_weights.copy()
    bias = 0.0 if init_bias is None else init_bias
    loss_history = []
    
    for iteration in range(max_iterations):
        predictions = X @ weights + bias
        residuals = predictions - y  
        
        grad_weights = (2.0 / n_samples) * (X.T @ residuals)
        grad_bias = (2.0 / n_samples) * np.sum(residuals)
        
        weights = weights - learning_rate * grad_weights
        bias = bias - learning_rate * grad_bias
        
        mse = np.sum(residuals ** 2) / n_samples
        loss_history.append(mse)
        
        if iteration > 0 and abs(loss_history[-2] - loss_history[-1]) < tolerance:
            break
        
        if np.isnan(mse) or np.isinf(mse):
            break
    
    return weights, bias, loss_history


# Univariate Models Set 1 Standardized
univariate_results_std = []

for i, fname in enumerate(feature_names):
    X_tr = X_train_std[:, i:i+1] 
    X_te = X_test_std[:, i:i+1]   
    lr = 0.01
    max_iter = 100000
    
    weights, bias, loss_hist = gradient_descent(
        X_tr, y_train, learning_rate=lr, max_iterations=max_iter
    )
    
    train_mse = compute_mse(X_tr, y_train, weights, bias)
    train_r2 = compute_r_squared(X_tr, y_train, weights, bias)
    test_mse = compute_mse(X_te, y_test, weights, bias)
    test_r2 = compute_r_squared(X_te, y_test, weights, bias)
    
    univariate_results_std.append({
        'feature': fname,
        'weight': weights[0],
        'bias': bias,
        'train_mse': train_mse,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'lr': lr,
        'iterations': len(loss_hist),
        'loss_history': loss_hist,
    })
    

# Univariate Models Set 2 Raw

univariate_results_raw = []
raw_learning_rates = {
    'Cement': 1e-6,
    'Blast_Furnace_Slag': 1e-7,
    'Fly_Ash': 1e-7,
    'Water': 1e-5,
    'Superplasticizer': 1e-4,
    'Coarse_Aggregate': 1e-7,
    'Fine_Aggregate': 1e-7,
    'Age': 1e-4,
}

for i, fname in enumerate(feature_names):
    X_tr = X_train_raw[:, i:i+1]
    X_te = X_test_raw[:, i:i+1]
    
    lr = raw_learning_rates[fname]
    max_iter = 100000
    
    weights, bias, loss_hist = gradient_descent(
        X_tr, y_train, learning_rate=lr, max_iterations=max_iter
    )
    
    train_mse = compute_mse(X_tr, y_train, weights, bias)
    train_r2 = compute_r_squared(X_tr, y_train, weights, bias)
    test_mse = compute_mse(X_te, y_test, weights, bias)
    test_r2 = compute_r_squared(X_te, y_test, weights, bias)
    
    univariate_results_raw.append({
        'feature': fname,
        'weight': weights[0],
        'bias': bias,
        'train_mse': train_mse,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'lr': lr,
        'iterations': len(loss_hist),
        'loss_history': loss_hist,
    })

# Multivariate Model Set 1 Standardized

weights_mv_std, bias_mv_std, loss_mv_std = gradient_descent(
    X_train_std, y_train, learning_rate=0.01, max_iterations=100000, verbose=True
)

mv_std_train_mse = compute_mse(X_train_std, y_train, weights_mv_std, bias_mv_std)
mv_std_train_r2 = compute_r_squared(X_train_std, y_train, weights_mv_std, bias_mv_std)

mv_std_test_mse = compute_mse(X_test_std, y_test, weights_mv_std, bias_mv_std)
mv_std_test_r2 = compute_r_squared(X_test_std, y_test, weights_mv_std, bias_mv_std)

# Multivariate Model Set 2 Raw

weights_mv_raw, bias_mv_raw, loss_mv_raw = gradient_descent(
    X_train_raw, y_train, learning_rate=1e-7, max_iterations=100000, verbose=True
)

mv_raw_train_mse = compute_mse(X_train_raw, y_train, weights_mv_raw, bias_mv_raw)
mv_raw_train_r2 = compute_r_squared(X_train_raw, y_train, weights_mv_raw, bias_mv_raw)

mv_raw_test_mse = compute_mse(X_test_raw, y_test, weights_mv_raw, bias_mv_raw)
mv_raw_test_r2 = compute_r_squared(X_test_raw, y_test, weights_mv_raw, bias_mv_raw)

# Plots

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot 1: Multivariate Standardized loss curve
# ax.plot(loss_mv_std, color='steelblue', linewidth=1)
# ax.set_title('Multivariate Model (Standardized) - Loss Curve', fontweight='bold')
# ax.set_xlabel('Iteration')
# ax.set_ylabel('MSE')
# ax.set_yscale('log')
# ax.grid(True, alpha=0.3)

# # Plot 2: Multivariate Raw loss curve
# ax = axes[0, 1]
ax.plot(loss_mv_raw, color='darkorange', linewidth=1)
ax.set_title('Multivariate Model (Raw) - Loss Curve', fontweight='bold')
ax.set_xlabel('Iteration')
ax.set_ylabel('MSE')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# # Plot 3: Best univariate standardized (Cement)
# ax = axes[1, 0]
# best_uni_std = max(univariate_results_std, key=lambda x: x['train_r2'])
# ax.plot(best_uni_std['loss_history'], color='forestgreen', linewidth=1)
# ax.set_title(f"Univariate (Std) - {best_uni_std['feature']} - Loss Curve", fontweight='bold')
# ax.set_xlabel('Iteration')
# ax.set_ylabel('MSE')
# ax.set_yscale('log')
# ax.grid(True, alpha=0.3)

# # Plot 4: Best univariate raw (Cement)
# best_uni_raw = max(univariate_results_raw, key=lambda x: x['train_r2'])
# ax.plot(best_uni_raw['loss_history'], color='crimson', linewidth=1)
# ax.set_title(f"Univariate (Raw) - {best_uni_raw['feature']} - Loss Curve", fontweight='bold')
# ax.set_xlabel('Iteration')
# ax.set_ylabel('MSE')
# ax.set_yscale('log')
# ax.grid(True, alpha=0.3)

# plt.suptitle('Gradient Descent Convergence: MSE Loss Over Iterations', 
#              fontsize=14, fontweight='bold', y=1.02)
# plt.tight_layout()
# plt.savefig('loss_curves.png', dpi=150, bbox_inches='tight')



# Result
print("\n" + "=" * 80)
print("COMPLETE RESULTS SUMMARY")
print("=" * 80)

print("\n--- Univariate Models (Standardized) ---")
print(f"{'Feature':25s} | {'Weight':>10s} | {'Bias':>10s} | {'Train MSE':>10s} | {'Train R²':>10s} | {'Test MSE':>10s} | {'Test R²':>10s} | {'LR':>10s} | {'Iters':>6s}")
print("-" * 120)
for r in univariate_results_std:
    print(f"{r['feature']:25s} | {r['weight']:+10.4f} | {r['bias']:10.4f} | {r['train_mse']:10.4f} | {r['train_r2']:+10.4f} | {r['test_mse']:10.4f} | {r['test_r2']:+10.4f} | {r['lr']:10.6f} | {r['iterations']:6d}")

print("\n--- Univariate Models (Raw) ---")
print(f"{'Feature':25s} | {'Weight':>12s} | {'Bias':>10s} | {'Train MSE':>10s} | {'Train R²':>10s} | {'Test MSE':>10s} | {'Test R²':>10s} | {'LR':>10s} | {'Iters':>6s}")
print("-" * 128)
for r in univariate_results_raw:
    print(f"{r['feature']:25s} | {r['weight']:+12.6f} | {r['bias']:10.4f} | {r['train_mse']:10.4f} | {r['train_r2']:+10.4f} | {r['test_mse']:10.4f} | {r['test_r2']:+10.4f} | {r['lr']:10.8f} | {r['iterations']:6d}")

print(f"\n--- Multivariate Model (Standardized) ---")
print(f"  Train MSE: {mv_std_train_mse:.4f}  |  Train R²: {mv_std_train_r2:.4f}")
print(f"  Test  MSE: {mv_std_test_mse:.4f}  |  Test  R²: {mv_std_test_r2:.4f}")
for fname, w in zip(feature_names, weights_mv_std):
    print(f"    {fname:25s}: {w:+.6f}")
print(f"    {'Bias':25s}: {bias_mv_std:+.6f}")

print(f"\n--- Multivariate Model (Raw) ---")
print(f"  Train MSE: {mv_raw_train_mse:.4f}  |  Train R²: {mv_raw_train_r2:.4f}")
print(f"  Test  MSE: {mv_raw_test_mse:.4f}  |  Test  R²: {mv_raw_test_r2:.4f}")
for fname, w in zip(feature_names, weights_mv_raw):
    print(f"    {fname:25s}: {w:+.8f}")
print(f"    {'Bias':25s}: {bias_mv_raw:+.6f}")


# Q2.1

X_test1 = np.array([[3, 4, 5]])
y_test1 = np.array([4])

w1, b1, _ = gradient_descent( X_test1, y_test1, learning_rate=0.1, max_iterations=1, init_weights=np.array([1.0, 1.0, 1.0]), init_bias=1.0)
print("\nQ2.1:")
print(f"  m_1 = {w1[0]}")  
print(f"  m_2 = {w1[1]}")   
print(f"  m_3 = {w1[2]}")   
print(f"  b   = {b1}")     


# Q2.2 

X_test2 = np.array([[3,4,4], [4,2,1], [10,2,5], [3,4,5], [11,1,1]])
y_test2 = np.array([3, 2, 8, 4, 5])

w2, b2, _ = gradient_descent(X_test2, y_test2, learning_rate=0.1, max_iterations=1, init_weights=np.array([1.0, 1.0, 1.0]), init_bias=1.0)

print("\nQ2.2:")
print(f"  m_1 = {w2[0]}")   
print(f"  m_2 = {w2[1]}")   
print(f"  m_3 = {w2[2]}")   
print(f"  b   = {b2}")      

plt.show()
