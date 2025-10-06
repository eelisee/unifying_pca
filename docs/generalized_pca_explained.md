# Generalized PCA: A Unified Framework for Regression and Dimensionality Reduction

This document explains how our Generalized PCA framework combines Linear Regression and Principal Component Analysis (PCA) into a single, unified mathematical approach.

---

## **What Problem Are We Solving?**

Imagine you have:
- **Features**: X (like height, weight, age of people)
- **Target**: y (like income you want to predict)

Traditionally, you have two separate tools:
1. **Linear Regression**: Predicts y from X
2. **PCA**: Finds the patterns in X to approximate the original data on lower dimension

**Our question**: What if we could do both simultaneously and let them help each other?

---

## **The Mathematical Foundation**

### Step 1: Create the Augmented Data Matrix

Instead of treating X and y separately, we combine them:

```
X̃ = [y, X₁, X₂, ..., Xₚ]
```

**Example**: If you have 3 features (height, weight, age) and want to predict income:
```
X̃ = [income, height, weight, age]
```

Each row is now a complete "data point" with both the target and features.

### Step 2: The Unified Optimization Problem

We want to find an operator A that minimizes:

```
minimize ||X̃ - A · X̃||²_F
```

### Step 3: The Block Structure of Operator A

It has a specific block form:

```
A = [A_σ   A_β^T]
    [A_β   A_μ  ]
```

Where:
- **A_σ**: A single number (usually ≈ 0)
- **A_β**: A vector of size p (the regression coefficients!)
- **A_μ**: A p×p matrix (the PCA projection)

---

## **How the Implementation Works**

### Algorithm Overview

```python
def generalized_pca(X, y, n_components):
    # Step 1: Prepare the data
    X_centered = X - mean(X)
    y_centered = y - mean(y)
    X_tilde = [y_centered | X_centered]  # Concatenate horizontally
    
    # Step 2: Find the best rank-r approximation using SVD
    U, S, V^T = SVD(X_tilde)
    V_r = V^T[:n_components, :].T  # Take first r components
    A = V_r @ V_r.T               # This is our operator A
    
    # Step 3: Extract the blocks
    A_sigma = A[0, 0]             # Top-left corner
    A_beta = A[1:, 0]            # First column (except top)
    A_mu = A[1:, 1:]             # Bottom-right block
    
    return A_sigma, A_beta, A_mu
```

### What Each Block Does

1. **A_β (Regression Block)**:
   ```
   y_predicted = X @ A_β
   ```
   This gives predictions, just like linear regression.

2. **A_μ (PCA Block)**:
   ```
   X_reconstructed = X @ A_μ
   ```
   This reconstructs X using the most important patterns.

3. **A_σ (Coupling Block)**:
   Usually ≈ 0, but theoretically important for the framework.

---

## **Concrete Example**

Let's say we have:
- **X**: 100 people with [height, weight] 
- **y**: Their income
- **n_components = 1**: We want 1-dimensional approximation

### Step-by-Step Calculation

```python
# Original data
X = [[170, 70], [180, 80], [160, 60], ...]  # height, weight
y = [50000, 60000, 45000, ...]              # income

# Step 1: Create augmented matrix
X_tilde = [[50000, 170, 70],
           [60000, 180, 80], 
           [45000, 160, 60], ...]

# Step 2: SVD to find best rank-1 approximation
U, S, V_T = svd(X_tilde)
V_1 = V_T[0, :].reshape(-1, 1)  # First singular vector
A = V_1 @ V_1.T

# Step 3: A looks like:
A = [[0.01,  0.3,   0.2 ],    # [A_σ,  A_β^T    ]
     [0.4,   0.7,   0.1 ],    # [      A_μ       ]
     [0.2,   0.1,   0.8 ]]    # [A_β,           ]

# Extract blocks:
A_sigma = 0.01          # Coupling term
A_beta = [0.4, 0.2]     # Regression: income = 0.4*height + 0.2*weight
A_mu = [[0.7, 0.1],     # PCA projection matrix
        [0.1, 0.8]]
```

### Making Predictions

```python
# Predict income for new person: height=175, weight=75
new_person = [175, 75]
predicted_income = new_person @ A_beta  # = 175*0.4 + 75*0.2 = 85

# Reconstruct their "typical" features using PCA patterns
reconstructed_features = new_person @ A_mu  # Apply PCA projection
```

---

## **Why This Works: The Implicit Weighting**

The algorithm automatically balances:

1. **Prediction Accuracy**: How well A_β predicts y from X
2. **Reconstruction Quality**: How well A_μ captures patterns in X

### The Trade-off

When we minimize ||X̃ - A·X̃||², we're simultaneously saying:
- "Make y_predicted ≈ y_actual" (regression objective)
- "Make X_reconstructed ≈ X_actual" (PCA objective)

The SVD finds the **optimal balance** between these competing goals!

### Mathematical Intuition

```
Total Error = ||y - A_β·X||² + ||X - A_μ·X||²
              ↑________________   ↑________________
              Regression Error    Reconstruction Error
```

The algorithm minimizes both simultaneously, with the balance determined by the data's natural structure.

---

## **Controlling the Model Behavior**

### Method 1: Data Perspective (What We Did Earlier)

Change what goes into X̃:
- **Pure Regression**: X̃ = [y, X], then force A_μ = 0
- **Pure PCA**: X̃ = [X], ignore y completely  
- **Joint**: X̃ = [y, X], let A optimize freely

### Method 2: Model Perspective (More Elegant)

Keep X̃ = [y, X] fixed, but constrain A:

```python
# Pure Regression Model
A_regression = [[0,    A_β^T],     # A_μ = 0 (no PCA)
                [A_β,  0    ]]

# Pure PCA Model  
A_pca = [[0,  0  ],               # A_β = 0 (no regression)
         [0,  A_μ]]

# Joint Model (both active)
A_joint = [[A_σ,  A_β^T],         # Both blocks active
           [A_β,  A_μ  ]]
```