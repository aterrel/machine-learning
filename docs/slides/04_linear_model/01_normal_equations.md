# Linear Regression: Normal Equations

Linear regression finds the weight vector `w` that minimizes `||Xw - y||²`. The closed-form solution is the normal equation: `(XᵀX) w = Xᵀy`. Solve for `w` with a linear system solver (no iterative gradient descent needed). This decomposition is GPU-friendly: `XᵀX` is a matrix product and `Xᵀy` is a matrix-vector product — both are massively parallel.

The GPU computes `XᵀX` and `Xᵀy` on device; the small linear system solve (`n_features × n_features`) runs on the CPU via `np.linalg.solve`. R² and MSE are then computed from the fitted weights.

```
Normal equation:  XᵀX · w = Xᵀy

Solve:  w = (XᵀX)⁻¹ Xᵀy
           (implemented as np.linalg.solve, not explicit inverse)

Metrics:
  ŷ     = X @ w
  ss_res = sum((y - ŷ)²)
  ss_tot = sum((y - ȳ)²)
  R²    = 1 - ss_res / ss_tot
  MSE   = ss_res / n_samples
```

**Source:** `demos/04_linear_model/gpu_linear.py:33–54`
