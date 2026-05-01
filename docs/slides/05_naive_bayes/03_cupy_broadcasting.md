# Naive Bayes: CuPy Vectorized Broadcasting

The CuPy variant eliminates the explicit kernel by expressing the log-likelihood computation as a broadcasting expression over a `(n_test, n_classes, n_features)` virtual tensor. Expanding `X_test` with `[:, cp.newaxis, :]` and `means` with `[cp.newaxis, :, :]` causes NumPy-style broadcasting rules to apply on the GPU — computing all squared differences simultaneously without any explicit loop.

`cp.sum(..., axis=2)` reduces the feature dimension, yielding the `(n_test, n_classes)` log-probability matrix in one fused operation dispatched by cuDNN/CuPy's elementwise kernel framework. This is more memory-intensive than the custom kernel (it materializes the 3D intermediate) but requires zero CUDA programming.

```python
X_gpu = cp.asarray(X_test)                    # (n_test, n_features)
mu    = cp.asarray(means_f32)                  # (n_classes, n_features)
var   = cp.asarray(vars_f32)                   # (n_classes, n_features)
lp    = cp.asarray(log_priors_f32)             # (n_classes,)

# Expand for broadcasting: (n_test, 1, n_features) vs (1, n_classes, n_features)
X_exp   = X_gpu[:, cp.newaxis, :]             # (n_test, 1, n_features)
diff_sq = (X_exp - mu[cp.newaxis, :, :]) ** 2 # (n_test, n_classes, n_features)

log_gaussian = -0.5 * cp.log(2.0 * cp.pi * var) - 0.5 * diff_sq / var
log_probs    = lp[cp.newaxis, :] + cp.sum(log_gaussian, axis=2)  # (n_test, n_classes)

predictions = cp.asnumpy(cp.argmax(log_probs, axis=1).astype(cp.int32))
```

**Source:** `demos/05_naive_bayes/cupy_nb.py:55–67`
