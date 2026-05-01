# Naive Bayes: Backend Comparison

All backends implement `(predictions, means, vars_, log_priors) = gaussian_nb_*(X_train, y_train, X_test, n_classes)`. Training always runs on the CPU. The backends differ only in how inference is parallelized.

cuda-python uses a 2D kernel grid `(n_test // 256, n_classes)` — fine-grained control over the parallelism structure. CuPy uses 3D broadcasting, which is concise but may allocate a `(n_test, n_classes, n_features)` intermediate tensor. Numba-CUDA's kernel uses `import math as _math` at module level (inside the `try: from numba import cuda` block) because `numba.cuda.jit` kernels cannot call Python's `math` module at runtime — only names compiled at JIT time are available.

```python
# cuda-python: 2D grid — x covers samples, y covers classes
block  = (256, 1, 1)
grid_x = (n_test + 255) // 256
grid_ll = (grid_x, n_classes, 1)
ll_kernel.launch(grid_ll, block, [...])

# CuPy: broadcasting over (n_test, n_classes, n_features)
log_probs = lp[cp.newaxis, :] + cp.sum(log_gaussian, axis=2)

# Numba: import math at module level (required for jit kernels)
import math as _math   # inside try: from numba import cuda block
@nb_cuda.jit
def _compute_log_likelihood(X_test, means, vars_, log_priors, log_probs):
    ...
    ll += -0.5 * _math.log(2.0 * _math.pi * var) - 0.5 * (x-mu)*(x-mu)/var
```

**Source:** `demos/05_naive_bayes/gpu_nb.py:217–235`, `demos/05_naive_bayes/cupy_nb.py:64–65`, `demos/05_naive_bayes/numba_nb.py`
