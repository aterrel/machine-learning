# Linear Regression: GPU XtX Kernel

The `xtx_xty` kernel uses a 2D grid where `blockIdx.x` is the row `i` and `blockIdx.y` is the column `j` of the output `XᵀX` matrix. Each thread computes one dot product by iterating over all `n_samples` rows. Thread `(i, 0)` additionally computes `Xᵀy[i]` while it is already iterating over samples — avoiding a second kernel launch for the right-hand side.

The result is a small `(n_features × n_features)` matrix and `n_features`-length vector copied back to the host for the final `np.linalg.solve`. This pattern — GPU for the expensive quadratic-in-samples computation, CPU for the cheap linear system — is typical when `n_samples >> n_features`.

```cuda
__global__ void xtx_xty(const float* X, const float* y,
                          float* XtX, float* Xty,
                          int n_samples, int n_features) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (i >= n_features || j >= n_features) return;
    float s = 0.0f;
    for (int k = 0; k < n_samples; k++)
        s += X[k * n_features + i] * X[k * n_features + j];
    XtX[i * n_features + j] = s;
    if (j == 0) {  // compute Xty only once per row
        float t = 0.0f;
        for (int k = 0; k < n_samples; k++) t += X[k * n_features + i] * y[k];
        Xty[i] = t;
    }
}
```

**Source:** `demos/04_linear_model/gpu_linear.py:12–29`
