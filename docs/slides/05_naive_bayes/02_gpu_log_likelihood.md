# Naive Bayes: GPU Log-Likelihood Kernel

The `compute_log_likelihood` kernel launches a 2D grid: the x-dimension covers test samples, the y-dimension covers classes. Each thread handles one `(sample, class)` pair — computing the full Gaussian log-likelihood for that pair by looping over all features. A second kernel `argmax_predictions` then reduces each row of the `(n_test, n_classes)` log-probability matrix to the winning class index.

Using `logf` (single-precision log) and working in log-space avoids numerical underflow that would occur if you multiplied probabilities directly (which become extremely small for high-dimensional data).

```cuda
__global__ void compute_log_likelihood(
    const float* X_test, const float* class_means,
    const float* class_vars, const float* log_priors,
    float* log_probs, int n_test, int n_features, int n_classes) {

    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    int cls    = blockIdx.y;
    if (sample >= n_test || cls >= n_classes) return;

    float ll = log_priors[cls];
    for (int f = 0; f < n_features; f++) {
        float x   = X_test[sample * n_features + f];
        float mu  = class_means[cls * n_features + f];
        float var = class_vars[cls * n_features + f];
        ll += -0.5f * logf(2.0f * 3.14159265f * var)
              - 0.5f * (x - mu) * (x - mu) / var;
    }
    log_probs[sample * n_classes + cls] = ll;
}
```

**Source:** `demos/05_naive_bayes/gpu_nb.py:13–37`
