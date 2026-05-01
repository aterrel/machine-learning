# K-Means: NumPy CPU Baseline

The NumPy baseline uses broadcast subtraction to compute all `(n_samples, k)` pairwise distances in one expression, then `argmin` to assign labels. This is idiomatic NumPy — readable and correct — but materializes the full `(n_samples, k, n_features)` intermediate tensor, which can be large. The update step loops over `k` classes on the CPU.

The CPU baseline sets the correctness bar: every GPU variant must produce centroids and labels that agree with this reference. All backends use the same random seed for centroid initialization so results are directly comparable.

```python
# Assignment step: all pairwise squared distances in one broadcast
# dists shape: (n_samples, k)
dists = np.sum(
    (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
    axis=2,
)
new_labels = np.argmin(dists, axis=1).astype(np.int32)

# Update step
for j in range(k):
    mask = new_labels == j
    if mask.any():
        new_centroids[j] = X[mask].mean(axis=0)
    else:
        new_centroids[j] = X[rng.integers(n_samples)]  # empty cluster
```

**Source:** `demos/02_kmeans/cpu_kmeans.py:29–45`
