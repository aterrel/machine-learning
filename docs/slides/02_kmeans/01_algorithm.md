# K-Means Clustering: Algorithm

K-means (Lloyd's algorithm) iterates two steps until convergence: (1) assign each sample to the nearest centroid, and (2) recompute each centroid as the mean of its assigned samples. "Nearest" is measured by squared Euclidean distance. The algorithm stops when centroid shifts fall below a tolerance `tol` or after `max_iter` iterations.

Both steps are naturally parallel: assignment is embarrassingly parallel over samples (each sample is independent), and accumulation uses atomic adds to aggregate sums across threads. This makes k-means an ideal first GPU algorithm — the parallelism structure maps cleanly to CUDA's 1D thread grid.

```
Input:  X (n_samples, n_features), k clusters, max_iter, tol

Loop until converged:
  1. Assign:     labels[i] = argmin_c  ||X[i] - centroids[c]||^2
  2. Accumulate: new_centroids[c] += X[i]  for all i where labels[i] == c
  3. Normalize:  centroids[c] = new_centroids[c] / count[c]
  4. Check:      max shift < tol  →  break

Output: centroids (k, n_features), labels (n_samples,), inertia
```

**Source:** `demos/02_kmeans/gpu_kmeans.py:66–72` (function signature and docstring)
