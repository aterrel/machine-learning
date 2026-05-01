"""GPU k-means using custom CUDA kernels (Lloyd's algorithm)."""

from __future__ import annotations

import sys

import numpy as np

# Kernel 1: assign each sample the label of the nearest centroid
ASSIGN_LABELS_SRC = r"""
extern "C" __global__
void assign_labels(
    const float* X,
    const float* centroids,
    int* labels,
    int n_samples,
    int n_features,
    int k
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_samples) return;

    float best_dist = 1e38f;
    int best_label = 0;
    for (int c = 0; c < k; c++) {
        float dist = 0.0f;
        for (int f = 0; f < n_features; f++) {
            float diff = X[i * n_features + f] - centroids[c * n_features + f];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_label = c;
        }
    }
    labels[i] = best_label;
}
"""

# Kernel 2: accumulate sums for centroid update via atomicAdd
ACCUMULATE_CENTROIDS_SRC = r"""
extern "C" __global__
void accumulate_centroids(
    const float* X,
    const int* labels,
    float* new_centroids,
    int* counts,
    int n_samples,
    int n_features,
    int k
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_samples) return;

    int c = labels[i];
    atomicAdd(&counts[c], 1);
    for (int f = 0; f < n_features; f++) {
        atomicAdd(&new_centroids[c * n_features + f], X[i * n_features + f]);
    }
}
"""

BLOCK_SIZE = 256


def kmeans_gpu(
    X: np.ndarray,
    k: int = 8,
    max_iter: int = 100,
    seed: int = 42,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """GPU k-means via custom CUDA kernels.

    Returns (centroids (k, n_features), labels (n_samples,), inertia).
    """
    try:
        from cuda.bindings import cudart

        from src.kernels.compiler import KernelCompiler
        from src.utils.device import get_device
        from src.utils.memory import DeviceBuffer
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        sys.exit(1)

    try:
        device = get_device(0)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    X = X.astype(np.float32)
    n_samples, n_features = X.shape

    # Same random centroid initialisation as CPU variant (same seed, same RNG)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_samples, size=k, replace=False)
    centroids = X[idx].copy()

    stream = device.create_stream()
    compiler = KernelCompiler(device)
    assign_kernel = compiler.compile(
        ASSIGN_LABELS_SRC, "assign_labels", cache_key="assign_labels", stream=stream
    )
    accum_kernel = compiler.compile(
        ACCUMULATE_CENTROIDS_SRC,
        "accumulate_centroids",
        cache_key="accumulate_centroids",
        stream=stream,
    )

    x_bytes = X.nbytes
    c_bytes = centroids.nbytes
    labels_bytes = n_samples * np.dtype(np.int32).itemsize
    counts_bytes = k * np.dtype(np.int32).itemsize

    # Separate RNG for empty-cluster reinitialisation; seed+1 avoids correlation
    # with the centroid initialisation RNG above.
    rng_empty = np.random.default_rng(seed + 1)

    grid_samples = assign_kernel.compute_grid_1d(n_samples, BLOCK_SIZE)
    block = (BLOCK_SIZE, 1, 1)

    def _h2d(dst_handle: int, src_arr: np.ndarray, nbytes: int) -> None:
        (err,) = cudart.cudaMemcpy(
            dst_handle,
            src_arr.ctypes.data,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if err.value != 0:
            raise RuntimeError(f"H2D copy failed: {err.value}")

    def _d2h(dst_arr: np.ndarray, src_handle: int, nbytes: int) -> None:
        (err,) = cudart.cudaMemcpy(
            dst_arr.ctypes.data,
            src_handle,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        if err.value != 0:
            raise RuntimeError(f"D2H copy failed: {err.value}")

    with (
        DeviceBuffer(x_bytes, stream=stream, device=device) as d_X,
        DeviceBuffer(c_bytes, stream=stream, device=device) as d_centroids,
        DeviceBuffer(labels_bytes, stream=stream, device=device) as d_labels,
        DeviceBuffer(c_bytes, stream=stream, device=device) as d_new_centroids,
        DeviceBuffer(counts_bytes, stream=stream, device=device) as d_counts,
    ):
        _h2d(d_X.handle, X, x_bytes)

        labels_host = np.empty(n_samples, dtype=np.int32)

        for _ in range(max_iter):
            _h2d(d_centroids.handle, centroids, c_bytes)

            # Assign labels
            assign_kernel.launch(
                grid_samples,
                block,
                [
                    d_X.handle,
                    d_centroids.handle,
                    d_labels.handle,
                    np.int32(n_samples),
                    np.int32(n_features),
                    np.int32(k),
                ],
                stream=stream,
            )

            # Zero out accumulation buffers via cudaMemset
            (err,) = cudart.cudaMemset(d_new_centroids.handle, 0, c_bytes)
            if err.value != 0:
                raise RuntimeError(f"cudaMemset centroids failed: {err.value}")
            (err,) = cudart.cudaMemset(d_counts.handle, 0, counts_bytes)
            if err.value != 0:
                raise RuntimeError(f"cudaMemset counts failed: {err.value}")

            # Accumulate centroid sums
            accum_kernel.launch(
                grid_samples,
                block,
                [
                    d_X.handle,
                    d_labels.handle,
                    d_new_centroids.handle,
                    d_counts.handle,
                    np.int32(n_samples),
                    np.int32(n_features),
                    np.int32(k),
                ],
                stream=stream,
            )
            stream.sync()

            # Normalize centroids on CPU
            new_centroids_host = np.empty_like(centroids)
            counts_host = np.empty(k, dtype=np.int32)
            _d2h(new_centroids_host, d_new_centroids.handle, c_bytes)
            _d2h(counts_host, d_counts.handle, counts_bytes)

            for j in range(k):
                if counts_host[j] > 0:
                    new_centroids_host[j] /= counts_host[j]
                else:
                    # Empty cluster: reinitialise to random sample (CPU-side)
                    new_centroids_host[j] = X[rng_empty.integers(n_samples)]

            shift = float(np.max(np.linalg.norm(new_centroids_host - centroids, axis=1)))
            centroids = new_centroids_host

            if shift < tol:
                break

        # Final label assignment to get host labels
        _h2d(d_centroids.handle, centroids, c_bytes)
        assign_kernel.launch(
            grid_samples,
            block,
            [
                d_X.handle,
                d_centroids.handle,
                d_labels.handle,
                np.int32(n_samples),
                np.int32(n_features),
                np.int32(k),
            ],
            stream=stream,
        )
        stream.sync()
        _d2h(labels_host, d_labels.handle, labels_bytes)

    stream.sync()
    stream.close()

    # Compute inertia on CPU
    dists_final = np.sum((X - centroids[labels_host]) ** 2, axis=1)
    inertia = float(dists_final.sum())

    return centroids, labels_host, inertia
