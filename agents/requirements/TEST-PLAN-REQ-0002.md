# Test Plan — REQ-0002: GPU-Accelerated ML Algorithm Demos

**Author**: QA Agent
**Date**: 2026-05-01
**REQ Reference**: agents/requirements/REQ-0002.md
**Implementation**: Sprint 1 (commit on main branch)

---

## Scope

This plan covers the k-means clustering component of REQ-0002 (the only algorithm implemented
in Sprint 1). It tests:
- CPU baseline (`kmeans_cpu`) — output shapes, label validity, inertia, determinism, convergence
- GPU implementation (`kmeans_gpu`) — correctness match against CPU baseline within 1e-3 tolerance

PCA, linear regression, and naive Bayes are deferred to Sprint 2 and are explicitly out of scope here.

Tests are split into CPU-safe (no GPU required) and GPU-required categories.

---

## Test Environment

- Runtime: Python 3.11+, NumPy >= 1.26
- GPU: NVIDIA GPU with CUDA 12.x (required for GPU tests only)
- External services: None
- Test command:
  ```bash
  # All tests (GPU tests auto-skip if no GPU):
  pytest tests/test_kmeans.py -v

  # CPU-only (CI safe):
  pytest tests/test_kmeans.py -v -m "not gpu"

  # GPU tests only:
  pytest tests/test_kmeans.py -v -m gpu
  ```

---

## Test Cases

### Happy Path

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-001 | CPU k-means output shapes correct | Run `kmeans_cpu(X_1000x4, k=3)` | centroids shape `(3, 4)`, labels shape `(1000,)`, inertia `> 0` | No |
| TC-002 | All labels in valid cluster range | Run `kmeans_cpu` on 1000 samples, k=3 | All labels in `{0, 1, 2}` | No |
| TC-003 | Inertia is positive | Run `kmeans_cpu` on random data | `inertia > 0.0` | No |
| TC-004 | CPU k-means is deterministic | Run `kmeans_cpu` twice with same seed | Both centroid arrays are element-wise identical | No |
| TC-005 | CPU k-means converges on well-separated data | Generate 3 tight clusters at [0,0], [10,0], [0,10] (std=0.1, 300 pts each) | Each computed centroid is within 0.5 of its true center | No |
| TC-006 | GPU k-means output matches CPU within tolerance | Run both `kmeans_cpu` and `kmeans_gpu` on same data/seed, sort centroids | `max(|gpu_centroid - cpu_centroid|) < 1e-3` | Yes |

### Error Cases

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-010 | Empty cluster handled gracefully | Use pathological data where k > natural clusters | Does not raise exception; re-initializes empty cluster | No |

### Edge Cases

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-020 | k=1 produces single centroid at data mean | Run `kmeans_cpu(X, k=1)` | centroid shape `(1, n_features)`, all labels == 0 | No |
| TC-021 | k equals n_samples | Run `kmeans_cpu(X_10, k=10)` | centroids shape `(10, n_features)`, no exception | No |
| TC-022 | Single iteration (max_iter=1) returns valid output | Run `kmeans_cpu(X, max_iter=1)` | Returns centroids/labels/inertia without error | No |

---

## Acceptance Criteria Coverage

| AC from REQ-0002 | Test Case(s) | Status |
|------------------|--------------|--------|
| AC-1: User runs k-means demo and sees cluster assignments, inertia, timing | Manual integration test | Out of scope for unit tests |
| AC-2: K-means GPU centroids match CPU within 1e-3 on same seed | TC-006 | Pending |
| AC-3: PCA demo runs and shows explained variance | Sprint 2 — not yet implemented | Deferred |
| AC-4: PCA GPU eigenvectors match NumPy within 1e-4 | Sprint 2 — not yet implemented | Deferred |
| AC-5: Linear regression demo runs and shows R²/MSE | Sprint 2 — not yet implemented | Deferred |
| AC-6: Linear regression GPU coefficients match NumPy within 1e-5 | Sprint 2 — not yet implemented | Deferred |
| AC-7: Each demo prints summary line in correct format | TC-001 (inertia printed), TC-006 (BenchmarkResult format) | Pending |
| AC-8: 100k sample demo achieves >= 2x GPU speedup | Performance test (deferred, requires GPU hardware) | Deferred |

---

## Test Data Strategy

**Well-separated clusters test (TC-005):**
```python
rng = np.random.default_rng(0)
centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
X = np.vstack([rng.normal(c, 0.1, (100, 2)) for c in centers]).astype(np.float32)
```
With std=0.1 and cluster separation of 10 units, the algorithm is guaranteed to converge
to centroids within 0.5 of the true centers regardless of random init.

**GPU vs CPU comparison (TC-006):**
Uses the same `n_samples=500, n_features=4, k=3, seed=42` for both implementations.
Centroids are sorted (by row-wise sum) before comparison to handle cluster-index permutations.

---

## Out of Scope

- PCA, linear regression, naive Bayes (Sprint 2)
- Performance/throughput benchmarks (requires GPU and large datasets)
- Memory leak detection for GPU k-means
- End-to-end `main.py` CLI invocation

---

## Bugs Found

| Bug ID | Description | Severity | Status |
|--------|-------------|----------|--------|
| — | No bugs found during test plan authoring | — | — |
