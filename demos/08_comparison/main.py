"""Unified multi-backend comparison demo.

Runs the same ML algorithm across all available GPU backends and prints
a side-by-side timing table.  Backends that are not installed show as SKIPPED.

Usage:
  python demos/08_comparison/main.py
  python demos/08_comparison/main.py --algorithm kmeans
  python demos/08_comparison/main.py --n-samples 50000
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

# (display_name, module_path, function_name)
_KMEANS_BACKENDS = [
    ("numpy (CPU)",  "demos.02_kmeans.cpu_kmeans",     "kmeans_cpu"),
    ("cuda-python",  "demos.02_kmeans.gpu_kmeans",     "kmeans_gpu"),
    ("cupy",         "demos.02_kmeans.cupy_kmeans",    "kmeans_cupy"),
    ("numba",        "demos.02_kmeans.numba_kmeans",   "kmeans_numba"),
    ("cuml",         "demos.08_comparison.backends.cuml_backends", "kmeans_cuml"),
]

_PCA_BACKENDS = [
    ("numpy (CPU)",  "demos.03_pca.cpu_pca",           "pca_cpu"),
    ("cuda-python",  "demos.03_pca.gpu_pca",           "pca_gpu"),
    ("cupy",         "demos.03_pca.cupy_pca",          "pca_cupy"),
    ("numba",        "demos.03_pca.numba_pca",         "pca_numba"),
    ("cuml",         "demos.08_comparison.backends.cuml_backends", "pca_cuml"),
]

_LINEAR_BACKENDS = [
    ("numpy (CPU)",  "demos.04_linear_model.cpu_linear",   "linear_regression_cpu"),
    ("cuda-python",  "demos.04_linear_model.gpu_linear",   "linear_regression_gpu"),
    ("cupy",         "demos.04_linear_model.cupy_linear",  "linear_regression_cupy"),
    ("numba",        "demos.04_linear_model.numba_linear", "linear_regression_numba"),
    ("cuml",         "demos.08_comparison.backends.cuml_backends", "linear_regression_cuml"),
]

_NB_BACKENDS = [
    ("numpy (CPU)",  "demos.05_naive_bayes.cpu_nb",     "gaussian_nb_cpu"),
    ("cuda-python",  "demos.05_naive_bayes.gpu_nb",     "gaussian_nb_gpu"),
    ("cupy",         "demos.05_naive_bayes.cupy_nb",    "gaussian_nb_cupy"),
    ("numba",        "demos.05_naive_bayes.numba_nb",   "gaussian_nb_numba"),
    ("cuml",         "demos.08_comparison.backends.cuml_backends", "gaussian_nb_cuml"),
]

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _timed(fn, *args, warmup: int = 1, n_repeats: int = 3, **kwargs) -> tuple[float, object]:
    """Run fn(*args) with warmup, return (mean_seconds, last_result)."""
    result = None
    for _ in range(warmup):
        result = fn(*args, **kwargs)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), result

# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_COL = 14  # backend name column width

def _print_table_header() -> None:
    print(f"  {'Backend':<{_COL}} {'Time(s)':>8}  {'Speedup':>9}  {'Correct':>8}")
    print("  " + "-" * 46)


def _print_table_row(name: str, elapsed: float, speedup: float, correct) -> None:
    if correct is None:
        print(f"  {name:<{_COL}} {'SKIPPED (not installed)':>38}")
    else:
        print(f"  {name:<{_COL}} {elapsed:8.3f}s  {speedup:8.1f}x  {str(correct):>8}")


def _print_baseline_row(name: str, elapsed: float) -> None:
    print(f"  {name:<{_COL}} {elapsed:8.3f}s  (baseline)")

# ---------------------------------------------------------------------------
# Per-algorithm comparison runners
# ---------------------------------------------------------------------------

def run_kmeans_comparison(n_samples: int) -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, 32)).astype(np.float32)

    print(f"\n{'='*60}")
    print(f"Algorithm: K-Means  (n_samples={n_samples}, k=8)")
    print(f"{'='*60}")
    _print_table_header()

    cpu_time = None
    ref_centroids = None

    for name, modpath, fnname in _KMEANS_BACKENDS:
        try:
            mod = importlib.import_module(modpath)
            fn = getattr(mod, fnname)
            elapsed, result = _timed(fn, X, k=8, seed=42)
            centroids, _, _ = result if isinstance(result, tuple) else (result, None, None)
            centroids = np.asarray(centroids)

            if cpu_time is None:
                cpu_time = elapsed
                ref_centroids = centroids
                _print_baseline_row(name, elapsed)
            else:
                speedup = cpu_time / elapsed if elapsed > 0 else float("inf")
                max_err = float(np.max(np.abs(np.sort(ref_centroids, axis=0) - np.sort(centroids, axis=0))))
                correct = max_err < 1e-3
                _print_table_row(name, elapsed, speedup, correct)
        except ImportError:
            _print_table_row(name, 0, 0, None)
        except SystemExit:
            print(f"  {name:<{_COL}} SKIPPED (no GPU / cuda-python not installed)")
        except Exception as exc:
            print(f"  {name:<{_COL}} ERROR: {exc}")


def run_pca_comparison(n_samples: int) -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, 16)).astype(np.float32)

    print(f"\n{'='*60}")
    print(f"Algorithm: PCA  (n_samples={n_samples}, n_components=2)")
    print(f"{'='*60}")
    _print_table_header()

    cpu_time = None
    ref_comps = None

    for name, modpath, fnname in _PCA_BACKENDS:
        try:
            mod = importlib.import_module(modpath)
            fn = getattr(mod, fnname)
            elapsed, result = _timed(fn, X, n_components=2)
            comps, _, _ = result if isinstance(result, tuple) else (result, None, None)
            comps = np.asarray(comps)

            if cpu_time is None:
                cpu_time = elapsed
                ref_comps = comps
                _print_baseline_row(name, elapsed)
            else:
                speedup = cpu_time / elapsed if elapsed > 0 else float("inf")
                alignment = float(np.min(np.abs(np.diag(ref_comps @ comps.T))))
                correct = alignment > 1 - 1e-3
                _print_table_row(name, elapsed, speedup, correct)
        except ImportError:
            _print_table_row(name, 0, 0, None)
        except SystemExit:
            print(f"  {name:<{_COL}} SKIPPED (no GPU / cuda-python not installed)")
        except Exception as exc:
            print(f"  {name:<{_COL}} ERROR: {exc}")


def run_linear_comparison(n_samples: int) -> None:
    rng = np.random.default_rng(42)
    n_features = 32
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    w_true = rng.standard_normal(n_features).astype(np.float32)
    y = X @ w_true + 0.01 * rng.standard_normal(n_samples).astype(np.float32)

    print(f"\n{'='*60}")
    print(f"Algorithm: Linear Regression  (n_samples={n_samples}, n_features={n_features})")
    print(f"{'='*60}")
    _print_table_header()

    cpu_time = None
    ref_w = None

    for name, modpath, fnname in _LINEAR_BACKENDS:
        try:
            mod = importlib.import_module(modpath)
            fn = getattr(mod, fnname)
            elapsed, result = _timed(fn, X, y)
            w, _, _ = result if isinstance(result, tuple) else (result, None, None)
            w = np.asarray(w)

            if cpu_time is None:
                cpu_time = elapsed
                ref_w = w
                _print_baseline_row(name, elapsed)
            else:
                speedup = cpu_time / elapsed if elapsed > 0 else float("inf")
                max_err = float(np.max(np.abs(ref_w - w)))
                correct = max_err < 1e-3
                _print_table_row(name, elapsed, speedup, correct)
        except ImportError:
            _print_table_row(name, 0, 0, None)
        except SystemExit:
            print(f"  {name:<{_COL}} SKIPPED (no GPU / cuda-python not installed)")
        except Exception as exc:
            print(f"  {name:<{_COL}} ERROR: {exc}")


def run_naive_bayes_comparison(n_samples: int) -> None:
    rng = np.random.default_rng(42)
    n_classes, n_features = 4, 16
    n_train = int(n_samples * 0.8)
    n_test = n_samples - n_train
    centers = rng.standard_normal((n_classes, n_features)) * 5.0
    X_parts, y_parts = [], []
    for c in range(n_classes):
        n = n_train // n_classes
        X_parts.append(rng.standard_normal((n, n_features)) * 0.5 + centers[c])
        y_parts.append(np.full(n, c, dtype=np.int32))
    X_tr = np.concatenate(X_parts).astype(np.float32)
    y_tr = np.concatenate(y_parts)
    X_te = rng.standard_normal((n_test, n_features)).astype(np.float32)

    print(f"\n{'='*60}")
    print(f"Algorithm: Gaussian Naive Bayes  (n_samples={n_samples}, n_classes={n_classes})")
    print(f"{'='*60}")
    _print_table_header()

    cpu_time = None
    ref_preds = None

    for name, modpath, fnname in _NB_BACKENDS:
        try:
            mod = importlib.import_module(modpath)
            fn = getattr(mod, fnname)
            elapsed, result = _timed(fn, X_tr, y_tr, X_te, n_classes=n_classes)
            preds, *_ = result if isinstance(result, tuple) else (result,)
            preds = np.asarray(preds)

            if cpu_time is None:
                cpu_time = elapsed
                ref_preds = preds
                _print_baseline_row(name, elapsed)
            else:
                speedup = cpu_time / elapsed if elapsed > 0 else float("inf")
                agreement = float((ref_preds == preds).mean())
                correct = agreement > 0.99
                _print_table_row(name, elapsed, speedup, correct)
        except ImportError:
            _print_table_row(name, 0, 0, None)
        except SystemExit:
            print(f"  {name:<{_COL}} SKIPPED (no GPU / cuda-python not installed)")
        except Exception as exc:
            print(f"  {name:<{_COL}} ERROR: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified multi-backend ML comparison (NumPy / cuda-python / CuPy / Numba / cuML)"
    )
    parser.add_argument(
        "--algorithm",
        choices=["kmeans", "pca", "linear", "naive_bayes", "all"],
        default="all",
    )
    parser.add_argument("--n-samples", type=int, default=10_000)
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Backend ML Comparison")
    print("Backends: NumPy / cuda-python / CuPy / Numba / cuML")
    print("SKIPPED = backend not installed")
    print("=" * 60)

    if args.algorithm in ("kmeans", "all"):
        run_kmeans_comparison(args.n_samples)
    if args.algorithm in ("pca", "all"):
        run_pca_comparison(args.n_samples)
    if args.algorithm in ("linear", "all"):
        run_linear_comparison(args.n_samples)
    if args.algorithm in ("naive_bayes", "all"):
        run_naive_bayes_comparison(args.n_samples)

    print()


if __name__ == "__main__":
    main()
