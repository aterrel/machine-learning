"""CLI benchmark runner: runs all available demos and prints a summary table.

Usage:
  python benchmarks/run_all.py                        # cuda-python, all demos
  python benchmarks/run_all.py --backend numba        # Numba variants only
  python benchmarks/run_all.py --backend cupy         # CuPy variants only
  python benchmarks/run_all.py --backend all          # 3-way comparison table
  python benchmarks/run_all.py --demo 02_kmeans       # single demo
"""

from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_row(name: str, cpu_s: float, gpu_s: float, speedup: float, correct: bool) -> str:
    return (
        f"  {name:<22} CPU: {cpu_s:7.3f}s  GPU: {gpu_s:7.3f}s"
        f"  Speedup: {speedup:5.1f}x  Correct: {correct}"
    )


def _fmt_cmp_header() -> str:
    return f"  {'Backend':<14} {'GPU(s)':>8}  {'Speedup':>8}  {'Correct':>7}"


def _fmt_cmp_row(backend: str, gpu_s: float, speedup: float, correct) -> str:
    if correct is None:
        return f"  {backend:<14} {'SKIPPED (not installed)':>30}"
    return f"  {backend:<14} {gpu_s:8.3f}  {speedup:8.1f}x  {str(correct):>7}"


# ---------------------------------------------------------------------------
# Single-backend run functions (cuda-python)
# ---------------------------------------------------------------------------

def run_01_core_apis(n_samples: int) -> list:
    results = []
    try:
        va_mod = importlib.import_module("demos.01_core_apis.vector_add")
        r = va_mod.run_vector_add(n=n_samples)
        results.append(r)
        print(_fmt_row(r.demo_name, r.cpu_time_mean_s, r.gpu_time_mean_s, r.speedup, r.correct))
    except Exception as exc:
        print(f"  01_core_apis/vector_add FAILED: {exc}")
    return results


def run_02_kmeans(n_samples: int) -> list:
    results = []
    try:
        from src.utils.timing import BenchmarkResult, BenchmarkRunner

        cpu_mod = importlib.import_module("demos.02_kmeans.cpu_kmeans")
        gpu_mod = importlib.import_module("demos.02_kmeans.gpu_kmeans")

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_samples, 32)).astype(np.float32)
        runner = BenchmarkRunner(n_repeats=3, warmup=1)

        centroids_cpu, _, _ = cpu_mod.kmeans_cpu(X, k=8, seed=42)
        cpu_time = runner.time_cpu(cpu_mod.kmeans_cpu, X, k=8, seed=42)

        centroids_gpu, _, _ = gpu_mod.kmeans_gpu(X, k=8, seed=42)
        gpu_time = runner.time_cpu(gpu_mod.kmeans_gpu, X, k=8, seed=42)

        max_err = float(np.max(np.abs(np.sort(centroids_cpu, axis=0) - np.sort(centroids_gpu, axis=0))))
        correct = max_err < 1e-3
        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        r = BenchmarkResult(demo_name="kmeans", cpu_time_mean_s=cpu_time, gpu_time_mean_s=gpu_time,
                            speedup=speedup, correct=correct, max_abs_error=max_err)
        results.append(r)
        print(_fmt_row(r.demo_name, r.cpu_time_mean_s, r.gpu_time_mean_s, r.speedup, r.correct))
    except Exception as exc:
        print(f"  02_kmeans FAILED: {exc}")
    return results


def run_03_pca(n_samples: int) -> list:
    results = []
    try:
        from src.utils.timing import BenchmarkResult, BenchmarkRunner

        cpu_mod = importlib.import_module("demos.03_pca.cpu_pca")
        gpu_mod = importlib.import_module("demos.03_pca.gpu_pca")

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_samples, 16)).astype(np.float32)
        runner = BenchmarkRunner(n_repeats=3, warmup=1)

        cpu_comps, _, _ = cpu_mod.pca_cpu(X, n_components=2)
        cpu_time = runner.time_cpu(cpu_mod.pca_cpu, X, n_components=2)

        gpu_comps, _, _ = gpu_mod.pca_gpu(X, n_components=2)
        gpu_time = runner.time_cpu(gpu_mod.pca_gpu, X, n_components=2)

        alignment = float(np.min(np.abs(np.diag(cpu_comps @ gpu_comps.T))))
        correct = alignment > 1 - 1e-3
        max_err = float(1.0 - alignment)
        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        r = BenchmarkResult(demo_name="pca", cpu_time_mean_s=cpu_time, gpu_time_mean_s=gpu_time,
                            speedup=speedup, correct=correct, max_abs_error=max_err)
        results.append(r)
        print(_fmt_row(r.demo_name, r.cpu_time_mean_s, r.gpu_time_mean_s, r.speedup, r.correct))
    except Exception as exc:
        print(f"  03_pca FAILED: {exc}")
    return results


def run_04_linear_model(n_samples: int) -> list:
    results = []
    try:
        from src.utils.timing import BenchmarkResult, BenchmarkRunner

        cpu_mod = importlib.import_module("demos.04_linear_model.cpu_linear")
        gpu_mod = importlib.import_module("demos.04_linear_model.gpu_linear")

        rng = np.random.default_rng(42)
        n_features = 32
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        w_true = rng.standard_normal(n_features).astype(np.float32)
        y = X @ w_true + 0.01 * rng.standard_normal(n_samples).astype(np.float32)
        runner = BenchmarkRunner(n_repeats=3, warmup=1)

        w_cpu, _, _ = cpu_mod.linear_regression_cpu(X, y)
        cpu_time = runner.time_cpu(cpu_mod.linear_regression_cpu, X, y)

        w_gpu, _, _ = gpu_mod.linear_regression_gpu(X, y)
        gpu_time = runner.time_cpu(gpu_mod.linear_regression_gpu, X, y)

        max_err = float(np.max(np.abs(w_cpu - w_gpu)))
        correct = max_err < 1e-3
        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        r = BenchmarkResult(demo_name="linear_regression", cpu_time_mean_s=cpu_time,
                            gpu_time_mean_s=gpu_time, speedup=speedup, correct=correct, max_abs_error=max_err)
        results.append(r)
        print(_fmt_row(r.demo_name, r.cpu_time_mean_s, r.gpu_time_mean_s, r.speedup, r.correct))
    except Exception as exc:
        print(f"  04_linear_model FAILED: {exc}")
    return results


def run_05_kernels(_n_samples: int) -> list:
    results = []
    try:
        gemm_mod = importlib.import_module("demos.05_kernels.gemm")
        act_mod = importlib.import_module("demos.05_kernels.activations")

        r_gemm = gemm_mod.run_gemm_demo()
        results.append(r_gemm)
        print(_fmt_row(r_gemm.demo_name, r_gemm.cpu_time_mean_s, r_gemm.gpu_time_mean_s, r_gemm.speedup, r_gemm.correct))

        r_act = act_mod.run_activations_demo()
        results.append(r_act)
        print(_fmt_row(r_act.demo_name, r_act.cpu_time_mean_s, r_act.gpu_time_mean_s, r_act.speedup, r_act.correct))
    except Exception as exc:
        print(f"  05_kernels FAILED: {exc}")
    return results


def run_05_naive_bayes(n_samples: int) -> list:
    results = []
    try:
        from src.utils.timing import BenchmarkResult, BenchmarkRunner

        cpu_mod = importlib.import_module("demos.05_naive_bayes.cpu_nb")
        gpu_mod = importlib.import_module("demos.05_naive_bayes.gpu_nb")

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

        runner = BenchmarkRunner(n_repeats=3, warmup=1)

        cpu_preds, *_ = cpu_mod.gaussian_nb_cpu(X_tr, y_tr, X_te, n_classes=n_classes)
        cpu_time = runner.time_cpu(cpu_mod.gaussian_nb_cpu, X_tr, y_tr, X_te, n_classes=n_classes)

        gpu_preds, *_ = gpu_mod.gaussian_nb_gpu(X_tr, y_tr, X_te, n_classes=n_classes)
        gpu_time = runner.time_cpu(gpu_mod.gaussian_nb_gpu, X_tr, y_tr, X_te, n_classes=n_classes)

        agreement = float((cpu_preds == gpu_preds).mean())
        correct = agreement > 0.99
        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        r = BenchmarkResult(demo_name="naive_bayes", cpu_time_mean_s=cpu_time,
                            gpu_time_mean_s=gpu_time, speedup=speedup, correct=correct,
                            max_abs_error=float(1.0 - agreement))
        results.append(r)
        print(_fmt_row(r.demo_name, r.cpu_time_mean_s, r.gpu_time_mean_s, r.speedup, r.correct))
    except Exception as exc:
        print(f"  05_naive_bayes FAILED: {exc}")
    return results


# ---------------------------------------------------------------------------
# Multi-backend comparison helpers
# ---------------------------------------------------------------------------

def _time_backend(fn, *args, warmup: int = 1, n_repeats: int = 3, **kwargs) -> float:
    """Wall-clock time fn(*args, **kwargs), with warmup. Returns mean seconds."""
    import time
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def _run_kmeans_comparison(n_samples: int) -> None:
    """Print 3-way comparison table for k-means."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, 32)).astype(np.float32)

    cpu_mod = importlib.import_module("demos.02_kmeans.cpu_kmeans")
    _, _, _ = cpu_mod.kmeans_cpu(X, k=8, seed=42)
    cpu_time = _time_backend(cpu_mod.kmeans_cpu, X, k=8, seed=42)
    ref_centroids, _, _ = cpu_mod.kmeans_cpu(X, k=8, seed=42)

    print(f"\n[02_kmeans]  n_samples={n_samples}")
    print(f"  CPU (NumPy)    {cpu_time:8.3f}s  (baseline)")
    print(_fmt_cmp_header())

    backends = [
        ("cuda-python", "demos.02_kmeans.gpu_kmeans", "kmeans_gpu"),
        ("numba",       "demos.02_kmeans.numba_kmeans", "kmeans_numba"),
        ("cupy",        "demos.02_kmeans.cupy_kmeans",  "kmeans_cupy"),
    ]
    for bname, modpath, fnname in backends:
        try:
            mod = importlib.import_module(modpath)
            fn = getattr(mod, fnname)
            gpu_time = _time_backend(fn, X, k=8, seed=42)
            cents, _, _ = fn(X, k=8, seed=42)
            max_err = float(np.max(np.abs(np.sort(ref_centroids, axis=0) - np.sort(cents, axis=0))))
            correct = max_err < 1e-3
            speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
            print(_fmt_cmp_row(bname, gpu_time, speedup, correct))
        except ImportError:
            print(_fmt_cmp_row(bname, 0, 0, None))
        except SystemExit:
            print(_fmt_cmp_row(bname, 0, 0, None))
        except Exception as exc:
            print(f"  {bname:<14} FAILED: {exc}")


def _run_pca_comparison(n_samples: int) -> None:
    """Print 3-way comparison table for PCA."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, 16)).astype(np.float32)

    cpu_mod = importlib.import_module("demos.03_pca.cpu_pca")
    ref_comps, _, _ = cpu_mod.pca_cpu(X, n_components=2)
    cpu_time = _time_backend(cpu_mod.pca_cpu, X, n_components=2)

    print(f"\n[03_pca]  n_samples={n_samples}")
    print(f"  CPU (NumPy)    {cpu_time:8.3f}s  (baseline)")
    print(_fmt_cmp_header())

    backends = [
        ("cuda-python", "demos.03_pca.gpu_pca",         "pca_gpu"),
        ("numba",       "demos.03_pca.numba_pca",        "pca_numba"),
        ("cupy",        "demos.03_pca.cupy_pca",         "pca_cupy"),
    ]
    for bname, modpath, fnname in backends:
        try:
            mod = importlib.import_module(modpath)
            fn = getattr(mod, fnname)
            gpu_time = _time_backend(fn, X, n_components=2)
            comps, _, _ = fn(X, n_components=2)
            alignment = float(np.min(np.abs(np.diag(ref_comps @ comps.T))))
            correct = alignment > 1 - 1e-3
            speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
            print(_fmt_cmp_row(bname, gpu_time, speedup, correct))
        except ImportError:
            print(_fmt_cmp_row(bname, 0, 0, None))
        except SystemExit:
            print(_fmt_cmp_row(bname, 0, 0, None))
        except Exception as exc:
            print(f"  {bname:<14} FAILED: {exc}")


def _run_linear_comparison(n_samples: int) -> None:
    """Print 3-way comparison table for linear regression."""
    rng = np.random.default_rng(42)
    n_features = 32
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    w_true = rng.standard_normal(n_features).astype(np.float32)
    y = X @ w_true + 0.01 * rng.standard_normal(n_samples).astype(np.float32)

    cpu_mod = importlib.import_module("demos.04_linear_model.cpu_linear")
    w_ref, _, _ = cpu_mod.linear_regression_cpu(X, y)
    cpu_time = _time_backend(cpu_mod.linear_regression_cpu, X, y)

    print(f"\n[04_linear_model]  n_samples={n_samples}")
    print(f"  CPU (NumPy)    {cpu_time:8.3f}s  (baseline)")
    print(_fmt_cmp_header())

    backends = [
        ("cuda-python", "demos.04_linear_model.gpu_linear",    "linear_regression_gpu"),
        ("numba",       "demos.04_linear_model.numba_linear",   "linear_regression_numba"),
        ("cupy",        "demos.04_linear_model.cupy_linear",    "linear_regression_cupy"),
    ]
    for bname, modpath, fnname in backends:
        try:
            mod = importlib.import_module(modpath)
            fn = getattr(mod, fnname)
            gpu_time = _time_backend(fn, X, y)
            w_gpu, _, _ = fn(X, y)
            max_err = float(np.max(np.abs(w_ref - w_gpu)))
            correct = max_err < 1e-3
            speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
            print(_fmt_cmp_row(bname, gpu_time, speedup, correct))
        except ImportError:
            print(_fmt_cmp_row(bname, 0, 0, None))
        except SystemExit:
            print(_fmt_cmp_row(bname, 0, 0, None))
        except Exception as exc:
            print(f"  {bname:<14} FAILED: {exc}")


def _run_nb_comparison(n_samples: int) -> None:
    """Print 3-way comparison table for Gaussian Naive Bayes."""
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

    cpu_mod = importlib.import_module("demos.05_naive_bayes.cpu_nb")
    ref_preds, *_ = cpu_mod.gaussian_nb_cpu(X_tr, y_tr, X_te, n_classes=n_classes)
    cpu_time = _time_backend(cpu_mod.gaussian_nb_cpu, X_tr, y_tr, X_te, n_classes=n_classes)

    print(f"\n[05_naive_bayes]  n_samples={n_samples}")
    print(f"  CPU (NumPy)    {cpu_time:8.3f}s  (baseline)")
    print(_fmt_cmp_header())

    backends = [
        ("cuda-python", "demos.05_naive_bayes.gpu_nb",      "gaussian_nb_gpu"),
        ("numba",       "demos.05_naive_bayes.numba_nb",     "gaussian_nb_numba"),
        ("cupy",        "demos.05_naive_bayes.cupy_nb",      "gaussian_nb_cupy"),
    ]
    for bname, modpath, fnname in backends:
        try:
            mod = importlib.import_module(modpath)
            fn = getattr(mod, fnname)
            gpu_time = _time_backend(fn, X_tr, y_tr, X_te, n_classes=n_classes)
            preds, *_ = fn(X_tr, y_tr, X_te, n_classes=n_classes)
            agreement = float((ref_preds == preds).mean())
            correct = agreement > 0.99
            speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
            print(_fmt_cmp_row(bname, gpu_time, speedup, correct))
        except ImportError:
            print(_fmt_cmp_row(bname, 0, 0, None))
        except SystemExit:
            print(_fmt_cmp_row(bname, 0, 0, None))
        except Exception as exc:
            print(f"  {bname:<14} FAILED: {exc}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all CUDA Python ML demo benchmarks")
    parser.add_argument(
        "--demo",
        choices=["01_core_apis", "02_kmeans", "03_pca", "04_linear_model", "05_kernels", "05_naive_bayes", "all"],
        default="all",
    )
    parser.add_argument(
        "--backend",
        choices=["cuda-python", "numba", "cupy", "all"],
        default="cuda-python",
        help="Backend to benchmark. 'all' prints a 3-way comparison table.",
    )
    parser.add_argument("--n-samples", type=int, default=10_000)
    args = parser.parse_args()

    # --backend all: multi-column comparison mode (no CUDA availability check needed — backends skip gracefully)
    if args.backend == "all":
        print("=" * 70)
        print("CUDA Python ML Demos — Multi-Backend Comparison")
        print("=" * 70)
        if args.demo in ("02_kmeans", "all"):
            _run_kmeans_comparison(args.n_samples)
        if args.demo in ("03_pca", "all"):
            _run_pca_comparison(args.n_samples)
        if args.demo in ("04_linear_model", "all"):
            _run_linear_comparison(args.n_samples)
        if args.demo in ("05_naive_bayes", "all"):
            _run_nb_comparison(args.n_samples)
        print("\n" + "=" * 70)
        return

    # Single-backend modes
    if args.backend == "numba":
        try:
            import numba  # noqa: F401
        except ImportError:
            print("Numba not available — install with: pip install numba")
            sys.exit(1)
    elif args.backend == "cupy":
        try:
            import cupy  # noqa: F401
        except ImportError:
            print("CuPy not available — install with: pip install cupy-cuda12x")
            sys.exit(1)
    else:
        # cuda-python backend: check CUDA available
        try:
            from src.utils.device import check_cuda_available
        except ImportError as exc:
            print(f"ERROR: src/utils not available: {exc}")
            sys.exit(1)
        if not check_cuda_available():
            print(
                "ERROR: No CUDA GPU detected or cuda-python not installed.\n"
                "Install CUDA Toolkit 12.x and run: pip install cuda-python"
            )
            sys.exit(1)

    print("=" * 70)
    print(f"CUDA Python ML Demos — Benchmark Summary (backend: {args.backend})")
    print("=" * 70)

    all_results = []

    if args.backend == "cuda-python":
        if args.demo in ("01_core_apis", "all"):
            print("\n[01_core_apis]")
            all_results.extend(run_01_core_apis(args.n_samples))
        if args.demo in ("02_kmeans", "all"):
            print("\n[02_kmeans]")
            all_results.extend(run_02_kmeans(args.n_samples))
        if args.demo in ("03_pca", "all"):
            print("\n[03_pca]")
            all_results.extend(run_03_pca(args.n_samples))
        if args.demo in ("04_linear_model", "all"):
            print("\n[04_linear_model]")
            all_results.extend(run_04_linear_model(args.n_samples))
        if args.demo in ("05_kernels", "all"):
            print("\n[05_kernels]")
            all_results.extend(run_05_kernels(args.n_samples))
        if args.demo in ("05_naive_bayes", "all"):
            print("\n[05_naive_bayes]")
            all_results.extend(run_05_naive_bayes(args.n_samples))

    elif args.backend in ("numba", "cupy"):
        # Route through the comparison helpers — they handle missing backends gracefully.
        # The requested backend row will show timing; absent backends show SKIPPED.
        if args.demo in ("02_kmeans", "all"):
            _run_kmeans_comparison(args.n_samples)
        if args.demo in ("03_pca", "all"):
            _run_pca_comparison(args.n_samples)
        if args.demo in ("04_linear_model", "all"):
            _run_linear_comparison(args.n_samples)
        if args.demo in ("05_naive_bayes", "all"):
            _run_nb_comparison(args.n_samples)

    print("\n" + "=" * 70)
    if all_results:
        all_correct = all(r.correct for r in all_results)
        print(f"Total demos run: {len(all_results)}")
        print(f"All correct    : {all_correct}")
    print("=" * 70)


if __name__ == "__main__":
    main()
