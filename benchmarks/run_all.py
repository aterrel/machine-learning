"""CLI benchmark runner: runs all available demos and prints a summary table."""

from __future__ import annotations

import argparse
import importlib
import sys


def _fmt_row(name: str, cpu_s: float, gpu_s: float, speedup: float, correct: bool) -> str:
    return (
        f"  {name:<20} CPU: {cpu_s:7.3f}s  GPU: {gpu_s:7.3f}s"
        f"  Speedup: {speedup:5.1f}x  Correct: {correct}"
    )


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
        import numpy as np

        from src.utils.timing import BenchmarkResult, BenchmarkRunner

        cpu_mod = importlib.import_module("demos.02_kmeans.cpu_kmeans")
        gpu_mod = importlib.import_module("demos.02_kmeans.gpu_kmeans")

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_samples, 32)).astype(np.float32)
        runner = BenchmarkRunner(n_repeats=3, warmup=1)

        centroids_cpu, _, _ = cpu_mod.kmeans_cpu(X, k=8, seed=42)
        cpu_time = runner.time_cpu(cpu_mod.kmeans_cpu, X, k=8, seed=42)

        centroids_gpu, _, _ = gpu_mod.kmeans_gpu(X, k=8, seed=42)
        # wall-clock timing; kmeans_gpu manages its own stream internally
        gpu_time = runner.time_cpu(gpu_mod.kmeans_gpu, X, k=8, seed=42)

        max_err = float(
            np.max(
                np.abs(np.sort(centroids_cpu, axis=0) - np.sort(centroids_gpu, axis=0))
            )
        )
        correct = max_err < 1e-3
        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        r = BenchmarkResult(
            demo_name="kmeans",
            cpu_time_mean_s=cpu_time,
            gpu_time_mean_s=gpu_time,
            speedup=speedup,
            correct=correct,
            max_abs_error=max_err,
        )
        results.append(r)
        print(_fmt_row(r.demo_name, r.cpu_time_mean_s, r.gpu_time_mean_s, r.speedup, r.correct))
    except Exception as exc:
        print(f"  02_kmeans FAILED: {exc}")
    return results


def run_03_pca(n_samples: int) -> list:
    results = []
    try:
        import numpy as np

        from src.utils.timing import BenchmarkResult, BenchmarkRunner

        cpu_mod = importlib.import_module("demos.03_pca.cpu_pca")
        gpu_mod = importlib.import_module("demos.03_pca.gpu_pca")

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_samples, 16)).astype(np.float32)
        runner = BenchmarkRunner(n_repeats=3, warmup=1)

        cpu_comps, cpu_var, _ = cpu_mod.pca_cpu(X, n_components=2)
        cpu_time = runner.time_cpu(cpu_mod.pca_cpu, X, n_components=2)

        gpu_comps, gpu_var, _ = gpu_mod.pca_gpu(X, n_components=2)
        gpu_time = runner.time_cpu(gpu_mod.pca_gpu, X, n_components=2)

        alignment = float(np.min(np.abs(np.diag(cpu_comps @ gpu_comps.T))))
        correct = alignment > 1 - 1e-3
        max_err = float(1.0 - alignment)
        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        r = BenchmarkResult(
            demo_name="pca",
            cpu_time_mean_s=cpu_time,
            gpu_time_mean_s=gpu_time,
            speedup=speedup,
            correct=correct,
            max_abs_error=max_err,
        )
        results.append(r)
        print(_fmt_row(r.demo_name, r.cpu_time_mean_s, r.gpu_time_mean_s, r.speedup, r.correct))
    except Exception as exc:
        print(f"  03_pca FAILED: {exc}")
    return results


def run_04_linear_model(n_samples: int) -> list:
    results = []
    try:
        import numpy as np

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
        r = BenchmarkResult(
            demo_name="linear_regression",
            cpu_time_mean_s=cpu_time,
            gpu_time_mean_s=gpu_time,
            speedup=speedup,
            correct=correct,
            max_abs_error=max_err,
        )
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
        import numpy as np

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
        # wall-clock timing; gaussian_nb_gpu manages its own stream internally

        agreement = float((cpu_preds == gpu_preds).mean())
        correct = agreement > 0.99
        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        r = BenchmarkResult(
            demo_name="naive_bayes",
            cpu_time_mean_s=cpu_time,
            gpu_time_mean_s=gpu_time,
            speedup=speedup,
            correct=correct,
            max_abs_error=float(1.0 - agreement),
        )
        results.append(r)
        print(_fmt_row(r.demo_name, r.cpu_time_mean_s, r.gpu_time_mean_s, r.speedup, r.correct))
    except Exception as exc:
        print(f"  05_naive_bayes FAILED: {exc}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all CUDA Python ML demo benchmarks")
    parser.add_argument(
        "--demo",
        choices=["01_core_apis", "02_kmeans", "03_pca", "04_linear_model", "05_kernels", "05_naive_bayes", "all"],
        default="all",
    )
    parser.add_argument("--n-samples", type=int, default=10_000)
    args = parser.parse_args()

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
    print("CUDA Python ML Demos — Benchmark Summary")
    print("=" * 70)

    all_results = []

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

    print("\n" + "=" * 70)
    print(f"Total demos run: {len(all_results)}")
    all_correct = all(r.correct for r in all_results)
    print(f"All correct    : {all_correct}")
    print("=" * 70)


if __name__ == "__main__":
    main()
