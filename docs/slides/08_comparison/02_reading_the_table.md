# Reading the Comparison Table

`python demos/08_comparison/main.py` prints a table with one row per backend. The **Time (s)** column is the mean wall time over 3 timed runs (after 1 warmup run). The **Speedup** column is `numpy_time / backend_time`. The **Correct** column verifies that the GPU output matches the NumPy reference within a tolerance; `--` appears for the NumPy baseline row. `SKIPPED` appears for any backend that is not installed.

Interpret the table carefully: speedup includes data transfer time (H2D + D2H). For small `n_samples` (default 10,000) the transfer overhead often dominates and speedup can be < 1×. Re-run with `--n-samples 500000` to see compute-bound behavior where GPU backends pull ahead.

```
$ python demos/08_comparison/main.py --algorithm kmeans --n-samples 50000

============================================================
Algorithm: K-Means  (n_samples=50000, k=8)
============================================================
Backend          Time (s)   Speedup    Correct
------------------------------------------------------------
numpy (CPU)        8.412      1.0x      --
cuda-python        0.243     34.6x      True
cupy               0.187     45.0x      True
numba              0.261     32.2x      True
cuml               0.091     92.4x      True
------------------------------------------------------------

Run with: python demos/08_comparison/main.py --algorithm all --n-samples 50000
```

**Source:** `demos/08_comparison/main.py`, `benchmarks/run_all.py`
