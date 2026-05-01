# Kernel Launch

A kernel launch requires three things: a grid shape `(grid_x, grid_y, grid_z)`, a block shape `(block_x, block_y, block_z)`, and a list of kernel arguments. Arguments are passed as Python objects — integers, floats, and raw device pointers (integers). The `CompiledKernel.launch()` method marshals these to the C types CUDA expects.

Grid size is computed as `ceil(n / block_size)`: enough blocks to cover all elements, with each thread checking `if (i < n)` to avoid out-of-bounds access. Always synchronize the stream after launching if you need results on the host before the next step.

```python
BLOCK_SIZE = 256
n = 1_000_000

grid  = kernel.compute_grid_1d(n, BLOCK_SIZE)  # ((n + 255) // 256, 1, 1)
block = (BLOCK_SIZE, 1, 1)

kernel.launch(
    grid,
    block,
    [d_a.handle, d_b.handle, d_c.handle, np.int32(n)],
    stream=stream,
)
stream.sync()  # wait for kernel to finish before D2H copy
```

**Source:** `demos/01_core_apis/vector_add.py:74–91`
