# Device Setup

Before any GPU work, you must select and initialize a CUDA device. `Device(0)` acquires device 0 (the first GPU), and `set_current()` makes it the active device for all subsequent operations in this thread. `get_device_props()` queries the device for its compute capability, total VRAM, and SM architecture — information needed to compile kernels for the correct target.

The compute capability determines which CUDA features are available (e.g., `sm_89` for Ada Lovelace). NVRTC uses this to select the instruction set when compiling kernels. Always query it programmatically rather than hard-coding it, so the same code works across different GPUs.

```python
from cuda.core.experimental import Device

device = Device(0)
device.set_current()

props = get_device_props(device)
major, minor = props["compute_capability"]
print(f"Device name        : {props['name']}")
print(f"Compute capability : {major}.{minor}")
print(f"Total VRAM         : {props['total_memory_gb']:.2f} GB")
print(f"SM architecture    : sm_{props['arch']}")
```

**Source:** `demos/01_core_apis/device_info.py:14–31`
