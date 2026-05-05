"""CPU-safe unit tests for PTXTracer (Sprint 8)."""
from __future__ import annotations

import pytest
from src.kernel_model import DeviceSpec, PTXTracer

# ---------------------------------------------------------------------------
# PTX fixtures as module-level strings
# ---------------------------------------------------------------------------

_PTX_VECTOR_ADD = """
.version 8.0
.target sm_80
.address_size 64
.visible .entry k() {
    .reg .f32 %f<3>;
    .reg .u64 %rd<3>;
    ld.global.f32   %f0, [%rd0];
    ld.global.f32   %f1, [%rd1];
    fma.rn.f32      %f2, %f0, 1.0, %f1;
    st.global.f32   [%rd2], %f2;
    ret;
}
"""

_PTX_MMA_SYNC = """
.version 8.0
.target sm_80
.address_size 64
.visible .entry k() {
    .reg .f32 %f<4>;
    .reg .b16 %h<6>;
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {%f0,%f1,%f2,%f3}, {%h0,%h1,%h2,%h3}, {%h4,%h5}, {%f0,%f1,%f2,%f3};
    ret;
}
"""

_PTX_WGMMA = """
.version 8.5
.target sm_90
.address_size 64
.visible .entry k() {
    .reg .f32 %f<8>;
    wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
        {%f0,%f1,%f2,%f3,%f4,%f5,%f6,%f7}, [smem_a], [smem_b], 1, 1, 1;
    ret;
}
"""

_PTX_TCGEN05 = """
.version 8.5
.target sm_100
.address_size 64
.visible .entry k() {
    .reg .b32 %r0;
    tcgen05.alloc.cta_group::1.sync.aligned %r0, 64;
    tcgen05.mma.cta_group::1.kind::f16 [%r0], [smem_a], [smem_b], 1, 1;
    tcgen05.dealloc.cta_group::1.sync.aligned %r0, 64;
    ret;
}
"""

_PTX_SHARED_DECL = """
.version 8.0
.target sm_80
.address_size 64
.visible .entry k() {
    .shared .align 128 .b8 smem_buf[8192];
    ret;
}
"""

_PTX_ASYNC_GROUPS = """
.version 8.0
.target sm_80
.address_size 64
.visible .entry k() {
    .reg .u64 %rd<2>;
    cp.async.ca.shared.global [smem_a], [%rd0], 16;
    cp.async.commit_group;
    cp.async.ca.shared.global [smem_b], [%rd1], 16;
    cp.async.commit_group;
    cp.async.ca.shared.global [smem_c], [%rd1], 16;
    cp.async.commit_group;
    cp.async.wait_all;
    ret;
}
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_trace_vector_add():
    device = DeviceSpec.from_name("a100")
    tracer = PTXTracer(device)
    result = tracer.trace(_PTX_VECTOR_ADD)
    assert result.by_category.get("global_mem", 0) >= 2
    assert result.by_category.get("fused_fp", 0) >= 1
    assert result.mma_warp_count == 0
    assert result.mma_warpgroup_count == 0
    assert result.mma_cta_count == 0


def test_trace_mma_sync():
    device = DeviceSpec.from_name("a100")
    tracer = PTXTracer(device)
    result = tracer.trace(_PTX_MMA_SYNC)
    assert result.mma_warp_count == 1
    assert result.mma_warpgroup_count == 0
    assert result.mma_cta_count == 0
    assert result.sm_version == "sm_80"


def test_trace_wgmma():
    device = DeviceSpec.from_name("h100")
    tracer = PTXTracer(device)
    result = tracer.trace(_PTX_WGMMA)
    assert result.mma_warpgroup_count == 1
    assert result.mma_warp_count == 0
    assert result.mma_cta_count == 0


def test_trace_tcgen05():
    device = DeviceSpec.from_name("rtx5090")
    tracer = PTXTracer(device)
    result = tracer.trace(_PTX_TCGEN05)
    assert result.mma_cta_count == 1
    assert result.mma_warp_count == 0


def test_arch_mismatch_warning():
    device = DeviceSpec.from_name("a100")   # sm_80
    tracer = PTXTracer(device)
    result = tracer.trace(_PTX_TCGEN05)    # contains tcgen05 (sm_100)
    assert len(result.unsupported_instructions) > 0
    assert any("tcgen05" in instr for instr in result.unsupported_instructions)
    assert len(result.notes) > 0


def test_smem_declaration_parse():
    device = DeviceSpec.from_name("a100")
    tracer = PTXTracer(device)
    result = tracer.trace(_PTX_SHARED_DECL)
    assert result.smem_alloc_bytes == 8192


def test_tmem_alloc_parse():
    device = DeviceSpec.from_name("rtx5090")  # sm_100
    tracer = PTXTracer(device)
    result = tracer.trace(_PTX_TCGEN05)
    # tcgen05.alloc with n=64 → 64 * 512 = 32768
    assert result.tmem_alloc_bytes == 32768


def test_async_copy_group_count():
    device = DeviceSpec.from_name("a100")
    tracer = PTXTracer(device)
    result = tracer.trace(_PTX_ASYNC_GROUPS)
    assert result.async_copy_groups == 3


def test_unknown_instruction():
    device = DeviceSpec.from_name("a100")
    tracer = PTXTracer(device)
    ptx = """
.version 8.0
.target sm_80
.address_size 64
.visible .entry k() {
    xyzzy.foo.bar %r0, %r1;
    ret;
}
"""
    result = tracer.trace(ptx)  # must not raise
    assert result.by_category.get("other", 0) >= 1


def test_bottleneck_compute():
    # Many MMA instructions → compute bound
    device = DeviceSpec.from_name("a100")
    tracer = PTXTracer(device)
    # Build PTX with 50 mma.sync and 1 global load
    mma_lines = "\n    ".join(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f0,%f1,%f2,%f3}, {%h0,%h1,%h2,%h3}, {%h4,%h5}, {%f0,%f1,%f2,%f3};"
        for _ in range(50)
    )
    ptx = f"""
.version 8.0
.target sm_80
.address_size 64
.visible .entry k() {{
    .reg .f32 %f<4>;
    .reg .b16 %h<6>;
    .reg .u64 %rd0;
    ld.global.f32 %f0, [%rd0];
    {mma_lines}
    ret;
}}
"""
    result = tracer.trace(ptx)
    assert result.bottleneck == "compute"


def test_bottleneck_memory():
    # Many global loads, no MMA → memory bound
    device = DeviceSpec.from_name("a100")
    tracer = PTXTracer(device)
    loads = "\n    ".join(f"ld.global.f32 %f0, [%rd0];" for _ in range(50))
    ptx = f"""
.version 8.0
.target sm_80
.address_size 64
.visible .entry k() {{
    .reg .f32 %f0;
    .reg .u64 %rd0;
    {loads}
    ret;
}}
"""
    result = tracer.trace(ptx)
    assert result.bottleneck == "memory"


def test_performance_10k_lines():
    import time
    device = DeviceSpec.from_name("a100")
    tracer = PTXTracer(device)
    line = "    ld.global.f32 %f0, [%rd0];"
    ptx_header = ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry k() {\n    .reg .f32 %f0;\n    .reg .u64 %rd0;\n"
    ptx_footer = "\n    ret;\n}"
    ptx = ptx_header + "\n".join([line] * 10_000) + ptx_footer
    t0 = time.perf_counter()
    tracer.trace(ptx)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"Traced 10k lines in {elapsed:.3f}s (expected < 1.0s)"


def test_arch_table_coverage():
    # Every DeviceSpec SKU sm_version must be in _ARCH_TABLE
    from src.kernel_model._arch_table import _ARCH_TABLE
    names = ["v100", "a100-40gb", "a100", "h100", "b100", "rtx3090", "rtx4090", "rtx5090"]
    for name in names:
        spec = DeviceSpec.from_name(name)
        assert spec.sm_version in _ARCH_TABLE, f"{name} → {spec.sm_version} not in _ARCH_TABLE"
