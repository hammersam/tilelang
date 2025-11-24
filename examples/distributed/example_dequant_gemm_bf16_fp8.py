import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm import DataType
from tvm import tir
import torch
# from tilelang.engine.callback import register_cuda_postproc_callback

tilelang.disable_cache()
    
@tilelang.jit(out_idx=[-1],)
def matmul(M,
           N,
           K,
           A_dtype,
           B_dtype,
           C_dtype,
           accum_dtype,
           scale_size=32,
           block_M=256,
           block_N=128,
           block_K=64,
           num_stages=2,
           threads=384,
           use_cluster=False):
    
    assert A_dtype in ["bfloat16"]
    assert B_dtype in ["float8_e4m3", "bfloat16"]
    assert C_dtype in ["bfloat16"]
    assert accum_dtype in ["float32"]

    A_shape = (M, K)
    B_shape = (K, N)
    cluster_size = 2
    
    @T.prim_func
    def main_cluster(
            A: T.Tensor(A_shape, A_dtype),
            B: T.Tensor(B_shape, B_dtype),
            C: T.Tensor((M, N), C_dtype),
    ):
        with T.ScopeKernel(
                grid=(T.ceildiv(M, block_M), T.ceildiv(N, block_N), 1),
                cluster=(cluster_size, 1, 1),
                threads=threads):
            bx, by, _ = T.get_block_bindings()
            # FIXME: fix cluster bindings
            # cx, cy, cz = T.get_cluster_bindings()
            cx = bx % cluster_size
            tx = T.get_thread_binding(0)
            idx_in_warpgroup = tx % 128
            warp_idx = idx_in_warpgroup // 32
            lane_idx = idx_in_warpgroup % 32
            
            A_shared = T.alloc_shared((block_M, block_K), A_dtype)
            B_shared = T.alloc_shared((block_K, block_N), "bfloat16")
            C_shared = T.alloc_shared((block_M, block_N), C_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            cur_fp8x16 = T.alloc_local((16,), "float8_e4m3")
            cur_fp32x8_lo = T.alloc_local([8], "float32")
            cur_fp32x8_hi = T.alloc_local([8], "float32")
            cur_bf16x8_lo = T.alloc_local([8], "bfloat16")
            cur_bf16x8_hi = T.alloc_local([8], "bfloat16")
            data_is_ready = T.alloc_barrier(arrive_count=128)
            compute_is_done = T.alloc_barrier(arrive_count=256)
            bar_k_remote_ready = T.alloc_barrier(arrive_count=1)
            # FIXME: adjust arrive count
            bar_k_avail = T.alloc_barrier(arrive_count=512)

            if tx < 256:
                T.clear(C_local)
            
            if tx >= 256:
                T.set_max_nreg(152, 0)
                for ko in T.serial(T.ceildiv(K, block_K)):
                    T.barrier_wait(bar_k_avail, (ko + 1) % 2)
                    if idx_in_warpgroup == 0:
                        T.ptx_arrive_barrier_expect_tx(
                            bar_k_remote_ready[0], 
                            block_K * block_N // cluster_size * 2)
                    T.barrier_wait(compute_is_done, (ko + 1) % 2)
                    T.copy(A[bx * block_M, ko * block_K], A_shared)
                    for ki in T.serial(block_K // cluster_size // 16):
                        offset_k = block_K // cluster_size * cx + ki * 16 + warp_idx * 4 + lane_idx // 8
                        T.load_128b_from_gmem(B[ko * block_K + offset_k, by * block_N + lane_idx % 8 * 16], cur_fp8x16)
                        for i in T.vectorized(8):
                            cur_fp32x8_lo[i] = T.cast(cur_fp8x16[i], "float")
                        for i in T.vectorized(8):
                            cur_bf16x8_lo[i] = T.cast(cur_fp32x8_lo[i], "bfloat16")
                        T.put_thread(
                            src=T.address_of(cur_bf16x8_lo[0]),
                            dst=T.address_of(B_shared[offset_k, lane_idx % 8 * 16]),
                            size=0,
                            mbar=T.address_of(bar_k_remote_ready),
                            dst_pe=(cx + 1) % cluster_size,
                            scope="cluster")
                        for i in T.vectorized(8):
                            # FIXME: check shared memory swizzle
                            B_shared[offset_k, lane_idx % 8 * 16 + i] = cur_bf16x8_lo[i]
                        for i in T.vectorized(8):
                            cur_fp32x8_hi[i] = T.cast(cur_fp8x16[i + 8], "float")
                        for i in T.vectorized(8):
                            cur_bf16x8_hi[i] = T.cast(cur_fp32x8_hi[i], "bfloat16")
                        T.put_thread(
                            src=T.address_of(cur_bf16x8_hi[0]),
                            dst=T.address_of(B_shared[offset_k, lane_idx % 8 * 16 + 8]),
                            size=0,
                            mbar=T.address_of(bar_k_remote_ready),
                            dst_pe=(cx + 1) % cluster_size,
                            scope="cluster")
                        for i in T.vectorized(8):
                            # FIXME: check shared memory swizzle
                            B_shared[offset_k, lane_idx % 8 * 16 + i + 8] = cur_bf16x8_hi[i]
                    T.barrier_arrive(data_is_ready)
            else:
                T.set_max_nreg(160, 1)
                for ko in T.serial(T.ceildiv(K, block_K)):
                    T.barrier_wait(data_is_ready, ko % 2)
                    T.barrier_wait(bar_k_remote_ready, ko % 2)
                    T.gemm(A_shared, B_shared, C_local)
                    T.barrier_arrive(compute_is_done)
                    # TODO: better to provide a cleaner API
                    T.barrier_arrive(bar_k_avail, 0)
                    T.barrier_arrive(bar_k_avail, 1)

                T.copy(C_local, C_shared)
                T.copy(C_shared, C[bx * block_M, by * block_N])
            T.sync_cluster()
            
    @T.prim_func
    def main(
            A: T.Tensor(A_shape, A_dtype),
            B: T.Tensor(B_shape, B_dtype),
            C: T.Tensor((M, N), C_dtype),
    ):
        with T.ScopeKernel(
                grid=(T.ceildiv(M, block_M), T.ceildiv(N, block_N), 1),
                cluster=(2, 1, 1),
                threads=threads):
            bx, by, _ = T.get_block_bindings()
            cx, cy, cz = T.get_cluster_bindings()
            tx = T.get_thread_binding(0)
            idx_in_warpgroup = tx % 128
            warp_idx = idx_in_warpgroup // 32
            lane_idx = idx_in_warpgroup % 32
            
            A_shared = T.alloc_shared((block_M, block_K), A_dtype)
            B_shared = T.alloc_shared((block_K, block_N), "bfloat16")
            C_shared = T.alloc_shared((block_M, block_N), C_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            cur_fp8x16 = T.alloc_local((16,), "float8_e4m3")
            cur_fp32x8 = T.alloc_local([8], "float32")
            cur_bf16x8 = T.alloc_local([8], "bfloat16")
            data_is_ready = T.alloc_barrier(arrive_count=128)
            compute_is_done = T.alloc_barrier(arrive_count=256)

            if tx < 256:
                T.clear(C_local)
            
            if tx >= 256:
                T.set_max_nreg(152, 0)
                for ko in T.serial(T.ceildiv(K, block_K)):
                    T.barrier_wait(compute_is_done, (ko + 1) % 2)
                    T.copy(A[bx * block_M, ko * block_K], A_shared)
                    for ki in T.serial(block_K // 16):
                        T.load_128b_from_gmem(B[ko * block_K + ki * 16 + warp_idx * 4 + lane_idx // 8, by * block_N + lane_idx % 8 * 16], cur_fp8x16)
                        for i in T.vectorized(8):
                            cur_fp32x8[i] = T.cast(cur_fp8x16[i], "float")
                        for i in T.vectorized(8):
                            cur_bf16x8[i] = T.cast(cur_fp32x8[i], "bfloat16")
                        for i in T.vectorized(8):
                            # FIXME: check shared memory swizzle
                            B_shared[ki * 16 + warp_idx * 4 + lane_idx // 8, lane_idx % 8 * 16 + i] = cur_bf16x8[i]
                        for i in T.vectorized(8):
                            cur_fp32x8[i] = T.cast(cur_fp8x16[i + 8], "float")
                        for i in T.vectorized(8):
                            cur_bf16x8[i] = T.cast(cur_fp32x8[i], "bfloat16")
                        for i in T.vectorized(8):
                            # FIXME: check shared memory swizzle
                            B_shared[ki * 16 + warp_idx * 4 + lane_idx // 8, lane_idx % 8 * 16 + i + 8] = cur_bf16x8[i]
                    T.barrier_arrive(data_is_ready)
            else:
                T.set_max_nreg(160, 1)
                for ko in T.serial(T.ceildiv(K, block_K)):
                    T.barrier_wait(data_is_ready, ko % 2)
                    T.gemm(A_shared, B_shared, C_local)
                    T.barrier_arrive(compute_is_done)

                T.copy(C_local, C_shared)
                T.copy(C_shared, C[bx * block_M, by * block_N])
                
    return main_cluster if use_cluster else main

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul_naive(M,
           N,
           K,
           A_dtype,
           B_dtype,
           C_dtype,
           accum_dtype,
           scale_size=32,
           block_M=256,
           block_N=128,
           block_K=64,
           num_stages=2,
           threads=384):
    
    assert A_dtype in ["bfloat16"]
    assert B_dtype in ["float8_e4m3", "bfloat16"]
    assert C_dtype in ["bfloat16"]
    assert accum_dtype in ["float32"]

    A_shape = (M, K)
    B_shape = (K, N)
    
    @T.prim_func
    def main(
            A: T.Tensor(A_shape, "bfloat16"),
            B: T.Tensor(B_shape, "bfloat16"),
            C: T.Tensor((M, N), "bfloat16"),
    ):
        with T.ScopeKernel(
                grid=(T.ceildiv(M, block_M), T.ceildiv(N, block_N), 1),
                cluster=(2, 1, 1),
                threads=threads):
            bx, by, _ = T.get_block_bindings()
            tx = T.get_thread_binding(0)
            
            A_shared = T.alloc_shared((block_M, block_K), A_dtype)
            B_shared = T.alloc_shared((block_K, block_N), B_dtype)
            C_shared = T.alloc_shared((block_M, block_N), C_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            data_is_ready = T.alloc_barrier(arrive_count=128)
            compute_is_done = T.alloc_barrier(arrive_count=256)

            if tx < 256:
                T.clear(C_local)

            
            if tx >= 256:
                T.set_max_nreg(152, 0)
                for ko in T.serial(T.ceildiv(K, block_K)):
                    T.barrier_wait(compute_is_done, (ko + 1) % 2)
                    T.copy(A[bx * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, by * block_N], B_shared)
                    T.barrier_arrive(data_is_ready)
            else:
                T.set_max_nreg(160, 1)
                for ko in T.serial(T.ceildiv(K, block_K)):
                    T.barrier_wait(data_is_ready, ko % 2)
                    T.gemm(A_shared, B_shared, C_local)
                    T.barrier_arrive(compute_is_done)

                T.copy(C_local, C_shared)
                T.copy(C_shared, C[bx * block_M, by * block_N])

    return main



# def ref_program_simple(A, qB, Scale, Bias=None):
#     """
#     Compute a BF16 matrix product A · B^T from a quantized B with simple (non-twiddling) dequantization.

#     Converts the quantized tensor `qB` to floating B via `torch_convert`, applies a per-element scale factor computed as 2^(Scale[i][j//32] - 127) (Scale supplies exponent offsets in 32-column groups), then computes C = A · B^T and returns the result converted to bfloat16.

#     Parameters:
#     - A: 2D tensor representing the left operand (will be cast to float32 for the matmul).
#     - qB: Quantized representation of B accepted by `torch_convert`.
#     - Scale: 2D tensor of exponent offsets; Scale[i][g] is applied to columns j where g == j // 32.

#     Returns:
#     - 2D bfloat16 tensor C containing the matrix product A · B^T.

#     No in-place modification is performed on inputs (a local floating copy of B is scaled).
#     """
#     dtypeC = "bfloat16"
#     B = torch_convert(qB)
#     for i in range(B.shape[0]):
#         for j in range(B.shape[1]):
#             B[i][j] = B[i][j] * (2**(Scale[i][j // 32]))
#     C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
#     C = C.to(torch.__getattribute__(dtypeC))
#     return C

def gemm_ref_bf16_fp8(
    A: torch.Tensor,          # [..., M, K], bf16
    B: torch.Tensor,          # [..., K, N], fp8
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:

    assert A.dtype == torch.bfloat16, f"A must be bf16, got {A.dtype}"
    assert B.dtype in (
        getattr(torch, "float8_e4m3fn", B.dtype),
        getattr(torch, "float8_e5m2", B.dtype),
    ), f"B must be fp8, got {B.dtype}"

    B_bf16 = B.to(torch.bfloat16)

    # GEMM / batched GEMM
    C_f32 = A @ B_bf16

    C = C_f32.to(out_dtype)
    return C


def main(m=256, n=256, k=256, scale_size=32, use_naive=False, use_cluster=False):
    torch.manual_seed(42)
    total_flops = 2 * m * n * k
    fn = matmul_naive if use_naive else matmul
    kernel = fn(
        m,
        n,
        k,
        "bfloat16",
        "float8_e4m3",
        "bfloat16",
        "float32",
        scale_size=scale_size,
        block_M=256,
        block_N=128,
        block_K=64,
        num_stages=2,
        threads=384,
        use_cluster=use_cluster,)
    
    print(kernel.get_kernel_source())
    
    profiler = kernel.get_profiler(tilelang.TensorSupplyType.Auto)

    # if fast_dequant:
    #     if with_bias:
    #         profiler.assert_allclose(ref_program_twiddling_with_bias, rtol=0.01, atol=0.01)
    #     else:
    #         profiler.assert_allclose(ref_program_twiddling, rtol=0.01, atol=0.01)
    # else:
    #     if with_bias:
    #         profiler.assert_allclose(ref_program_simple_with_bias, rtol=0.01, atol=0.01)
    #     else:
    #         profiler.assert_allclose(ref_program_simple, rtol=0.01, atol=0.01)
    # print("All checks pass.")
    A = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    B_fp32 = torch.randn((k, n), dtype=torch.float32, device="cuda")
    B = B_fp32.to(torch.float8_e4m3fn)
    
    C_ref = gemm_ref_bf16_fp8(A, B, out_dtype=torch.bfloat16)
    C = kernel(A, B)
    
    print(f"C: {C}")
    print(f"C_ref: {C_ref}")
    
    rtol = 0.01
    atol = 0.01

    close = torch.isclose(C, C_ref, rtol=rtol, atol=atol)

    num_total = C.numel()
    num_not_close = (~close).sum().item()
    ratio_not_close = num_not_close / num_total if num_total > 0 else 0.0

    diff = torch.abs(C - C_ref)
    max_diff = diff.max().item()

    print(f"not close ratio: {ratio_not_close * 100:.6f}% ({num_not_close}/{num_total})")
    print(f"max abs diff: {max_diff:.6e}")

    if num_not_close == 0:
        print("All checks passed. ✅")
    else:
        print("Some elements are not close enough.")
    
    # latency = profiler.do_bench(warmup=500)
    # print("Tile-lang: {:.2f} ms".format(latency))
    # print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    
    def fn():
        return kernel(A, B)
    
    from tilelang.profiler import do_bench
    latency = do_bench(
        fn,
        rep=10,
        warmup=10,
    )
    print("Tile-lang: {:.2f} ms".format(latency))
    print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))


if __name__ == "__main__":
    M, N, K = 8192, 8192, 8192
    scale_size = 32
    main(M, N, K, scale_size, use_naive=False, use_cluster=True)
