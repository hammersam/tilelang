import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
import argparse
import torch

tilelang.disable_cache()


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })
def gemm_with_multicast(M, N, K, block_M, block_N, block_K, threads, cluster, dtype="float16"):
    """
    GEMM with TMA multicast within clusters.

    Args:
        M, N, K: Matrix dimensions (A: M×K, B: K×N, C: M×N)
        block_M, block_N, block_K: Block tile sizes
        threads: Number of threads per block
        cluster: Cluster dimensions (cluster_M, cluster_N, cluster_Z)
        dtype: Data type

    Design:
        - Grid: (M // block_M, N // block_N)
        - Cluster: cluster_M blocks along M-axis share the same B tile
        - Each block computes C[bx*block_M:(bx+1)*block_M, by*block_N:(by+1)*block_N]
        - Only cx==0 blocks load B tiles and multicast to the entire cluster
    """

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), "float32"),
    ):
        with T.ScopeKernel(
                grid=(T.ceildiv(M, block_M), T.ceildiv(N, block_N), 1),
                cluster=cluster,
                threads=threads):

            bx, by, _ = T.get_block_bindings()
            cx = T.get_cluster_binding(0)

            # Shared memory allocation
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            # Register allocation for accumulator
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)

            # Create TMA descriptor for B matrix
            B_desc = T.create_tma_descriptor(B, [K, N], [block_K, block_N])

            # Main computation loop over K dimension
            for k in T.serial(T.ceildiv(K, block_K)):
                # Load A tile: each block loads its own portion
                T.copy(A[bx * block_M, k * block_K], A_shared)

                # Load B tile: only cx==0 blocks load and multicast to cluster
                if cx == 0:
                    T.tma_load_multicast(
                        descriptor=B_desc,
                        smem_addr=T.address_of(B_shared[0, 0]),
                        multicast_mask=(1 << cluster[0]) - 1,  # Broadcast to all blocks in cluster
                        coords=[k * block_K, by * block_N]
                    )

                # Cluster-level synchronization
                T.sync_cluster()

                # Compute: C_local += A_shared @ B_shared
                T.gemm(A_shared, B_shared, C_local)

            # Write results back to global memory
            T.copy(C_local, C[bx * block_M, by * block_N])

    return main


def ref_program(A, B):
    """Reference GEMM implementation using PyTorch."""
    return torch.matmul(A.float(), B.float())


def main(M=256, N=256, K=256, cluster=None):
    """
    Main test function.

    Args:
        M, N, K: Matrix dimensions
        cluster: Cluster configuration tuple (cluster_M, cluster_N, cluster_Z)
    """
    # Configuration
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    threads = 128

    if cluster is None:
        cluster = (2, 1, 1)

    print(f"Testing GEMM with multicast:")
    print(f"  Matrix sizes: A({M}×{K}) × B({K}×{N}) = C({M}×{N})")
    print(f"  Block sizes: ({BLOCK_M}, {BLOCK_N}, {BLOCK_K})")
    print(f"  Cluster config: {cluster}")
    print(f"  Threads per block: {threads}")
    print()

    # Create kernel
    kernel = gemm_with_multicast(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, threads, cluster)

    # Print kernel source (optional, for debugging)
    # print("Generated kernel source:")
    # print(kernel.get_kernel_source())
    # print()

    # Generate random input matrices
    A = torch.randn((M, K), dtype=torch.float16).cuda()
    B = torch.randn((K, N), dtype=torch.float16).cuda()

    # Run TileLang kernel
    print("Running TileLang kernel...")
    C_tl = kernel(A, B)

    # Run reference implementation
    print("Running reference implementation...")
    C_ref = ref_program(A, B)

    # Verify correctness
    print("Verifying results...")
    max_diff = torch.max(torch.abs(C_tl - C_ref)).item()
    mean_diff = torch.mean(torch.abs(C_tl - C_ref)).item()

    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    # Check with tolerance
    rtol = 1e-2
    atol = 1e-2
    passed = torch.allclose(C_tl, C_ref, rtol=rtol, atol=atol)

    if passed:
        print(f"✅ All checks passed! (rtol={rtol}, atol={atol})")
    else:
        print(f"❌ Test failed! Differences exceed tolerance (rtol={rtol}, atol={atol})")
        # Print some sample values for debugging
        print(f"\nSample values (first 4×4):")
        print(f"TileLang output:\n{C_tl[:4, :4]}")
        print(f"Reference output:\n{C_ref[:4, :4]}")
        print(f"Difference:\n{(C_tl - C_ref)[:4, :4]}")

    # Performance benchmark (optional)
    if passed:
        print("\nRunning performance benchmark...")

        # Benchmark TileLang kernel using tilelang.profiler.do_bench
        avg_time_ms = do_bench(lambda: kernel(A, B), warmup=25, rep=100)

        flops = 2 * M * N * K  # Multiply-add counts as 2 ops
        tflops = (flops / avg_time_ms / 1e9)  # TFLOPS

        print(f"  Average time: {avg_time_ms:.3f} ms")
        print(f"  Performance: {tflops:.2f} TFLOPS")

        # Compare with PyTorch
        avg_time_torch_ms = do_bench(lambda: torch.matmul(A.float(), B.float()), warmup=25, rep=100)
        tflops_torch = (flops / avg_time_torch_ms / 1e9)

        print(f"\nPyTorch baseline:")
        print(f"  Average time: {avg_time_torch_ms:.3f} ms")
        print(f"  Performance: {tflops_torch:.2f} TFLOPS")
        print(f"  Speedup: {avg_time_torch_ms / avg_time_ms:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEMM with TMA multicast within clusters")
    parser.add_argument('--M', type=int, default=256, help='M dimension (rows of A)')
    parser.add_argument('--N', type=int, default=256, help='N dimension (cols of B)')
    parser.add_argument('--K', type=int, default=256, help='K dimension (cols of A, rows of B)')
    parser.add_argument("--cluster", type=int, nargs='+', default=[2, 1, 1],
                       help="Cluster size (cluster_M, cluster_N, cluster_Z)")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    cluster = tuple(args.cluster)

    main(M, N, K, cluster)
