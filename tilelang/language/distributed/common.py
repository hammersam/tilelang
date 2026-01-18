"""The language interface for tl programs."""
from __future__ import annotations

from tvm import tir
from tvm.tir import PrimExpr


def get_rank():
    """Get the rank of the current process.
    """
    return tir.call_intrin("uint64", tir.op.Op.get("tl.get_rank"))


def get_num_ranks():
    """Get the number of processes.
    """
    return tir.call_intrin("uint64", tir.op.Op.get("tl.get_num_ranks"))


def put_thread(src: PrimExpr,
               dst: PrimExpr,
               size: PrimExpr,
               mbar: PrimExpr | None = None,
               dst_pe: PrimExpr | None = None,
               unroll_factor: int = 4,
               scope: str = "gpu"):
    """Put to a remote buffer with unrolled loop.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the put in elements.
        dst_pe: PrimExpr | None
            The PE index of the destination.
            If provided, the dst is a symmetric address, otherwise it is a UVA address.
            If not provided, the dst is a UVA address and dst_pe is None.
        unroll_factor: int
            The unroll factor
        scope: str
            The copy scopy, can be gpu or cluster.
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.put"), src, dst, size, dst_pe, unroll_factor,
                           scope, "thread", mbar)


def put_warp(src: PrimExpr,
             dst: PrimExpr,
             size: PrimExpr,
             dst_pe: PrimExpr | None = None,
             unroll_factor: int = 4,
             scope: str = "gpu"):
    """Put to a remote buffer with unrolled loop.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the put in elements.
        dst_pe: PrimExpr | None
            The PE index of the destination.
            If provided, the dst is a symmetric address, otherwise it is a UVA address.
            If not provided, the dst is a UVA address and dst_pe is None.
        unroll_factor: int
            The unroll factor
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.put"), src, dst, size, dst_pe, unroll_factor,
                           scope, "warp")


def get_warp(src: PrimExpr,
             dst: PrimExpr,
             size: PrimExpr,
             src_pe: PrimExpr | None = None,
             unroll_factor: int = 4):
    """Get from a remote buffer with unrolled loop.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the get in elements.
        src_pe: PrimExpr | None
            The PE index of the source.
            If provided, the src is a symmetric address, otherwise it is a UVA address.
            If not provided, the src is a UVA address and src_pe is None.
        unroll_factor: int
            The unroll factor
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.get"), src, dst, size, src_pe, unroll_factor,
                           "warp")


def put_block(src: PrimExpr,
              dst: PrimExpr,
              size: PrimExpr,
              dst_pe: PrimExpr | None = None,
              scope: str = "gpu"):
    """Put to a remote buffer.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the put in elements.
        dst_pe: PrimExpr | None
            The PE index of the destination.
            If provided, the dst is a symmetric address, otherwise it is a UVA address.
            If not provided, the dst is a UVA address and dst_pe is None.
        scope: str
            The copy scopy, can be gpu or cluster.
    """
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.put"), src, dst, size, dst_pe, 0, scope, "block"
    )  # NOTE(wt): unroll_factor is not needed because currently we implement block-level comm based on NVSHMEM-style copy


def get_block(src: PrimExpr, dst: PrimExpr, size: PrimExpr, src_pe: PrimExpr | None = None):
    """Get from a remote buffer.

    Args:
        src: PrimExpr
            The source address.
        dst: PrimExpr
            The destination address.
        size: PrimExpr
            The size of the get in elements.
        src_pe: PrimExpr | None
            The PE index of the source.
            If provided, the src is a symmetric address, otherwise it is a UVA address.
            If not provided, the src is a UVA address and src_pe is None.
    """
    return tir.call_intrin(
        "handle", tir.op.Op.get("tl.get"), src, dst, size, src_pe, 0, "block"
    )  # NOTE(wt): unroll_factor is not needed because currently we implement block-level comm based on NVSHMEM-style copy


def tma_load_multicast(descriptor: PrimExpr,
                       smem_addr: PrimExpr,
                       multicast_mask: PrimExpr,
                       coords: list,
                       eviction_policy: int = 0):
    """TMA load with multicast to multiple blocks in a cluster.

    This function performs a TMA (Tensor Memory Accelerator) load operation
    that broadcasts the loaded data to multiple blocks within a cluster.

    Args:
        descriptor: PrimExpr
            The TMA descriptor for the tensor to load.
        smem_addr: PrimExpr
            The shared memory address to store the loaded data.
        multicast_mask: PrimExpr
            A 16-bit mask indicating which blocks in the cluster should
            receive the data. For example:
            - mask=0b11 broadcasts to 2 blocks (block 0 and 1)
            - mask=0b1111 broadcasts to 4 blocks
        coords: list[PrimExpr]
            The coordinates in the tensor to load from.
        eviction_policy: int
            The L2 cache eviction policy (0=EVICT_NORMAL, 1=EVICT_FIRST,
            2=EVICT_LAST). Defaults to 0.

    Returns:
        A TIR call expression representing the TMA load multicast operation.

    Example:
        ```python
        with T.ScopeKernel(grid=(64, 64, 1), cluster=(2, 1, 1), threads=128):
            cx = T.get_cluster_binding(0)

            # Only block 0 in each cluster performs the load
            if cx == 0:
                # Load and broadcast to both blocks in the cluster
                T.tma_load_multicast(
                    descriptor=desc,
                    smem_addr=T.address_of(A_shared[0, 0]),
                    multicast_mask=0b11,  # Broadcast to 2 blocks
                    coords=[coord_x, coord_y]
                )

            T.sync_cluster()
        ```
    """
    args = [descriptor, 0, smem_addr, multicast_mask]  # 0 is mbarrier placeholder
    args.extend(coords)
    args.append(eviction_policy)
    return tir.call_intrin("handle", tir.op.Op.get("tl.tma_load_multicast"), *args)
