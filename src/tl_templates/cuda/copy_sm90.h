#pragma once

#ifndef __CUDACC_RTC__
#include <cuda.h>
#endif

#include "barrier.h"
#include "common.h"

namespace tl {
enum class CacheHintSm90 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000,
};

template <typename BarrierType = uint64_t>
TL_DEVICE void tma_load(void *smem_ptr, void *gmem_ptr, BarrierType &smem_mbar,
                        uint32_t size) {
  uint32_t smem_int_mbar =
      smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
               "bytes [%0], [%1], %2, [%3]; \n" ::"r"(smem_int_ptr),
               "l"(gmem_ptr), "r"(size), "r"(smem_int_mbar)
               :);
}

/*

// 1D bulk copy, no tensor descriptor
// PTX: cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
// 适用于：连续内存块的简单拷贝

// 源地址和目标地址都是线性地址
void* src = global_ptr + offset;      // 全局内存线性地址
void* dst = smem_ptr;                 // 共享内存地址
size_t bytes = 2048;                  // 拷贝字节数
uint16_t mask = 0b11;                 // 广播到 2 个 block

tl::tma_load_multicast(dst, src, bytes, mbar, mask);

*/

TL_DEVICE void tma_load_multicast(void *smem_ptr, void *gmem_ptr,
                                  uint64_t &smem_mbar, uint32_t size,
                                  uint16_t mask) {
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes."
      "multicast::cluster [%0], [%1], %2, [%3], %4; \n" ::"r"(smem_int_ptr),
      "l"(gmem_ptr), "r"(size), "r"(smem_int_mbar), "h"(mask)
      :);
}

/*

// TensorMap 包含的信息
struct TensorMapInfo {
    void* base_address;        // 全局内存基地址
    uint64_t shape[5];         // 张量形状 (最多5D)
    uint64_t stride[5];        // 每个维度的 stride
    uint32_t box_size[5];      // 每次加载的 tile 大小
    DataType dtype;            // 数据类型
    SwizzleMode swizzle;       // Swizzle 模式 (优化 bank conflict)
    // ...
};

# 在 tilelang 中
desc = T.create_tma_descriptor(
    tensor=A,           # 源张量
    shape=[M, K],       # 张量形状
    tile_shape=[64, 64] # 每次加载的 tile 大小
)

*/

/*

// PTX: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint
// 适用于：复杂张量访问模式
// 使用坐标而不是线性地址
CUtensorMap descriptor;  // 预先创建的张量描述符
int coord_x = tile_x * TILE_M;        // X 坐标
int coord_y = tile_y * TILE_K;        // Y 坐标
uint16_t mask = 0b11;                 // 广播到 2 个 block

tl::tma_load_multicast(descriptor, mbar, smem_ptr, mask, coord_x, coord_y);

*/

// TMA load with multicast using TensorMap descriptor (2D)
template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_multicast(const CUtensorMap &descriptor,
                                  BarrierType &smem_mbar,
                                  void const *const smem_ptr,
                                  uint16_t const &multicast_mask,
                                  int32_t const &crd0, int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5, %6;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0),
        "r"(crd1), "h"(multicast_mask), "l"(cache_hint)
      : "memory");
}

// TMA load with multicast using TensorMap descriptor (3D)
template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_multicast(const CUtensorMap &descriptor,
                                  BarrierType &smem_mbar,
                                  void const *const smem_ptr,
                                  uint16_t const &multicast_mask,
                                  int32_t const &crd0, int32_t const &crd1,
                                  int32_t const &crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5}], [%2], %6, %7;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0),
        "r"(crd1), "r"(crd2), "h"(multicast_mask), "l"(cache_hint)
      : "memory");
}

// TMA load with multicast using TensorMap descriptor (4D)
template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_multicast(const CUtensorMap &descriptor,
                                  BarrierType &smem_mbar,
                                  void const *const smem_ptr,
                                  uint16_t const &multicast_mask,
                                  int32_t const &crd0, int32_t const &crd1,
                                  int32_t const &crd2, int32_t const &crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6}], [%2], %7, %8;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0),
        "r"(crd1), "r"(crd2), "r"(crd3), "h"(multicast_mask), "l"(cache_hint)
      : "memory");
}

// TMA load with multicast using TensorMap descriptor (5D)
template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load_multicast(const CUtensorMap &descriptor,
                                  BarrierType &smem_mbar,
                                  void const *const smem_ptr,
                                  uint16_t const &multicast_mask,
                                  int32_t const &crd0, int32_t const &crd1,
                                  int32_t const &crd2, int32_t const &crd3,
                                  int32_t const &crd4) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8, %9;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0),
        "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4), "h"(multicast_mask),
        "l"(cache_hint)
      : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3}], [%2], %4;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4}], [%2], %5;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1, int32_t const &crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
               : "memory");
}
template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1, int32_t const &crd2,
                        int32_t const &crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1, int32_t const &crd2,
                        int32_t const &crd3, int32_t const &crd4) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4),
                 "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void
tma_load_im2col(const CUtensorMap &descriptor, BarrierType &smem_mbar,
                void const *const smem_ptr, int32_t const &coord_c,
                int32_t const &coord_w, int32_t const &coord_h,
                int32_t const &coord_n, uint16_t const &offset_w,
                uint16_t const &offset_h) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar =
      smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier:"
               ":complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5, %6}], [%2], {%7, %8}, %9;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_n),
                 "h"(offset_w), "h"(offset_h), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(void *gmem_ptr, void *smem_ptr, uint32_t size) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group"
               ".L2::cache_hint [%0], [%1], %2, %3;"
               :
               : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(size), "l"(cache_hint)
               :);
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.bulk_group "
               ".L2::cache_hint [%0, {%2}], [%1], %3;"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0),
                 "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0,
                         int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
               ".L2::cache_hint [%0, {%2, %3}], [%1], %4;"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                 "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0,
                         int32_t const &crd1, int32_t const &crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
               ".L2::cache_hint [%0, {%2, %3, %4}], [%1], %5;"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                 "r"(crd2), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0,
                         int32_t const &crd1, int32_t const &crd2,
                         int32_t const &crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
               ".L2::cache_hint [%0, {%2, %3, %4, %5}], [%1], %6;"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                 "r"(crd2), "r"(crd3), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0,
                         int32_t const &crd1, int32_t const &crd2,
                         int32_t const &crd3, int32_t const &crd4) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group "
               ".L2::cache_hint [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                 "r"(crd2), "r"(crd3), "r"(crd4), "l"(cache_hint)
               : "memory");
}

TL_DEVICE void tma_store_add(float *const smem_ptr, float *gmem_ptr,
                             int32_t const &store_bytes) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
               "[%0], [%1], %2;\n"
               :
               : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes)
               : "memory");
}

TL_DEVICE void prefetch_tma_descriptor(const CUtensorMap &descriptor) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_int_desc) : "memory");
}

} // namespace tl
