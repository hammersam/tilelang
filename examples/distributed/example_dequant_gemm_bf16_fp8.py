import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm import DataType
from tvm import tir
import torch
from tilelang.engine.callback import register_cuda_postproc_callback

tilelang.disable_cache()

# @register_cuda_postproc_callback
# def tilelang_callback_cuda_postproc(code, _):
#     code = """
# #include <tl_templates/cuda/cuda_fp8.h>
# #include <tl_templates/cuda/gemm.h>
# #include <tl_templates/cuda/copy.h>
# #include <tl_templates/cuda/reduce.h>
# #include <tl_templates/cuda/ldsm.h>
# #include <tl_templates/cuda/threadblock_swizzle.h>
# #include <tl_templates/cuda/debug.h>
# #include <tl_templates/cuda/distributed.h>
# #include <tl_templates/cuda/sync.h>
# #ifdef ENABLE_BF16
# #include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
# #endif
# uint64_t __constant__ meta_data[1024];
# #ifdef ENABLE_BF16
# #include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
# #endif

# extern "C" __global__ void main_cluster_kernel(__grid_constant__ const CUtensorMap A_desc, fp8_e4_t* __restrict__ B, __grid_constant__ const CUtensorMap C_desc);
# extern "C" __global__ void __launch_bounds__(384, 1) main_cluster_kernel(__grid_constant__ const CUtensorMap A_desc, fp8_e4_t* __restrict__ B, __grid_constant__ const CUtensorMap C_desc) {
#   extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
#   __shared__ uint64_t bar_k_avail_1_mem[1];
#   auto bar_k_avail_1 = reinterpret_cast<Barrier*>(bar_k_avail_1_mem);
#   __shared__ uint64_t bar_k_avail_2_mem[1];
#   auto bar_k_avail_2 = reinterpret_cast<Barrier*>(bar_k_avail_2_mem);
#   __shared__ uint64_t compute_is_done_2_mem[1];
#   auto compute_is_done_2 = reinterpret_cast<Barrier*>(compute_is_done_2_mem);
#   __shared__ uint64_t bar_k_remote_ready_1_mem[1];
#   auto bar_k_remote_ready_1 = reinterpret_cast<Barrier*>(bar_k_remote_ready_1_mem);
#   __shared__ uint64_t compute_is_done_1_mem[1];
#   auto compute_is_done_1 = reinterpret_cast<Barrier*>(compute_is_done_1_mem);
#   __shared__ uint64_t data_is_ready_2_mem[1];
#   auto data_is_ready_2 = reinterpret_cast<Barrier*>(data_is_ready_2_mem);
#   __shared__ uint64_t data_is_ready_1_mem[1];
#   auto data_is_ready_1 = reinterpret_cast<Barrier*>(data_is_ready_1_mem);
#   __shared__ uint64_t data_is_ready_0_mem[1];
#   auto data_is_ready_0 = reinterpret_cast<Barrier*>(data_is_ready_0_mem);
#   __shared__ uint64_t bar_k_remote_ready_2_mem[1];
#   auto bar_k_remote_ready_2 = reinterpret_cast<Barrier*>(bar_k_remote_ready_2_mem);
#   __shared__ uint64_t compute_is_done_0_mem[1];
#   auto compute_is_done_0 = reinterpret_cast<Barrier*>(compute_is_done_0_mem);
#   __shared__ uint64_t bar_k_remote_ready_0_mem[1];
#   auto bar_k_remote_ready_0 = reinterpret_cast<Barrier*>(bar_k_remote_ready_0_mem);
#   __shared__ uint64_t bar_k_avail_0_mem[1];
#   auto bar_k_avail_0 = reinterpret_cast<Barrier*>(bar_k_avail_0_mem);
#   float C_local[128];
#   fp8_e4_t cur_fp8x16[16];
#   float cur_fp32x8_lo[8];
#   bfloat16_t cur_bf16x8_lo[8];
#   float cur_fp32x8_hi[8];
#   bfloat16_t cur_bf16x8_hi[8];
#   if (tl::tl_shuffle_elect<0>()) {
#     tl::prefetch_tma_descriptor(A_desc);
#     tl::prefetch_tma_descriptor(C_desc);
#   }
#   if (tl::tl_shuffle_elect<0>()) {
#     bar_k_avail_1[0].init(4);
#     bar_k_avail_2[0].init(4);
#     compute_is_done_2[0].init(256);
#     bar_k_remote_ready_1[0].init(1);
#     compute_is_done_1[0].init(256);
#     data_is_ready_2[0].init(128);
#     data_is_ready_1[0].init(128);
#     data_is_ready_0[0].init(128);
#     bar_k_remote_ready_2[0].init(1);
#     compute_is_done_0[0].init(256);
#     bar_k_remote_ready_0[0].init(1);
#     bar_k_avail_0[0].init(4);
#   }
#   __syncthreads();
#   if (((int)threadIdx.x) < 256) {
#     #pragma unroll
#     for (int i = 0; i < 64; ++i) {
#       *(float2*)(C_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
#     }
#   }
#   if (256 <= ((int)threadIdx.x)) {
#     tl::warpgroup_reg_dealloc<152>();
#     for (int ko = 0; ko < 64; ++ko) {
#       bar_k_avail_0[0].wait(((ko + 1) & 1));
#       if ((((int)threadIdx.x) % 128) == 0) {
#         bar_k_remote_ready_0[0].arrive_and_expect_tx(8192);
#       }
#       compute_is_done_0[0].wait(((ko + 1) & 1));
#       if (((int)threadIdx.x) == 256) {
#         data_is_ready_0[0].expect_transaction(32768);
#         tl::fence_proxy_async();
#         tl::tma_load(A_desc, data_is_ready_0[0], (&(((bfloat16_t*)buf_dyn_shmem)[32768])), (ko * 192), (((int)blockIdx.x) * 256));
#       }
#       for (int ki = 0; ki < 2; ++ki) {
#         tl::load_128b_from_gmem((&(cur_fp8x16[0])), (&(B[((((((ko * 1572864) + ((((int)blockIdx.x) & 1) * 262144)) + (ki * 131072)) + (((((int)threadIdx.x) & 127) >> 3) * 8192)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 7) * 16))])));
#         for (int i_1 = 0; i_1 < 2; ++i_1) {
#           float4 __1;
#           fp8_e4_4_t v_ = *(fp8_e4_4_t*)(cur_fp8x16 + (i_1 * 4));
#           __1.x = (float)(v_.x);
#           __1.y = (float)(v_.y);
#           __1.z = (float)(v_.z);
#           __1.w = (float)(v_.w);
#           *(float4*)(cur_fp32x8_lo + (i_1 * 4)) = __1;
#         }
#         for (int i_2 = 0; i_2 < 2; ++i_2) {
#           uint2 __2;
#           float4 v__1 = *(float4*)(cur_fp32x8_lo + (i_2 * 4));
#           (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(*(float2*)(&(v__1)));
#           (reinterpret_cast<__nv_bfloat162*>(&__2))[1] = __float22bfloat162_rn(*((float2*)(&(v__1))+1));
#           *(uint2*)(cur_bf16x8_lo + (i_2 * 4)) = __2;
#         }
#         // tl::__sync_thread_partial<3, 128>();
#         tl::st_async_128b(tl::get_peer_addr((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) & 7) >> 2) * 4096) + ((((int)blockIdx.x) & 1) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 127) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((int)threadIdx.x) & 15) >> 3) * 8)) + 81920)]))), *reinterpret_cast<float4*>((&(cur_bf16x8_lo[0]))), tl::get_peer_addr((&(bar_k_remote_ready_0[0]))));
#         // tl::__sync_thread_partial<3, 128>();
#         *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) & 7) >> 2) * 4096) + ((((int)blockIdx.x) & 1) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 127) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((int)threadIdx.x) & 15) >> 3) * 8)) + 81920)) = *(uint4*)(cur_bf16x8_lo + 0);
#         for (int i_3 = 0; i_3 < 2; ++i_3) {
#           float4 __3;
#           fp8_e4_4_t v__2 = *(fp8_e4_4_t*)(cur_fp8x16 + ((i_3 * 4) + 8));
#           __3.x = (float)(v__2.x);
#           __3.y = (float)(v__2.y);
#           __3.z = (float)(v__2.z);
#           __3.w = (float)(v__2.w);
#           *(float4*)(cur_fp32x8_hi + (i_3 * 4)) = __3;
#         }
#         for (int i_4 = 0; i_4 < 2; ++i_4) {
#           uint2 __4;
#           float4 v__3 = *(float4*)(cur_fp32x8_hi + (i_4 * 4));
#           (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(*(float2*)(&(v__3)));
#           (reinterpret_cast<__nv_bfloat162*>(&__4))[1] = __float22bfloat162_rn(*((float2*)(&(v__3))+1));
#           *(uint2*)(cur_bf16x8_hi + (i_4 * 4)) = __4;
#         }
#         // tl::__sync_thread_partial<3, 128>();
#         tl::st_async_128b(tl::get_peer_addr((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) & 7) >> 2) * 4096) + ((((int)blockIdx.x) & 1) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 127) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + 1) & 1) * 8)) + 81920)]))), *reinterpret_cast<float4*>((&(cur_bf16x8_hi[0]))), tl::get_peer_addr((&(bar_k_remote_ready_0[0]))));
#         // tl::__sync_thread_partial<3, 128>();
#         *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) & 7) >> 2) * 4096) + ((((int)blockIdx.x) & 1) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 127) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + 1) & 1) * 8)) + 81920)) = *(uint4*)(cur_bf16x8_hi + 0);
#       }
#       data_is_ready_0[0].arrive();
#       bar_k_avail_1[0].wait(((ko + 1) & 1));
#       if ((((int)threadIdx.x) % 128) == 0) {
#         bar_k_remote_ready_1[0].arrive_and_expect_tx(8192);
#       }
#       compute_is_done_1[0].wait(((ko + 1) & 1));
#       if (((int)threadIdx.x) == 256) {
#         data_is_ready_1[0].expect_transaction(32768);
#         tl::fence_proxy_async();
#         tl::tma_load(A_desc, data_is_ready_1[0], (&(((bfloat16_t*)buf_dyn_shmem)[49152])), ((ko * 192) + 64), (((int)blockIdx.x) * 256));
#       }
#       for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
#         tl::load_128b_from_gmem((&(cur_fp8x16[0])), (&(B[(((((((ko * 1572864) + ((((int)blockIdx.x) & 1) * 262144)) + (ki_1 * 131072)) + (((((int)threadIdx.x) & 127) >> 3) * 8192)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + 524288)])));
#         for (int i_5 = 0; i_5 < 2; ++i_5) {
#           float4 __5;
#           fp8_e4_4_t v__4 = *(fp8_e4_4_t*)(cur_fp8x16 + (i_5 * 4));
#           __5.x = (float)(v__4.x);
#           __5.y = (float)(v__4.y);
#           __5.z = (float)(v__4.z);
#           __5.w = (float)(v__4.w);
#           *(float4*)(cur_fp32x8_lo + (i_5 * 4)) = __5;
#         }
#         for (int i_6 = 0; i_6 < 2; ++i_6) {
#           uint2 __6;
#           float4 v__5 = *(float4*)(cur_fp32x8_lo + (i_6 * 4));
#           (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(*(float2*)(&(v__5)));
#           (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(*((float2*)(&(v__5))+1));
#           *(uint2*)(cur_bf16x8_lo + (i_6 * 4)) = __6;
#         }
#         // tl::__sync_thread_partial<3, 128>();
#         tl::st_async_128b(tl::get_peer_addr((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) & 7) >> 2) * 4096) + ((((int)blockIdx.x) & 1) * 2048)) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 127) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((int)threadIdx.x) & 15) >> 3) * 8)) + 90112)]))), *reinterpret_cast<float4*>((&(cur_bf16x8_lo[0]))), tl::get_peer_addr((&(bar_k_remote_ready_1[0]))));
#         // tl::__sync_thread_partial<3, 128>();
#         *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) & 7) >> 2) * 4096) + ((((int)blockIdx.x) & 1) * 2048)) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 127) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((int)threadIdx.x) & 15) >> 3) * 8)) + 90112)) = *(uint4*)(cur_bf16x8_lo + 0);
#         for (int i_7 = 0; i_7 < 2; ++i_7) {
#           float4 __7;
#           fp8_e4_4_t v__6 = *(fp8_e4_4_t*)(cur_fp8x16 + ((i_7 * 4) + 8));
#           __7.x = (float)(v__6.x);
#           __7.y = (float)(v__6.y);
#           __7.z = (float)(v__6.z);
#           __7.w = (float)(v__6.w);
#           *(float4*)(cur_fp32x8_hi + (i_7 * 4)) = __7;
#         }
#         for (int i_8 = 0; i_8 < 2; ++i_8) {
#           uint2 __8;
#           float4 v__7 = *(float4*)(cur_fp32x8_hi + (i_8 * 4));
#           (reinterpret_cast<__nv_bfloat162*>(&__8))[0] = __float22bfloat162_rn(*(float2*)(&(v__7)));
#           (reinterpret_cast<__nv_bfloat162*>(&__8))[1] = __float22bfloat162_rn(*((float2*)(&(v__7))+1));
#           *(uint2*)(cur_bf16x8_hi + (i_8 * 4)) = __8;
#         }
#         // tl::__sync_thread_partial<3, 128>();
#         tl::st_async_128b(tl::get_peer_addr((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) & 7) >> 2) * 4096) + ((((int)blockIdx.x) & 1) * 2048)) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 127) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + 1) & 1) * 8)) + 90112)]))), *reinterpret_cast<float4*>((&(cur_bf16x8_hi[0]))), tl::get_peer_addr((&(bar_k_remote_ready_1[0]))));
#         // tl::__sync_thread_partial<3, 128>();
#         *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) & 7) >> 2) * 4096) + ((((int)blockIdx.x) & 1) * 2048)) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 127) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + 1) & 1) * 8)) + 90112)) = *(uint4*)(cur_bf16x8_hi + 0);
#       }
#       data_is_ready_1[0].arrive();
#       bar_k_avail_2[0].wait(((ko + 1) & 1));
#       if ((((int)threadIdx.x) % 128) == 0) {
#         bar_k_remote_ready_2[0].arrive_and_expect_tx(8192);
#       }
#       compute_is_done_2[0].wait(((ko + 1) & 1));
#       if (((int)threadIdx.x) == 256) {
#         data_is_ready_2[0].expect_transaction(32768);
#         tl::fence_proxy_async();
#         tl::tma_load(A_desc, data_is_ready_2[0], (&(((bfloat16_t*)buf_dyn_shmem)[65536])), ((ko * 192) + 128), (((int)blockIdx.x) * 256));
#       }
#       for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
#         tl::load_128b_from_gmem((&(cur_fp8x16[0])), (&(B[(((((((ko * 1572864) + ((((int)blockIdx.x) & 1) * 262144)) + (ki_2 * 131072)) + (((((int)threadIdx.x) & 127) >> 3) * 8192)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + 1048576)])));
#         for (int i_9 = 0; i_9 < 2; ++i_9) {
#           float4 __9;
#           fp8_e4_4_t v__8 = *(fp8_e4_4_t*)(cur_fp8x16 + (i_9 * 4));
#           __9.x = (float)(v__8.x);
#           __9.y = (float)(v__8.y);
#           __9.z = (float)(v__8.z);
#           __9.w = (float)(v__8.w);
#           *(float4*)(cur_fp32x8_lo + (i_9 * 4)) = __9;
#         }
#         for (int i_10 = 0; i_10 < 2; ++i_10) {
#           uint2 __10;
#           float4 v__9 = *(float4*)(cur_fp32x8_lo + (i_10 * 4));
#           (reinterpret_cast<__nv_bfloat162*>(&__10))[0] = __float22bfloat162_rn(*(float2*)(&(v__9)));
#           (reinterpret_cast<__nv_bfloat162*>(&__10))[1] = __float22bfloat162_rn(*((float2*)(&(v__9))+1));
#           *(uint2*)(cur_bf16x8_lo + (i_10 * 4)) = __10;
#         }
#         // tl::__sync_thread_partial<3, 128>();
#         tl::st_async_128b(tl::get_peer_addr((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) & 7) >> 2) * 4096) + ((((int)blockIdx.x) & 1) * 2048)) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 127) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((int)threadIdx.x) & 15) >> 3) * 8)) + 98304)]))), *reinterpret_cast<float4*>((&(cur_bf16x8_lo[0]))), tl::get_peer_addr((&(bar_k_remote_ready_2[0]))));
#         // tl::__sync_thread_partial<3, 128>();
#         *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) & 7) >> 2) * 4096) + ((((int)blockIdx.x) & 1) * 2048)) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 127) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((int)threadIdx.x) & 15) >> 3) * 8)) + 98304)) = *(uint4*)(cur_bf16x8_lo + 0);
#         for (int i_11 = 0; i_11 < 2; ++i_11) {
#           float4 __11;
#           fp8_e4_4_t v__10 = *(fp8_e4_4_t*)(cur_fp8x16 + ((i_11 * 4) + 8));
#           __11.x = (float)(v__10.x);
#           __11.y = (float)(v__10.y);
#           __11.z = (float)(v__10.z);
#           __11.w = (float)(v__10.w);
#           *(float4*)(cur_fp32x8_hi + (i_11 * 4)) = __11;
#         }
#         for (int i_12 = 0; i_12 < 2; ++i_12) {
#           uint2 __12;
#           float4 v__11 = *(float4*)(cur_fp32x8_hi + (i_12 * 4));
#           (reinterpret_cast<__nv_bfloat162*>(&__12))[0] = __float22bfloat162_rn(*(float2*)(&(v__11)));
#           (reinterpret_cast<__nv_bfloat162*>(&__12))[1] = __float22bfloat162_rn(*((float2*)(&(v__11))+1));
#           *(uint2*)(cur_bf16x8_hi + (i_12 * 4)) = __12;
#         }
#         // tl::__sync_thread_partial<3, 128>();
#         tl::st_async_128b(tl::get_peer_addr((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) & 7) >> 2) * 4096) + ((((int)blockIdx.x) & 1) * 2048)) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 127) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + 1) & 1) * 8)) + 98304)]))), *reinterpret_cast<float4*>((&(cur_bf16x8_hi[0]))), tl::get_peer_addr((&(bar_k_remote_ready_2[0]))));
#         // tl::__sync_thread_partial<3, 128>();
#         *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) & 7) >> 2) * 4096) + ((((int)blockIdx.x) & 1) * 2048)) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 127) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + 1) & 1) * 8)) + 98304)) = *(uint4*)(cur_bf16x8_hi + 0);
#       }
#       data_is_ready_2[0].arrive();
#     }
#   } else {
#     tl::warpgroup_reg_alloc<160>();
#     for (int ko_1 = 0; ko_1 < 64; ++ko_1) {
#       data_is_ready_0[0].wait((ko_1 & 1));
#       bar_k_remote_ready_0[0].wait((ko_1 & 1));
#       tl::fence_proxy_async();
#       tl::gemm_ss<256, 128, 64, 4, 2, 0, 0, 0, 64, 128, 0, 0, true>((&(((bfloat16_t*)buf_dyn_shmem)[32768])), (&(((bfloat16_t*)buf_dyn_shmem)[81920])), (&(C_local[0])));
#       compute_is_done_0[0].arrive();
#       bar_k_avail_0[0].arrive(uint32_t(0), ((((int)threadIdx.x) & 127) == 32));
#       bar_k_avail_0[0].arrive(uint32_t(1), ((((int)threadIdx.x) & 127) == 64));
#       data_is_ready_1[0].wait((ko_1 & 1));
#       bar_k_remote_ready_1[0].wait((ko_1 & 1));
#       tl::fence_proxy_async();
#       tl::gemm_ss<256, 128, 64, 4, 2, 0, 0, 0, 64, 128, 0, 0, true>((&(((bfloat16_t*)buf_dyn_shmem)[49152])), (&(((bfloat16_t*)buf_dyn_shmem)[90112])), (&(C_local[0])));
#       compute_is_done_1[0].arrive();
#       bar_k_avail_1[0].arrive(uint32_t(0), ((((int)threadIdx.x) & 127) == 32));
#       bar_k_avail_1[0].arrive(uint32_t(1), ((((int)threadIdx.x) & 127) == 64));
#       data_is_ready_2[0].wait((ko_1 & 1));
#       bar_k_remote_ready_2[0].wait((ko_1 & 1));
#       tl::fence_proxy_async();
#       tl::gemm_ss<256, 128, 64, 4, 2, 0, 0, 0, 64, 128, 0, 0, true>((&(((bfloat16_t*)buf_dyn_shmem)[65536])), (&(((bfloat16_t*)buf_dyn_shmem)[98304])), (&(C_local[0])));
#       compute_is_done_2[0].arrive();
#       bar_k_avail_2[0].arrive(uint32_t(0), ((((int)threadIdx.x) & 127) == 32));
#       bar_k_avail_2[0].arrive(uint32_t(1), ((((int)threadIdx.x) & 127) == 64));
#     }
#     tl::__sync_thread_partial<4, 256>();
#     #pragma unroll
#     for (int i_13 = 0; i_13 < 16; ++i_13) {
#       tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) >> 7) * 16384) + ((i_13 >> 2) * 4096)) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)C_local[(i_13 * 8)]), ((bfloat16_t)C_local[((i_13 * 8) + 1)])), __pack_half2(((bfloat16_t)C_local[((i_13 * 8) + 2)]), ((bfloat16_t)C_local[((i_13 * 8) + 3)])), __pack_half2(((bfloat16_t)C_local[((i_13 * 8) + 4)]), ((bfloat16_t)C_local[((i_13 * 8) + 5)])), __pack_half2(((bfloat16_t)C_local[((i_13 * 8) + 6)]), ((bfloat16_t)C_local[((i_13 * 8) + 7)])));
#     }
#     tl::fence_proxy_async();
#     tl::__sync_thread_partial<4, 256>();
#     if (((int)threadIdx.x) == 0) {
#       tl::tma_store(C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), (((int)blockIdx.y) * 128), (((int)blockIdx.x) * 256));
#       tl::tma_store_arrive();
#       tl::tma_store_wait<0>();
#       tl::tma_store(C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[16384])), ((((int)blockIdx.y) * 128) + 64), (((int)blockIdx.x) * 256));
#       tl::tma_store_arrive();
#       tl::tma_store_wait<0>();
#     }
#   }
#   tl::sync_cluster();
# }
#     """
#     return code
    
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}
@tilelang.jit(
    out_idx=[-1],
    pass_configs=pass_configs,
)
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
    
    @T.macro
    def dequant_and_st(
            cur_fp32x8: T.LocalBuffer([8,], "float32"),
            cur_bf16x8: T.LocalBuffer([8,], "bfloat16"),
            cur_fp8x16: T.LocalBuffer((16,), "float8_e4m3"),
            B_shared: T.SharedBuffer((block_K, block_N), "bfloat16"),
            bar_k_remote_ready: T.SharedBuffer((1), "uint64"),
            offset_k: T.int32,
            thread_offset: T.int32,
            lane_idx: T.int32,
            cx: T.int32,
    ):
        for i in T.vectorized(8):
            cur_fp32x8[i] = T.cast(cur_fp8x16[i + thread_offset], "float")
        for i in T.vectorized(8):
            cur_bf16x8[i] = T.cast(cur_fp32x8[i], "bfloat16")
        if cluster_size > 1:
            T.put_thread(
                src=T.address_of(cur_bf16x8[0]),
                dst=T.address_of(B_shared[offset_k, lane_idx % 8 * 16 + thread_offset]),
                size=0,
                mbar=T.address_of(bar_k_remote_ready),
                dst_pe=(cx + 1) % cluster_size,
                scope="cluster")
        for i in T.vectorized(8):
            # FIXME: check shared memory swizzle
            B_shared[offset_k, lane_idx % 8 * 16 + i + thread_offset] = cur_bf16x8[i]
        
    @T.macro
    def load(
        ko,
        num_stages,
        buffer_id,
        A,
        B,
        bar_k_avail,
        cur_fp32x8_lo,
        cur_fp32x8_hi,
        cur_bf16x8_lo,
        cur_bf16x8_hi,
        cur_fp8x16,
        A_shared,
        B_shared,
        data_is_ready,
        compute_is_done,
        bar_k_remote_ready,
        bx, by,
        cx,
        idx_in_warpgroup,
        warp_idx,
        lane_idx,
    ):
        if cluster_size > 1:
            T.barrier_wait(bar_k_avail, (ko + 1) % 2)
            if idx_in_warpgroup == 0:
                T.ptx_arrive_barrier_expect_tx(
                    bar_k_remote_ready[0], 
                    block_K * block_N // cluster_size * 2)
        
        T.barrier_wait(compute_is_done, (ko + 1) % 2)
        T.copy(A[bx * block_M, (num_stages * ko + buffer_id) * block_K], A_shared)
        for ki in T.serial(block_K // cluster_size // 16):
            offset_k = block_K // cluster_size * cx + ki * 16 + warp_idx * 4 + lane_idx // 8
            T.load_128b_from_gmem(B[(num_stages * ko + buffer_id) * block_K + offset_k, by * block_N + lane_idx % 8 * 16], cur_fp8x16)
            dequant_and_st(
                cur_fp32x8_lo,
                cur_bf16x8_lo,
                cur_fp8x16,
                B_shared,
                bar_k_remote_ready,
                offset_k,
                0,
                lane_idx,
                cx,)
            dequant_and_st(
                cur_fp32x8_hi,
                cur_bf16x8_hi,
                cur_fp8x16,
                B_shared,
                bar_k_remote_ready,
                offset_k,
                8,
                lane_idx,
                cx,)
        T.barrier_arrive(data_is_ready)
        
    @T.macro
    def compute(
        ko,
        A_shared,
        B_shared,
        C_local,
        data_is_ready,
        bar_k_remote_ready,
        compute_is_done,
        bar_k_avail,
        idx_in_warpgroup,
    ):
        T.barrier_wait(data_is_ready, ko % 2)
        if cluster_size > 1:
            T.barrier_wait(bar_k_remote_ready, ko % 2)
        T.gemm(A_shared, B_shared, C_local)
        T.barrier_arrive(compute_is_done)
        # TODO: better to provide a cleaner API
        if cluster_size > 1:
            T.barrier_arrive(bar_k_avail, 0, idx_in_warpgroup == 32)
            T.barrier_arrive(bar_k_avail, 1, idx_in_warpgroup == 64)
        
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
            
            A_shared_0 = T.alloc_shared((block_M, block_K), A_dtype)
            A_shared_1 = T.alloc_shared((block_M, block_K), A_dtype)
            A_shared_2 = T.alloc_shared((block_M, block_K), A_dtype)
            B_shared_0 = T.alloc_shared((block_K, block_N), "bfloat16")
            B_shared_1 = T.alloc_shared((block_K, block_N), "bfloat16")
            B_shared_2 = T.alloc_shared((block_K, block_N), "bfloat16")
            C_shared = T.alloc_shared((block_M, block_N), C_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            cur_fp8x16 = T.alloc_local((16,), "float8_e4m3")
            cur_fp32x8_lo = T.alloc_local([8], "float32")
            cur_fp32x8_hi = T.alloc_local([8], "float32")
            cur_bf16x8_lo = T.alloc_local([8], "bfloat16")
            cur_bf16x8_hi = T.alloc_local([8], "bfloat16")
            data_is_ready_0 = T.alloc_barrier(arrive_count=128)
            data_is_ready_1 = T.alloc_barrier(arrive_count=128)
            data_is_ready_2 = T.alloc_barrier(arrive_count=128)
            compute_is_done_0 = T.alloc_barrier(arrive_count=256)
            compute_is_done_1 = T.alloc_barrier(arrive_count=256)
            compute_is_done_2 = T.alloc_barrier(arrive_count=256)
            bar_k_remote_ready_0 = T.alloc_barrier(arrive_count=1)
            bar_k_remote_ready_1 = T.alloc_barrier(arrive_count=1)
            bar_k_remote_ready_2 = T.alloc_barrier(arrive_count=1)
            bar_k_avail_0 = T.alloc_barrier(arrive_count=4)
            bar_k_avail_1 = T.alloc_barrier(arrive_count=4)
            bar_k_avail_2 = T.alloc_barrier(arrive_count=4)

            # T.use_swizzle(10)
            T.annotate_layout({
                C_shared: tilelang.layout.make_swizzled_layout(C_shared),
            })
            
            if tx < 256:
                T.clear(C_local)
            
            if tx >= 256:
                T.set_max_nreg(152, 0)
                for ko in T.serial(T.ceildiv(K, block_K) // num_stages):
                    load(
                        ko,
                        num_stages,
                        0,
                        A,
                        B,
                        bar_k_avail_0,
                        cur_fp32x8_lo,
                        cur_fp32x8_hi,
                        cur_bf16x8_lo,
                        cur_bf16x8_hi,
                        cur_fp8x16,
                        A_shared_0,
                        B_shared_0,
                        data_is_ready_0,
                        compute_is_done_0,
                        bar_k_remote_ready_0,
                        bx, by,
                        cx,
                        idx_in_warpgroup,
                        warp_idx,
                        lane_idx,)
                    
                    load(
                        ko,
                        num_stages,
                        1,
                        A,
                        B,
                        bar_k_avail_1,
                        cur_fp32x8_lo,
                        cur_fp32x8_hi,
                        cur_bf16x8_lo,
                        cur_bf16x8_hi,
                        cur_fp8x16,
                        A_shared_1,
                        B_shared_1,
                        data_is_ready_1,
                        compute_is_done_1,
                        bar_k_remote_ready_1,
                        bx, by,
                        cx,
                        idx_in_warpgroup,
                        warp_idx,
                        lane_idx,)
                    
                    load(
                        ko,
                        num_stages,
                        2,
                        A,
                        B,
                        bar_k_avail_2,
                        cur_fp32x8_lo,
                        cur_fp32x8_hi,
                        cur_bf16x8_lo,
                        cur_bf16x8_hi,
                        cur_fp8x16,
                        A_shared_2,
                        B_shared_2,
                        data_is_ready_2,
                        compute_is_done_2,
                        bar_k_remote_ready_2,
                        bx, by,
                        cx,
                        idx_in_warpgroup,
                        warp_idx,
                        lane_idx,)
            else:
                T.set_max_nreg(160, 1)
                for ko in T.serial(T.ceildiv(K, block_K) // num_stages):
                    compute(
                        ko,
                        A_shared_0,
                        B_shared_0,
                        C_local,
                        data_is_ready_0,
                        bar_k_remote_ready_0,
                        compute_is_done_0,
                        bar_k_avail_0,
                        idx_in_warpgroup,)
                    
                    compute(
                        ko,
                        A_shared_1,
                        B_shared_1,
                        C_local,
                        data_is_ready_1,
                        bar_k_remote_ready_1,
                        compute_is_done_1,
                        bar_k_avail_1,
                        idx_in_warpgroup,)
                    
                    compute(
                        ko,
                        A_shared_2,
                        B_shared_2,
                        C_local,
                        data_is_ready_2,
                        bar_k_remote_ready_2,
                        compute_is_done_2,
                        bar_k_avail_2,
                        idx_in_warpgroup,)
                    
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
        num_stages=3,
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
    M, N, K = 8192, 8192, 12288
    scale_size = 32
    main(M, N, K, scale_size, use_naive=False, use_cluster=True)
