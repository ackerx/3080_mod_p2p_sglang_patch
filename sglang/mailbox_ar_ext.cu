// mailbox_ar_ext.cu — sglang 信箱 push allreduce 的 torch 扩展
// 协议 = mailbox_graph.cu 已验证形态: 单调 round(GPU 自增) + 单调 flag/ack +
// last-block flag 写 + ping-pong 半区(规避跨轮 L2 stale)。kernel 参数全指针,
// cuda graph capture/重放兼容(mailbox_graph 500 次重放 PASS)。
// 生产差异: 数据=业务输入(无校验值生成), 支持 fp32/half/bf16, 超时 1s。
// 信箱布局: [flag int][ack int][pad 2int][data 2×24MB]; 状态区 int[8]:
//   [0]=eflag [2]=eack [4]=round [5]=err [7]=done
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <string>
#include <cstring>

#define MBX_HALF_BYTES (24ull << 20)
static const unsigned long long TIMEOUT_CLK = 1ULL * 3000000000ULL;

#define ST_EFLAG 0
#define ST_EACK 2
#define ST_ROUND 4
#define ST_ERR 5
#define ST_DONE 7

__global__ void mb_waitack_k(volatile int* remote_ack, int* st, int* err) {
  if (*err) return;
  int e = st[ST_EACK];
  unsigned long long t0 = clock64();
  while (*remote_ack != e) {
    if (clock64() - t0 > TIMEOUT_CLK) { atomicExch(err, 2); return; }
    __nanosleep(64);
  }
  st[ST_EACK] = e + 1;
}

__global__ void mb_wait_local_k(volatile int* my_flag, int* st, int* err) {
  if (*err) return;
  int e = st[ST_EFLAG];
  unsigned long long t0 = clock64();
  while (*my_flag != e) {
    if (clock64() - t0 > TIMEOUT_CLK) { atomicExch(err, 3); return; }
    __nanosleep(64);
  }
  st[ST_EFLAG] = e + 1;
}

template <typename T>
__global__ void mb_produce_push_k(T* dst_data, const T* local_in,
                                  size_t n, int* st) {
  if (st[ST_ERR]) return;
  int r = st[ST_ROUND];
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t half_elems = MBX_HALF_BYTES / sizeof(T);
  size_t half_off = ((size_t)(r & 1)) * half_elems;
  // 纯 store(大消息路径): 不做任何 fence/原子。数据可达性由同流的 mb_flag_k
  // 保证 —— 内核级流序保证 flag 内核在 produce 全部 block 退役后才发 flag, 而
  // 数据与 flag 同为本 GPU 发出的 PCIe posted 写、同 VC 按序投递, 对端观察到
  // flag==r 即数据已全部投递。(原 threadfence_system 的 PCIe 排空在双向并发
  // 下把 produce 压到 5.9 GB/s, 实测数据。)
  for (; i < n; i += stride) dst_data[half_off + i] = local_in[i];
}

// 小消息路径(≤4MB): fence(device 域)+块内 DONE 原子+末块写 flag, 自包含单内核,
// 省一次 1 线程内核发射(小消息时 ~4μs 占比可观)。
template <typename T>
__global__ void mb_produce_fb_k(T* dst_data, int* dst_flag, const T* local_in,
                                size_t n, int* st) {
  if (st[ST_ERR]) return;
  int r = st[ST_ROUND];
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t half_elems = MBX_HALF_BYTES / sizeof(T);
  size_t half_off = ((size_t)(r & 1)) * half_elems;
  for (; i < n; i += stride) dst_data[half_off + i] = local_in[i];
  __threadfence();          // device 域即可: 发布顺序保证 DONE 原子后于 store
  __syncthreads();
  if (threadIdx.x == 0) {
    if (atomicAdd(st + ST_DONE, 1) == (int)gridDim.x - 1) {
      atomicExch(st + ST_DONE, 0);
      atomicExch(dst_flag, r);   // posted 写与数据同 VC 按序投递
    }
  }
}

// flag 独立内核(大消息路径): 1 线程, 在 produce 之后同流发射(流序 = 数据全部退役后才执行)
__global__ void mb_flag_k(int* dst_flag, int* st) {
  atomicExch(dst_flag, st[ST_ROUND]);
}

template <typename T>
__device__ __forceinline__ T mb_add(T a, T b);
template <> __device__ __forceinline__ float mb_add<float>(float a, float b) { return a + b; }
template <> __device__ __forceinline__ __half mb_add<__half>(__half a, __half b) {
  return __float2half(__half2float(a) + __half2float(b));
}
template <> __device__ __forceinline__ __nv_bfloat16 mb_add<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

template <typename T>
__global__ void mb_reduce_k(const T* local_in, const T* mbx_data, T* out,
                            size_t n, int* st) {
  if (st[ST_ERR]) return;
  const size_t half_elems = MBX_HALF_BYTES / sizeof(T);
  size_t half_off = ((size_t)(st[ST_ROUND] & 1)) * half_elems;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) out[i] = mb_add<T>(local_in[i], mbx_data[half_off + i]);
}

__global__ void mb_ack_k(int* my_ack, int* st) {
  atomicExch(my_ack, st[ST_ROUND]);
  atomicAdd(st + ST_ROUND, 1);
}

static unsigned grid_for(size_t n) {
  size_t g = (n + 255) / 256;
  if (g < 1) g = 1; if (g > 65535) g = 65535;
  return (unsigned)g;
}

// dtype_id: 0=fp32, 1=fp16, 2=bf16
void mailbox_allreduce(int64_t inp_p, int64_t out_p, int64_t numel, int dtype_id,
                       int64_t my_mbx_p, int64_t peer_mbx_p, int64_t st_p) {
  int* mbx_flag = (int*)my_mbx_p;
  int* mbx_ack = mbx_flag + 1;
  char* mbx_base = (char*)my_mbx_p;
  // 数据区必须 128B+ 对齐: 原先 +16B 让每条 64B warp store 跨 3 个 L2 扇区,
  // PCIe push 掉 40% (10.26→6.2 GB/s, push_bw 微基准钉死) —— 头部让足 256B
  float* mbx_f = (float*)(mbx_base + 256);
  int* peer_flag = (int*)peer_mbx_p;
  volatile int* peer_ack = (int*)peer_mbx_p + 1;
  char* peer_base = (char*)peer_mbx_p;
  float* peer_f = (float*)(peer_base + 256);
  int* st = (int*)st_p;
  size_t n = (size_t)numel;
  unsigned grid = grid_for(n);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  mb_waitack_k<<<1, 1, 0, stream>>>(peer_ack, st, st + ST_ERR);
  const bool small = n * (dtype_id == 0 ? 4u : 2u) <= (4u << 20);
  if (small) {
    if (dtype_id == 0)
      mb_produce_fb_k<float><<<grid, 256, 0, stream>>>(peer_f, peer_flag, (float*)inp_p, n, st);
    else if (dtype_id == 1)
      mb_produce_fb_k<__half><<<grid, 256, 0, stream>>>((__half*)peer_f, peer_flag, (__half*)inp_p, n, st);
    else
      mb_produce_fb_k<__nv_bfloat16><<<grid, 256, 0, stream>>>((__nv_bfloat16*)peer_f, peer_flag, (__nv_bfloat16*)inp_p, n, st);
  } else {
    if (dtype_id == 0)
      mb_produce_push_k<float><<<grid, 256, 0, stream>>>(peer_f, (float*)inp_p, n, st);
    else if (dtype_id == 1)
      mb_produce_push_k<__half><<<grid, 256, 0, stream>>>((__half*)peer_f, (__half*)inp_p, n, st);
    else
      mb_produce_push_k<__nv_bfloat16><<<grid, 256, 0, stream>>>((__nv_bfloat16*)peer_f, (__nv_bfloat16*)inp_p, n, st);
    mb_flag_k<<<1, 1, 0, stream>>>(peer_flag, st);
  }
  mb_wait_local_k<<<1, 1, 0, stream>>>(mbx_flag, st, st + ST_ERR);
  if (dtype_id == 0)
    mb_reduce_k<float><<<grid, 256, 0, stream>>>((float*)inp_p, mbx_f, (float*)out_p, n, st);
  else if (dtype_id == 1)
    mb_reduce_k<__half><<<grid, 256, 0, stream>>>((__half*)inp_p, (__half*)mbx_f, (__half*)out_p, n, st);
  else
    mb_reduce_k<__nv_bfloat16><<<grid, 256, 0, stream>>>((__nv_bfloat16*)inp_p, (__nv_bfloat16*)mbx_f, (__nv_bfloat16*)out_p, n, st);
  mb_ack_k<<<1, 1, 0, stream>>>(mbx_ack, st);
}

py::bytes ipc_get_handle(int64_t mbx_p) {
  cudaIpcMemHandle_t h;
  cudaError_t e = cudaIpcGetMemHandle(&h, (void*)mbx_p);
  TORCH_CHECK(e == cudaSuccess, "cudaIpcGetMemHandle failed: ", cudaGetErrorString(e));
  return py::bytes((const char*)&h, sizeof h);   // 二进制 handle, 必须 bytes(std::string 会被按 utf-8 解码)
}

int64_t ipc_open(const py::bytes& hstr) {
  std::string s = hstr;
  TORCH_CHECK(s.size() == sizeof(cudaIpcMemHandle_t), "bad handle size");
  cudaIpcMemHandle_t h;
  memcpy(&h, s.data(), sizeof h);
  void* p = nullptr;
  cudaError_t e = cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess);
  TORCH_CHECK(e == cudaSuccess, "cudaIpcOpenMemHandle failed: ", cudaGetErrorString(e),
              " (信箱未落窗?)");
  return (int64_t)p;
}

void state_reset(int64_t st_p) {
  int h[8] = {1, 0, 0, 0, 1, 0, 0, 0};   // eflag=1 eack=0 round=1
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync((void*)st_p, h, 32, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
}

int64_t state_get(int64_t st_p, int64_t slot) {
  int v = 0;
  cudaMemcpy(&v, (void*)(st_p + slot * 4), 4, cudaMemcpyDeviceToHost);
  return v;
}

// 裸 push(微基准/调试用): 16B 向量化纯 store 一段字节到任意显存地址,
// 无协议无 flag。grid 封顶 grid_cap(grid-stride, 0=不封顶) —— 供 push×GEMM
// 并行争抢敏感性扫描。bytes 须 16B 对齐(信箱段 10240B/行天然满足)。
__global__ void mb_push_raw_k(char* dst, const char* src, size_t bytes) {
  size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 16;
  size_t stride = (size_t)gridDim.x * blockDim.x * 16;
  for (; i + 16 <= bytes; i += stride)
    *(uint4*)(dst + i) = *(const uint4*)(src + i);
}

void mailbox_push_raw(int64_t dst_p, int64_t src_p, int64_t bytes, int64_t grid_cap,
                      int64_t stream_ptr) {
  TORCH_CHECK(bytes % 16 == 0, "push_raw: bytes must be 16B aligned");
  unsigned g = grid_for((size_t)bytes / 16);
  if (grid_cap > 0 && g > (unsigned)grid_cap) g = (unsigned)grid_cap;
  cudaStream_t stream = (stream_ptr == 0) ? at::cuda::getCurrentCUDAStream()
                                          : (cudaStream_t)stream_ptr;
  mb_push_raw_k<<<g, 256, 0, stream>>>((char*)dst_p, (const char*)src_p, (size_t)bytes);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "push_raw: ", cudaGetErrorString(e));
}

// CE push: cudaMemcpyAsync 走 copy engine(DMA), 不占 SM ——
// 与 SM 上的 GEMM 是硬件级并行, 供 push×GEMM 重叠的第二路线验证。
void mailbox_ce_push(int64_t dst_p, int64_t src_p, int64_t bytes, int64_t stream_ptr) {
  cudaStream_t stream = (stream_ptr == 0) ? at::cuda::getCurrentCUDAStream()
                                          : (cudaStream_t)stream_ptr;
  cudaError_t e = cudaMemcpyAsync((void*)dst_p, (const void*)src_p, (size_t)bytes,
                                  cudaMemcpyDeviceToDevice, stream);
  TORCH_CHECK(e == cudaSuccess, "ce_push: ", cudaGetErrorString(e));
}

// ---- 分段 AR 协议(可选扩展, 默认不激活) ----
// 布局: 信箱尾部 16MB(fb/push 数据区 2×24MB 之外, 对既有路径零扰动;
// fb 满消息数据止于 48MiB+255, 紧贴 flag 区前一字节)。[flag 3int][3×5.25MiB 槽]。
// 段提交在调用方副流: 先 1-block spin 等对端 flag[slot]==0(上轮归约已清, 跨端
// BAR1 读同 fb 的 waitack 先例), 再 CE memcpy 段数据到对端槽, 同流 1-block 内核
// 写对端 flag[slot]=seq —— CE 完成语义 + 同流序保证"数据先于 flag"。
// 段等待在主流: spin 本端 flag[slot]==seq → 全量归约 → 1-block 清 flag[slot]=0
// (放行对端下一次复用)。seq 由 host 维护全局单调(从 1 起)。槽生命周期 =
// 置位(seq) → 归约 → 清 0, 复用依赖清 0 硬同步, 消灭"覆写先于对端归约读"竞态。
// 与 fb 路径(头部 int[0]/eflag/st 全槽)完全独立, 互不干扰。
#define MBX_SLOTS 3
#define MBX_SLOT_BYTES 5505024ull              // 5.25MiB; 512B flag + 3 槽 ≤ 16MB
#define MBX_SEG_FLAG_OFF ((48ull << 20) + 256) // flag[slot] = mbx + 48MiB + 256 + slot*4
#define MBX_SEG_DATA_OFF ((48ull << 20) + 512) // 槽数据 = mbx + 48MiB + 512 + slot*5.25MiB

__global__ void mb_flag_slot_k(char* peer_base, int64_t seq) {
  int slot = (int)(seq % MBX_SLOTS);
  atomicExch((int*)(peer_base + MBX_SEG_FLAG_OFF + slot * 4), (int)seq);
}

// 复用放行(副流, 1-block): 跨端 spin 等对端 flag[slot]==0 —— 对端归约完成并清 0
// 后, 本次覆写才放行(超时置 err=5)
__global__ void mb_wait_slot_free_k(volatile int* peer_flag_slot, int* st) {
  if (st[ST_ERR]) return;
  unsigned long long t0 = clock64();
  while (*peer_flag_slot != 0) {
    if (clock64() - t0 > TIMEOUT_CLK) { atomicExch(st + ST_ERR, 5); return; }
    __nanosleep(64);
  }
}

void mailbox_submit_seg(int64_t inp_p, int64_t bytes, int64_t seq,
                        int64_t peer_mbx_p, int64_t st_p, int64_t stream_ptr) {
  TORCH_CHECK(bytes > 0 && (size_t)bytes + 512 <= MBX_SLOT_BYTES,
              "submit_seg: seg ", bytes, " B exceeds slot ", MBX_SLOT_BYTES - 512);
  cudaStream_t stream = (stream_ptr == 0) ? at::cuda::getCurrentCUDAStream()
                                          : (cudaStream_t)stream_ptr;
  int slot = (int)(seq % MBX_SLOTS);
  volatile int* peer_flag_slot = (volatile int*)(peer_mbx_p + MBX_SEG_FLAG_OFF + slot * 4);
  int* st = (int*)st_p;
  mb_wait_slot_free_k<<<1, 1, 0, stream>>>(peer_flag_slot, st);
  char* dst = (char*)(peer_mbx_p + MBX_SEG_DATA_OFF + slot * MBX_SLOT_BYTES);
  cudaError_t e = cudaMemcpyAsync(dst, (const void*)inp_p, (size_t)bytes,
                                  cudaMemcpyDeviceToDevice, stream);
  TORCH_CHECK(e == cudaSuccess, "submit_seg memcpy: ", cudaGetErrorString(e));
  mb_flag_slot_k<<<1, 1, 0, stream>>>((char*)peer_mbx_p, seq);
  e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "submit_seg flag: ", cudaGetErrorString(e));
}

// 段等待(主流, 1-block spin —— 多 block spin 洪泛 L2 会挤死入站 P2P 写)
__global__ void mb_wait_slot_k(const char* mbx_base, int64_t seq, int* st) {
  if (st[ST_ERR]) return;
  int slot = (int)(seq % MBX_SLOTS);
  volatile const int* f = (const volatile int*)(mbx_base + MBX_SEG_FLAG_OFF + slot * 4);
  unsigned long long t0 = clock64();
  while (*f != (int)seq) {
    if (clock64() - t0 > TIMEOUT_CLK) { atomicExch(st + ST_ERR, 4); return; }
    __nanosleep(64);
  }
}

// 段归约(主流, 全量 grid): 前序 wait_slot 已保证该段数据投递完成(内核边界可见性)。
// out[i] = local[i] + 槽数据[i], 段内元素数 = bytes/esz。
template <typename T>
__global__ void mb_reduce_slot_k(const T* local_in, const char* mbx_base, T* out,
                                 int64_t seq, size_t elems, int* st) {
  if (st[ST_ERR]) return;
  int slot = (int)(seq % MBX_SLOTS);
  const T* seg = (const T*)(mbx_base + MBX_SEG_DATA_OFF + slot * MBX_SLOT_BYTES);
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < elems; i += stride)
    out[i] = mb_add<T>(local_in[i], seg[i]);
}

// 复用放行(主流, 1-block): 归约读完成后清 flag[slot]=0, 对端才可复用该槽。
// 与归约同流, 流序保证归约全部 block 退役后才清。
__global__ void mb_free_slot_k(char* mbx_base, int64_t seq) {
  int slot = (int)(seq % MBX_SLOTS);
  atomicExch((int*)(mbx_base + MBX_SEG_FLAG_OFF + slot * 4), 0);
}

void mailbox_wait_seg(int64_t seq, int64_t local_p, int64_t out_p, int64_t bytes,
                      int64_t dtype_id, int64_t mbx_p, int64_t st_p, int64_t stream_ptr) {
  const size_t esz = (dtype_id == 0) ? 4u : 2u;
  size_t elems = (size_t)bytes / esz;
  cudaStream_t stream = (stream_ptr == 0) ? at::cuda::getCurrentCUDAStream()
                                          : (cudaStream_t)stream_ptr;
  int* st = (int*)st_p;
  mb_wait_slot_k<<<1, 1, 0, stream>>>((const char*)mbx_p, seq, st);
  unsigned grid = grid_for(elems);
  if (dtype_id == 0)
    mb_reduce_slot_k<float><<<grid, 256, 0, stream>>>((const float*)local_p, (const char*)mbx_p, (float*)out_p, seq, elems, st);
  else if (dtype_id == 1)
    mb_reduce_slot_k<__half><<<grid, 256, 0, stream>>>((__half*)local_p, (const char*)mbx_p, (__half*)out_p, seq, elems, st);
  else
    mb_reduce_slot_k<__nv_bfloat16><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)local_p, (const char*)mbx_p, (__nv_bfloat16*)out_p, seq, elems, st);
  mb_free_slot_k<<<1, 1, 0, stream>>>((char*)mbx_p, seq);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "wait_seg: ", cudaGetErrorString(e));
}

// flag 槽清零(构造期调用): flag 初始为 torch.empty 垃圾值, 若恰 == 首个 seq
// 会误通过。清 0 + seq 从 1 起 = 等待条件充分。各端清自己的信箱(对端写本端 flag)。
void mailbox_seg_reset(int64_t mbx_p) {
  cudaMemset((void*)(mbx_p + MBX_SEG_FLAG_OFF), 0, MBX_SLOTS * 4);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "seg_reset: ", cudaGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mailbox_allreduce", &mailbox_allreduce, "mailbox push allreduce (5-kernel lockstep)");
  m.def("ipc_get_handle", &ipc_get_handle, "get IPC handle of mailbox base");
  m.def("ipc_open", &ipc_open, "open peer mailbox handle, return base ptr");
  m.def("state_reset", &state_reset, "reset mailbox state block");
  m.def("state_get", &state_get, "read state slot for diagnostics");
  m.def("mailbox_push_raw", &mailbox_push_raw, "raw push (gate bench/debug)");
  m.def("mailbox_ce_push", &mailbox_ce_push, "copy-engine push (gate bench)");
  m.def("mailbox_submit_seg", &mailbox_submit_seg, "seg submit: CE push to peer slot + flag");
  m.def("mailbox_wait_seg", &mailbox_wait_seg, "seg wait: spin flag + reduce one segment");
  m.def("mailbox_seg_reset", &mailbox_seg_reset, "clear seg flag slots");
}
