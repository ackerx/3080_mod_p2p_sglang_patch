# 性能数据

> 全部数字来自单台实验机：双 RTX 3080 20G（魔改 GA102）/ X99 平台 /
> PCIe 3.0 x16（Gen4 卡被平台封顶 Gen3）/ 功耗锁 320W / sglang 0.5.19 +
> Qwen3.8-27B-AWQ-INT4 (hidden=5120, 64 层)。换平台/拓扑数字会变，
> 但方法（口径、对照、判定纪律）可复用。

## 1. 链路与微基准

| 路径 | 实测 | 说明 |
|---|---|---|
| PCIe P2P push（纯链路） | **10.26 GB/s** | SM store 与 copy engine 两条独立硬件路径同值 → 硬件上限，无软件空间 |
| 同上，双向全双工 | 每方向 9.6~10.26，总 19.3~20.1 | 双卡同时互 push 仅掉 3~6% |
| AR 端到端（含协议） | 10MB 9.15 / 24MB 9.22~9.28 GB/s | push 2.34ms + 归约 0.10ms + flag/同步 0.15ms |
| NCCL（消费卡默认态） | ~6.8-6.9 GB/s | P2P 被禁 → SHM host 中转，每字节两次 PCIe 穿越 |
| NCCL 强开 P2P transport | 握手卡死 | 不可用 |
| NCCL（同驱动同环境实测） | 8.86 GB/s | sglang 内 CustomAllreduce 失败回退态 |
| 微基准同尺寸对照 | **信箱 ≈ 1.36× NCCL** | 全尺寸 AR 微基准 |

逐层带宽拆解（为什么 10.26 就是尽头）：16.0 原始 → 15.75（128b/130b 编码）
→ ~14.5（TLP 封装 92.1%，MPS=256B 顶格）→ 13.20（D2H 一跳）→ **10.26
（P2P 两跳，RC fabric 转发 + 对端控制器 BAR1→VRAM 的固有 ~22%）**。
每一层都是协议/芯片固定结构，发射端无软件空间。

## 2. 真机端到端（sglang TP=2）

AR 占比决定了端到端收益的量级（chunk 1024 口径）：

- 单次 AR 10.5MB ≈ 1.14ms；每 chunk 146ms 通信 vs 707ms 计算
  → **AR 占 prefill ~21%、占 decode ITL 仅 2~3%**；
- 端到端收益因此集中在：从"NCCL SHM 中转"默认态迁移时（AR 环节
  +34~49%）收益最明显；NCCL 已被调优到能用的环境下，AR 环节
  +4~5%，端到端 prefill ~+1%；
- decode 的小消息 AR（≤4MB 单内核路径）延迟优于 NCCL 集合通信栈，
  对 ITL 的改善在小 batch 单流场景可感知，但 AR 本身只占 ITL 2~3%。

**测量纪律（跨日对比会骗人）**：同机不同日重测同一配置，prefill 吞吐
漂移可达 1.5%（环境/温度/频率态）。判定必须用**同日同机 A/B**，
且变量一次只动一个。示例（26.13 四组，pp tok/s，每组 128K×2 + 40K×2）：

| 配置 | 128K | 40K |
|---|---|---|
| CHUNK=1024, SEG=off | 1119 / 1119 | 1464 / 1458 |
| CHUNK=1024, SEG=on | 1115 / 1117 | 1467 / 1461 |
| CHUNK=2048, SEG=off | 1129 / 1124 | 1478 / 1472 |
| CHUNK=2048, SEG=on | 1131 / 1126 | 1480 / 1475 |

（SEG = 库内分段 AR 实验开关，结论见下节；CHUNK 1024→2048 本身 +0.9%
是该实验的意外收获，与 AR 无关。）

## 3. 计算×通信重叠实验（为什么放出来但不默认开）

`mailbox_ar_ext.cu` 里带了一套分段 AR 协议（`submit_seg/wait_seg`，
copy-engine push + 槽位 flag 生命周期）。它把 GEMM 按行切段，段完成即
CE push 到对端槽位与下一段 GEMM 硬件并行。微基准层面成立
（GEMM×CE push 并行净赚 ~1ms/AR，GEMM 无掉速），正确性全绿
（60 项双端对照 + 200 轮压力锤子 bit 级相等），但在真机
Qwen3.8-27B-AWQ 上**净收益 ≈0（chunk 1024）~ +0.2%（chunk 2048）**：

- chunk 小 → 段 GEMM 只有 ~0.85ms，可藏的 push 只有 ~0.28ms，
  段切分的 weight 重读损失与重叠收益同量级，精确抵消；
- AWQ kernel 的段切分损失远大于微基准用的 bf16 dense。

教训：**微基准的正收益信号必须过真机同日 A/B 才算数**；
通信重叠的收益上限 = AR 占比 × 链路优势，先算账再动手。

## 4. 复现工具

| 工具 | 用途 |
|---|---|
| `tests/test_mailbox_torch.py` | 单机双进程信箱 AR 冒烟（正确性） |
| `tests/hammer_mbx.py` | 560 轮锁步压力锤（fb/大路径交替） |
| `tests/mailbox_sweep.py` | 尺寸扫描（8KB~24MB 带宽曲线） |
| `tests/gate8_ce_bench.py` | GEMM×CE push 重叠微基准（四组计时） |
| `tests/pp_bench.py` | 真机 prefill 吞吐 A/B（max_tokens=1 隔离 prefill，salt 防前缀缓存） |
