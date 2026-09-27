# 原理与设计

## 1. 问题背景

两块无 NVLink 的消费级 GPU（GeForce）做 TP=2 推理时，allreduce 只有两条路：

1. **NCCL 走 SHM host 中转**：每字节两次 PCIe 穿越（GPU→host→GPU），实测
   ~6.8-6.9 GB/s；强开 NCCL P2P transport 则握手卡死（消费卡无驱动层 P2P）；
2. **框架自带 custom allreduce**（vLLM/sglang）：单内核写对端显存 + 本地归约，
   但同样依赖底层 P2P 能力 —— 消费卡上 `cudaDeviceCanAccessPeer = 0`，
   buffer 注册失败自动回落 NCCL。

而 PCIe 协议本身完全支持 P2P DMA（设备写设备的物理地址，RC 转发）。
**卡的 PCIe 带宽（Gen3 x16 ≈ 16 GB/s 原始）远没有被用上**。

## 2. 三层结构

```
┌─────────────────────────────────────────────────────────┐
│ sglang (TP=2)                                            │
│   MailboxAR communicator（自定义 CustomAllreduce 接口面） │
│     decode CUDA graph 兼容（裸指针 capture/重放）          │
├─────────────────────────────────────────────────────────┤
│ 用户态协议层 mailbox_ar_ext.cu (JIT torch extension)      │
│   锁步 push allreduce：5-kernel 流水                      │
│   waitack → produce(push) → flag → wait_local → reduce → ack │
├─────────────────────────────────────────────────────────┤
│ 魔改内核模块（open-gpu-kernel-modules 595.91.07 + 补丁）   │
│   ① 使能 GeForce 的 BAR1 P2P（写对端 BAR1 空间=写对端显存）│
│   ② BAR1 双窗口：64MB 信箱(双方互映射)稳定落窗            │
└─────────────────────────────────────────────────────────┘
```

### 2.1 驱动层

- NVIDIA 内核模块在 RM 层（Resource Manager）用白名单阻止 GeForce 之间
  建立 peer 映射。补丁在 bus/peer 分配路径放行消费卡（这是社区
  tinygrad/Aikitoria 补丁已验证的路线）；
- **双窗口扩展是本仓库的差异点**：信箱方案需要每卡同时保持
  "自己 64MB 信箱 + 对端 64MB 信箱"两个 BAR1 映射。BAR1 映射窗口是
  驱动管理的稀缺资源（默认单窗口语义），不扩展时信箱能否落窗纯看启动期
  显存分配顺序，成功率低且不可靠。补丁把窗口管理改为双窗口，
  使 64MB 信箱在驱动加载后的首批分配即可稳定落窗；
- 验证手段：`lspci` 的 BAR1 尺寸、对端物理地址写入探测、FLR 后逐卡
  CUDA 就绪探测（见 `scripts/sglang-p2p.sh`）。

### 2.2 用户态协议（mailbox_ar_ext.cu）

每卡在驱动加载后**首批**分配一块 64MB 显存作为"信箱"，布局：

```
[flag int][ack int][pad 2int][data 半区A 24MB][data 半区B 24MB][seg 区 16MB]
           └─ 状态区 int[8]: eflag/eack/round/err/done
```

allreduce 五内核锁步（每 rank 对称）：

1. `waitack`：spin 对端信箱头部 ack == 期望值（对端上一轮已消费完，
   半区可覆写）；
2. `produce`：把自己的输入**直接 push 写**进对端信箱当前半区 —— 这是
   一次显存 store 指令，跨 PCIe 变成 posted write DMA，单向实测
   **10.26 GB/s**（硬件上限，SM store 与 copy engine 两条独立路径同值）；
3. `flag`：1-thread 内核写对端 flag = round（流序保证数据先于 flag 到达，
   同为 PCIe posted 写按序投递）；
4. `wait_local`：spin 自己信箱 flag == round（对端的 push 已投递完成）；
5. `reduce`：`out = local + 对端数据`（本地显存带宽归约），随后 `ack`
   递增 round。

关键工程决策（每条都有微基准数据支撑，详见性能文档）：

- **ping-pong 半区**：规避跨轮 L2 stale；
- **大路径无内核内 fence**：`__threadfence()` 的 PCIe 排空在双向并发下
  会把 produce 压到 5.9 GB/s 且整 grid 卡死 —— 数据/flag 的顺序可靠性
  改由同流内核边界保证；
- **数据区 256B 对齐**：原先 +16B 偏移让 64B warp store 跨 3 个 L2 扇区，
  push 掉 40%；
- **小消息（≤4MB）融合单内核**：fence(device 域)+块计数+末块写 flag，
  省一次内核发射（小消息时 ~4µs 占比可观）—— decode CUDA graph 内
  每步都在这条路径上；
- **CUDA graph 兼容**：所有内核参数是裸指针（信箱地址固定不变），
  capture/重放无需 register_graph_buffers —— 这是相比
  CustomAllreduce（动态 buffer + graph buffer 注册，129 个 buffer 会
  越出 BAR1 窗口）从根上绕开的墙。

## 3. 已知边界

- world_size == 2（信箱是 pairwise 协议；TP=2 单机双卡场景）；
- 单次 AR ≤ 24MB（半区容量），超限自动回落 NCCL；
- 需要 root 重装内核模块；信箱落窗依赖"驱动加载后首批分配"，
  由启动脚本的垫片预推保证；
- 无 ECC/校验：P2P 写的可靠性由 PCIe 链路层 CRC 保证，与本地显存写同级。
