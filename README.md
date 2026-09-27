# 写在前面
我只是略懂一些计算机系统结构，本项目所有工作是和人工智能（Artifical Intelligence,AI）一起研究的。
绝大多数工作是AI做的。我只是在关键环节提出了我的看法。细节我不是很清楚。
我建议你们使用时，也让AI自己来看。

# 原理
现有开源驱动实现卡间P2P的原理是利用vbios修改BAR（基地址寄存器）的功能，通过地址翻译，来得到其他卡上的目标显存地址，直接写入数据。
数据不通过内存存储中转，但指令和数据流仍需要从CPU走。这个开销似乎无法避免。
于是，问题在于这个地址翻译的空间有多少。
现有正规显卡的ReBar范围据说可以调到32G，但魔改卡只有最多256MB档。由于没有大bar的vbios，所以就无法直接利用现有的开源驱动。
我试过可以强制打开P2P，但实际是假的。数据根本写不过去。
所以剩下的路就是利用这个小小的256MB的窗口来收发数据。

# 效果
PP最大值，从1400提升到1700。
TG不开推测，50提升到60
但其实压下来功耗后，长上下文好像也就提升10%不到。我还没细测。你们自己琢磨哈。

# 如何评价锁消费卡P2P这种行为？
此处省略3000字。。。

# mailbox-allreduce — 无 NVLink 消费级双卡的 TP=2 真·P2P Allreduce

[中文](README.md) | [English](README.en.md)

让两块官方禁用 P2P 的 GeForce 卡，在 sglang TP=2 推理里把 allreduce 跑在
PCIe BAR1 真直连上（10.26 GB/s，≈1.36× 同环境 NCCL），decode CUDA graph
原生兼容。三层结构：

1. **内核模块补丁**（open-gpu-kernel-modules 595.91.07）：使能 GeForce BAR1
   P2P + BAR1 双窗口管理（64MB 信箱稳定落窗的硬件前提）；
2. **用户态信箱协议**（torch extension，JIT 编译）：5-kernel 锁步 push
   allreduce，把输入直接 DMA 写进对端显存的"信箱"，对端本地归约；
3. **sglang communicator**：实现 CustomAllreduce 同款接口面，3 行接线，
   `SGLANG_MAILBOX_AR=1` 启用，超限自动回落 NCCL。

## 为什么不是 NCCL / CustomAllreduce

消费卡上 `cudaDeviceCanAccessPeer = 0`：NCCL 只能走 SHM host 中转
（每字节两次 PCIe 穿越，~6.9 GB/s；强开 P2P transport 握手卡死）；
CustomAllreduce 需要 P2P 注册 buffer，注册失败同样回落 NCCL。本方案在
驱动层放行 P2P 后，用一块固定地址的 64MB 信箱绕开"每次 AR 动态分配
注册 buffer"的整个环节 —— 信箱地址固定，CUDA graph 内 kernel 直接记录
裸指针，decode 重放零开销。

## 性能摘要（双 RTX 3080-20G / PCIe 3.0 x16 实测）

| 路径 | AR 有效带宽 |
|---|---|
| NCCL（消费卡默认，SHM host 中转） | ~6.9 GB/s |
| NCCL（同驱动同环境实测） | 8.86 GB/s |
| **本方案（10MB~24MB）** | **9.15~9.28 GB/s**（纯 push 链路 10.26，全双工 19.3~20.1） |

端到端收益 = AR 占比 × 链路优势：prefill AR 占 ~21%（chunk 1024），
decode ITL 仅 2~3% —— **这是通信侧的"最后一公里"优化，不是数量级提升**；
从 NCCL SHM 默认态迁移是收益最大的场景。更完整的口径、测量纪律
（跨日对比会骗人）与"为什么计算×通信重叠在真机上不赚钱"的完整数据见
[docs/performance.md](docs/performance.md)。

## 硬件/软件前提

- 两张卡挂在同一 root complex 下（PCIe 直连，不过 NUMA 中转）；
- Linux，open kernel module 路线（595.91.07 上验证），root 权限；
- sglang 0.5.19（其他版本需对齐 communicator 接口面）；
- 接受"常驻魔改驱动 + D-state 卡死需整机重启"的实验机属性。

## 快速开始

```bash
# 1. 驱动: 构建并切换到魔改模块(FLR/就绪探测见脚本)
git clone --branch 595.91.07 --depth 1 https://github.com/NVIDIA/open-gpu-kernel-modules.git
cd open-gpu-kernel-modules && git apply /path/to/driver/p2p-mailbox-595.91.07.patch
cd kernel-open && make modules -j && sudo insmod nvidia.ko   # 详见 driver/README.md

# 2. 裸链路验证(双终端, CUDA_VISIBLE_DEVICES=0/1)
python tests/test_mailbox_torch.py

# 3. sglang 接入(两文件 + 3 行接线)并启动
SGLANG_MAILBOX_AR=1 python -m sglang.launch_server --model <model> --tp 2
# 日志确认: [MailboxAR] ready rank=0 attempt=0 ...
```

详细部署（FLR 复位、落窗排障、看门狗）：[docs/deploy.md](docs/deploy.md)

## 目录结构

```
├── driver/     驱动补丁(GA102 + BAR1 双窗口) + 构建说明
├── sglang/     mailbox_ar_ext.cu + mailbox_all_reduce.py + 接线说明
├── tests/      裸链路冒烟 / 560 轮压力锤 / 尺寸扫描 / 重叠微基准 / pp A/B
├── scripts/    一键启停(驱动切换+FLR+看门狗) / D-state 看门狗
└── docs/       原理与设计 / 部署指南 / 性能数据 / 限制与已知问题
```

## 文档

- [原理与设计](docs/principle.md) — 三层结构、信箱布局、锁步协议、
  每个工程决策背后的微基准证据
- [部署指南](docs/deploy.md) — 三层独立验证、故障排查速查
- [性能数据](docs/performance.md) — 带宽逐层拆解、端到端口径、测量纪律
- [限制与已知问题](docs/limitations.md) — 先算账再上：收益边界

## 许可与致谢

- sglang/tests/scripts 代码：MIT License
- 驱动补丁：跟随上游 open-gpu-kernel-modules（kernel-open 为 MIT，
  src/nvidia 为 NVIDIA Open Source License）
- 驱动 P2P 使能思路来自社区 [tinygrad open-gpu-kernel-modules
  550.54.15-p2p](https://github.com/tinygrad/open-gpu-kernel-modules/tree/550.54.15-p2p)
  分支与 Aikitoria 的 595-p2p 补丁，本仓库的差异（GA102 支持 + BAR1
  双窗口）在其基础上扩展

## 状态

实验级开源：单机验证、TP=2、无自动化 CI。欢迎在其它消费卡组合上
复测并回报数据。

