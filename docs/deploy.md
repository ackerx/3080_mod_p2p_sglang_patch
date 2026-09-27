# 部署指南

前提：双 PCIe 直连的 GeForce 卡（同一 root complex 下，不跨 NUMA 中转）、
Linux + open kernel module 路线、root 权限。完整流程分三层，每层独立可验证。

## 第 1 层：驱动（BAR1 P2P 使能 + 双窗口）

见 `driver/README.md`。要点：

1. clone 上游 `open-gpu-kernel-modules`（**版本必须与本机驱动一致**，
   595.91.07 上验证），`git apply p2p-mailbox-595.91.07.patch`；
2. `make modules` 构建，停 GPU 负载后 `rmmod` 官方模块、`insmod nvidia.ko`；
3. **FLR 复位两卡 + 逐卡 CUDA 就绪探测**是稳定初始化的关键
   （`scripts/sglang-p2p.sh` 有完整实现：卸载→FLR→加载→探测，整轮
   重试逻辑，卡死自动失败提示重启）；
4. 验证：`head -1 /proc/driver/nvidia/version` 的 version 串含
   "构建者@主机" 签名即魔改态。

## 第 2 层：信箱落窗验证（不上 sglang，先裸测）

信箱要求"驱动加载后首批分配"的显存能落进 BAR1 窗口。裸测链路：

```bash
# 双终端分别跑（CUDA_VISIBLE_DEVICES=0 / 1）:
python tests/test_mailbox_torch.py
```

通过标准：双方日志 `ready`，互写数据校验全对。这一步过了，硬件链路
（驱动窗口 + P2P DMA + 用户态协议）就算全通。

压力验证（可选但强烈建议）：

```bash
python tests/hammer_mbx.py   # 560 轮锁步，fb/大路径交替，任何 err 均为协议 bug
python tests/mailbox_sweep.py # 8KB~24MB 尺寸扫描，对照带宽曲线
```

## 第 3 层：sglang 接入

见 `sglang/README.md`（两文件 + 3 行接线）。启动与确认：

```bash
SGLANG_MAILBOX_AR=1 python -m sglang.launch_server \
    --model <你的模型> --tp 2 ...
grep -E 'MailboxAR' <日志> | tail -2
# 期望: [MailboxAR] ready rank=0 attempt=0 mbx=0x... peer=0x...
```

`attempt>0` = 重试后落窗；`信箱落窗失败` = 回落 NCCL 并打印原因，
重启一次通常可恢复（启动期显存分配顺序问题）。

## 日常启停

`scripts/sglang-p2p.sh`（需按自己环境改 sglang 启动段）：

- `start`：停服务 → 停桌面/监测（干净会话）→ 切魔改驱动 → FLR →
  看门狗 → `SGLANG_MAILBOX_AR=1` 启动；
- `stop`：停服务与看门狗（驱动保持魔改态）；
- `status`：驱动态 / GPU 数 / 服务 / 落窗日志。

`scripts/p2p-watchdog.sh`：D-state 看门狗。P2P 调试中内核态卡死表现为
用户进程 D 状态且 kill 不动，只能整机重启 —— 看门狗检测进程 D 状态
持续超时自动 `sysrq b` 重启，**避免实验把机器挂死在半夜**。

## 故障排查速查

| 症状 | 首查 |
|---|---|
| 落窗 16 轮耗尽 | 启动期是否有别的东西先占显存（桌面/监测进程）；垫片预推是否生效 |
| CUDA 设备未就绪 / invalid device ordinal | FLR 后惰性未就绪，看 `sglang-p2p.sh` 的逐卡探测 |
| 进程 D 状态 kill 不动 | 内核态卡死；看门狗在吗？没有就手动 sysrq 重启 |
| AR 走了 NCCL（带宽 6.9 GB/s 水平） | 日志查 `Using MailboxAR` 是否出现；`SGLANG_MAILBOX_AR=1` 是否真的传进去 |
| decode 图 capture 报错 | sglang 版本差异导致接口面变化；`capture/register_graph_buffers` 需与所用版本对齐 |
