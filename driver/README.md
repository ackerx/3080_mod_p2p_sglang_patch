# 驱动补丁：GeForce 双卡 BAR1 P2P + 双窗口信箱（open-gpu-kernel-modules 595.91.07）

## 这是什么

NVIDIA 官方在 GeForce 消费卡上禁用了 P2P（CUDA 自 Hopper 时代起对 GeForce 一律
`cudaDeviceCanAccessPeer = 0`）。社区补丁（tinygrad 的
[550.54.15-p2p](https://github.com/tinygrad/open-gpu-kernel-modules/tree/550.54.15-p2p)
分支、Aikitoria 的 595-p2p 补丁）通过修改内核模块使能了 **BAR1 P2P**：PCIe 对之间
直接 DMA 写对方 GPU 的物理地址，不再需要驱动层的 P2P 白名单。

本补丁基于同一思路，针对以下场景做了扩展（相对上游 595.91.07，20 个文件，
+706/-88 行）：

- **GA102 支持**：RTX 3080 / 3080-20G（魔改卡）等 Aikitoria 补丁未覆盖的型号；
- **BAR1 双窗口**：把设备 BAR1 的映射窗口管理扩展为两个独立窗口，让每卡可以
  同时把自己的 64MB 信箱 + 对端信箱都稳定映射到 BAR1 空间 —— 这是信箱
  push allreduce 的硬件前提。单窗口方案下信箱与其它 BAR1 分配互相挤占，
  落窗成败取决于启动期分配顺序（见 `scripts/sglang-p2p.sh` 的垫片预推）；
- 配套的 P2P DMA 路径修正（GMMU / io_vaspace / nv_gpu_ops 等）。

> 驱动魔改思路致敬 tinygrad / Aikitoria 社区工作。本补丁与它们的差异以
> GA102 + 双窗口为主，未对 3090/4090/5090 做回归测试。

## 构建与安装

```bash
# 1. 上游源码（版本必须一致：595.91.07）
git clone --branch 595.91.07 --depth 1 \
    https://github.com/NVIDIA/open-gpu-kernel-modules.git
cd open-gpu-kernel-modules

# 2. 应用补丁
git apply /path/to/p2p-mailbox-595.91.07.patch

# 3. 构建（需要与本机内核匹配的 headers）
cd kernel-open
make modules -j$(nproc)
# 产物: kernel-open/nvidia.ko（version 串会带上 "构建者@主机" 签名）

# 4. 安装（接管运行中的驱动需要先停 GPU 负载）
sudo systemctl stop <你的 GPU 服务/桌面>
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia   # 顺序不可反
sudo insmod nvidia.ko
sudo modprobe nvidia_uvm nvidia_drm nvidia_modeset
# 验证: version 串含 "@" 即魔改态
head -1 /proc/driver/nvidia/version
```

注意事项：

- **FLR 复位**：切换驱动后两卡需要 Function-Level Reset 才能稳定初始化，
  `scripts/sglang-p2p.sh` 里有完整流程（卸载→FLR→加载→逐卡 CUDA 就绪探测）；
- 模块加载/卸载步骤都有超时保护，若卡死（GPU 停在异常态）只能重启整机；
- 本补丁只在实验机（双 RTX 3080-20G / X99 平台 / PCIe 3.0 x16 /
  内核 6.x）上验证过，其他平台自行承担风险；
- 回滚：重装发行版驱动包即可（`apt install --reinstall nvidia-driver-xxx`）。

## 许可

跟随上游 open-gpu-kernel-modules：`kernel-open/` 为 MIT License，
`src/nvidia/` 为 NVIDIA Open Source License。本补丁作为其衍生修改，
同样适用上述许可。
