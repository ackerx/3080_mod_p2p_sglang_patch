#!/bin/bash
# sglang-p2p.sh — 魔改驱动(BAR1 双窗口) + 信箱 push allreduce sglang 一键启动
# 流程: 停 sglang → 停桌面/监测(干净会话, 信箱需驱动加载后首批分配落窗)
#       → 卸载发行版驱动 → FLR 复位两卡 → insmod 魔改 nvidia.ko → modprobe 配套
#       → 启 D-state 看门狗 → SGLANG_MAILBOX_AR=1 启动 sglang(TP2)
# 用法:
#   bash ~/sglang-p2p.sh start    启动(含驱动切换, 全程 sudo 需输一次密码)
#   bash ~/sglang-p2p.sh stop     停 sglang+看门狗(驱动保持魔改态)
#   bash ~/sglang-p2p.sh status   驱动/服务/看门狗/信箱落窗状态
# 注意:
#   * 会关闭桌面(display-manager), 请通过 SSH 使用; 完成后跑
#     恢复日常态(发行版驱动+桌面)请按你的环境自行处理(本仓库不含发行版驱动脚本)
#   * 若日志出现 "信箱落窗失败", 重跑一次 start(启动期分配挤占窗口时重试可救)
#   * 卸载/加载模块步骤均带 timeout 保护: 卡死(如 GPU 停在异常态)时自动失败
#     并提示重启整机, 不会无限挂起
set -u

# 可用环境变量覆盖: P2P_KO(魔改 nvidia.ko 路径) WD(看门狗) SGLANG_PY(python)
P2P_KO=${P2P_KO:-$HOME/p2p-mailbox/open-gpu-kernel-modules/kernel-open/nvidia.ko}
WD=${WD:-$(dirname "$(readlink -f "$0")")/p2p-watchdog.sh}
LOG=$HOME/logs/sglang-run.log

driver_kind() {
  local v; v=$(cat /proc/driver/nvidia/version 2>/dev/null | head -1)
  case "$v" in
    # 魔改 nvidia.ko 的 version 串含 "构建者@主机" 签名, 官方串无 @
    *@*) echo p2p ;;
    *)         echo normal ;;
  esac
}

gpu_count() { nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l; }

wait_cuda_ready() {
  # FLR 后 nvidia-smi/device_count 能过但设备可能惰性未就绪(如 /dev/nvidia1 缺失,
  # TP1 报 invalid device ordinal) —— 必须逐卡真实初始化
  local py=${SGLANG_PY:-python3} n i
  for i in 1 2 3 4 5 6; do
    n=$("$py" -c '
ok = 0
import torch
for d in range(2):
    try:
        torch.zeros(1, device=f"cuda:{d}")
        ok += 1
    except Exception:
        pass
print(ok)' 2>/dev/null || echo 0)
    [ "$n" = "2" ] && return 0
    echo "    CUDA 可用卡=$n(需2), 等 5s 重试 $i/6"
    sleep 5
  done
  echo "!! CUDA 设备未就绪"; return 1
}

# GPU 整轮重试: FLR 后个别 GPU 偶发停在 RESET_REQUIRED/GSP 半初始化态
# (单次复位救不回), 需整轮 卸载→复位→加载; 重试仍失败只能重启整机。
ensure_gpus_or_retry() {
  local target="$1" t
  wait_cuda_ready && return 0
  for t in 1 2; do
    echo "==> GPU 未就绪, 整轮重试 $t/2 (卸载→复位→加载)"
    sudo systemctl stop display-manager.service llm-monitor.service lactd.service 2>/dev/null
    sleep 3
    sudo timeout -k 5 60 modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null
    sleep 2
    echo 1 | sudo tee /sys/bus/pci/devices/0000:02:00.0/reset > /dev/null
    echo 1 | sudo tee /sys/bus/pci/devices/0000:03:00.0/reset > /dev/null
    sleep 5
    if [ "$target" = "normal" ]; then
      sudo timeout -k 5 90 modprobe nvidia
      sudo timeout -k 5 60 modprobe nvidia_uvm nvidia_modeset nvidia_drm
    else
      ( cd "$(dirname "$P2P_KO")" && sudo timeout -k 5 120 insmod "$(basename "$P2P_KO")" )
      sudo timeout -k 5 60 modprobe nvidia_uvm nvidia_modeset nvidia_drm
    fi
    sleep 4
    wait_cuda_ready && return 0
  done
  return 1
}


switch_to_p2p_driver() {
  echo "==> [1/6] 停桌面与 GPU 监测(信箱需要干净驱动会话)"
  sudo systemctl stop display-manager.service llm-monitor.service lactd.service
  sleep 8

  echo "==> [2/6] 卸载当前 nvidia 模块"
  sudo timeout -k 5 60 modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia
  rc=$?
  if [ $rc -ne 0 ]; then
    if [ $rc -ge 124 ]; then
      echo "!!  卸载卡死(timeout)——GPU 处于异常态且模块状态已脏, 最稳办法是重启整机后重试"
    else
      echo "!!  卸载失败(rc=$rc), 可能有进程占用:"; sudo fuser -v /dev/nvidia* 2>&1 | head -8
    fi
    return 1
  fi
  sleep 3

  echo "==> [3/6] FLR 复位两卡"
  echo 1 | sudo tee /sys/bus/pci/devices/0000:02:00.0/reset > /dev/null
  echo 1 | sudo tee /sys/bus/pci/devices/0000:03:00.0/reset > /dev/null
  sleep 2

  echo "==> [4/6] 加载魔改双窗口驱动(窗口2 = FB[0x13000000,+100MB))"
  # FLR 会触发 udev 自动拉起发行版模块, insmod 前先清理, 失败则重试(最多3次)
  insmod_ok=0
  for i in 1 2 3; do
    sudo timeout -k 5 60 modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null
    sleep 1
    if ( cd "$(dirname "$P2P_KO")" && sudo timeout -k 5 120 insmod "$(basename "$P2P_KO")" ); then
      insmod_ok=1; break
    fi
    echo "    insmod 被占用(udev 竞态/未清干净), 重试 $i/3"
    sleep 2
  done
  [ "$insmod_ok" = "1" ] || { echo "!! insmod 失败(模块/GPU 状态可能已脏) — 最稳办法是重启整机后重跑本脚本"; return 1; }
  sudo timeout -k 5 60 modprobe nvidia_uvm nvidia_modeset nvidia_drm
  sleep 4

  if [ "$(gpu_count)" != "2" ]; then
    echo "!!  GSP 初始化未完全(可见 $(gpu_count) 卡), 自动重试一次"
    sudo timeout -k 5 60 modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia; sleep 4
    echo 1 | sudo tee /sys/bus/pci/devices/0000:02:00.0/reset > /dev/null
    echo 1 | sudo tee /sys/bus/pci/devices/0000:03:00.0/reset > /dev/null
    sleep 8
    ( cd "$(dirname "$P2P_KO")" && sudo timeout -k 5 120 insmod "$(basename "$P2P_KO")" ) || return 1
    sudo timeout -k 5 60 modprobe nvidia_uvm nvidia_modeset nvidia_drm; sleep 4
  fi
  if [ "$(gpu_count)" != "2" ]; then
    echo "!!  两卡未全部就绪 — GSP 偶发 boot 失败, 最稳办法是重启整机后再跑本脚本"
    return 1
  fi
  echo "    两卡 OK: $(nvidia-smi --query-gpu=index,name --format=csv,noheader | tr '\n' ' ')"
}

do_start() {
  if ! sudo -n true 2>/dev/null; then
    if [ -n "${SUDO_PW:-}" ]; then
      echo "$SUDO_PW" | sudo -S -v 2>/dev/null    # 自动化/脚本调用时可 SUDO_PW=密码 喂入
    else
      echo "==> 需要 sudo 权限(输入密码):"
    fi
    sudo -v || { echo "!! sudo 认证失败"; exit 1; }
  fi   # 已有凭证则跳过
  # 下方的 sglang.sh 是本机业务启动脚本, 请替换为你自己的 sglang 启动方式
  bash "$HOME/sglang.sh" stop > /dev/null 2>&1

  echo "==> 当前驱动: $(driver_kind)"
  switch_to_p2p_driver || { echo "!!  驱动切换失败, 未启动服务"; exit 1; }
  ensure_gpus_or_retry p2p || { echo "!!  GPU 持续未就绪, 请重启整机后重试"; exit 1; }

  echo "==> [5/6] 启动 D-state 看门狗"
  pkill -f 'p2p-watchd[o]g.sh' 2>/dev/null
  nohup bash "$WD" > /tmp/wd.log 2>&1 &
  sleep 1

  echo "==> [6/6] 启动 sglang (SGLANG_MAILBOX_AR=1)"
  wait_cuda_ready || exit 1
  SGLANG_MAILBOX_AR=1 bash "$HOME/sglang.sh" bg   # 或手动: python -m sglang.launch_server ... TP=2
  echo
  echo "==> 就绪约需 1.5~2 分钟。验证信箱已落窗/已启用:"
  echo "    grep -E 'MailboxAR|Using MailboxAR' $LOG | tail -4"
  echo "    应看到 'ready rank=0/1' 而无 '落窗失败'; 之后跑 bash $0 status"
}

do_stop() {
  bash "$HOME/sglang.sh" stop
  pkill -f 'p2p-watchd[o]g.sh' 2>/dev/null && echo "==> 看门狗已停"
  echo "==> (驱动仍为魔改态)"
}

do_status() {
  echo "==> 驱动: $(driver_kind) ($(cat /proc/driver/nvidia/version 2>/dev/null | head -1 | cut -c1-60)...)"
  echo "==> 显卡: $(gpu_count) 卡可见"
  bash "$HOME/sglang.sh" status 2>/dev/null | head -3
  pgrep -f 'p2p-watchd[o]g.sh' > /dev/null && echo "==> 看门狗: 运行中" || echo "==> 看门狗: 未运行"
  echo "==> 信箱最近日志:"; grep -E 'MailboxAR' "$LOG" 2>/dev/null | tail -3 | sed 's/^/    /'
}

case "${1:-start}" in
  start)  do_start ;;
  stop)   do_stop ;;
  status) do_status ;;
  *) echo "用法: $0 [start|stop|status]"; exit 1 ;;
esac
