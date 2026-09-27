#!/bin/bash
# P2P 看门狗: 监测进程若持续 D 状态且存活 >60s, 连续 6 个采样周期(30s)不解除则强制重启
# (PCIe P2P 调试中内核态卡死会表现为用户进程 D 状态, 只能整机重启恢复)
# 进程名单按需修改 PROCS
PROCS='e1scan|e1sp|probe_ce|e5_poll|ipc2p|bw_test|peern|mailbox_ar|sglang.*|all_.*_perf|sendrecv_perf|broadcast_perf|hypercube_perf|gather_perf|reduce_scatter_perf'
LOG=/tmp/watchdog_ab1.log
echo "$(date +%T) watchdog start" >> $LOG
STRIKE=0
while true; do
  sleep 5
  BAD=$(ps -eo stat,pid,etimes,comm --no-headers | awk -v procs="$PROCS" '$1 ~ /^D/ && $4 ~ procs {print $2":"$3":"$4}')
  if [ -n "$BAD" ]; then
    echo "$(date +%T) D-state: $BAD" >> $LOG
    if [ -n "$(echo "$BAD" | awk -F: '$2 > 60')" ]; then
      STRIKE=$((STRIKE+1))
    else
      STRIKE=0
    fi
  else
    STRIKE=0
  fi
  if [ $STRIKE -ge 6 ]; then
    echo "$(date +%T) FORCE REBOOT (D-state持续90s)" >> $LOG
    sync & sleep 1
    echo b > /proc/sysrq-trigger
  fi
done
