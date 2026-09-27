#!/usr/bin/env python3
# test_mailbox_torch.py — MailboxAR 双进程验证(JIT 编译 + eager 校验 + graph 重放)
# 用法: 两个进程并发 rank=0/1
import os
import sys
import datetime

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29591")
RANK = int(sys.argv[1])

import torch
import torch.distributed as dist

dist.init_process_group("gloo", rank=RANK, world_size=2)
torch.cuda.set_device(RANK)

SGL = os.environ.get("SGLANG_PKG", "sglang")  # sglang 包路径(默认走 site-packages)
sys.path.insert(0, SGL)
from sglang.srt.distributed.device_communicators.mailbox_all_reduce import (
    MailboxAR,
    _MBX_BYTES,
    _HALF_BYTES,
)

grp = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=120))
ar = MailboxAR(group=grp, device=torch.device(f"cuda:{RANK}"))
assert not ar.disabled
print(f"r{RANK} mailbox ready mbx={ar.mbx.data_ptr():#x} peer={ar.peer_mbx:#x}",
      flush=True)

# ---- eager 100 轮: 随机尺寸 bf16, 两 rank 相同输入 → 期望 2x ----
torch.manual_seed(1234)
for it in range(100):
    nbytes = 4096 * (1 + (it * 37) % 300)          # 4KB~1.2MB 变尺寸, 16B 对齐
    n = nbytes // 2
    x = torch.randn(n, dtype=torch.bfloat16).to(RANK)
    out = ar.custom_all_reduce(x)
    assert out is not None, f"r{RANK} it{it} should_custom_ar False"
    want = (x.float() * 2).bfloat16()
    assert torch.equal(out, want), (
        f"r{RANK} it{it} mismatch n={n} bad={(out.float() != want.float()).sum().item()}")
torch.cuda.synchronize()
err = ar.ops.state_get(ar.st.data_ptr(), 5)
rnd = ar.ops.state_get(ar.st.data_ptr(), 4)
print(f"r{RANK} eager-100 OK err={err} round={rnd}", flush=True)
assert err == 0

# ---- graph: capture 1 轮 + 变值 replay 200 次 ----
nbytes = 512 * 1024
n = nbytes // 2
x = torch.randn(n, dtype=torch.bfloat16).to(RANK)
out = ar.custom_all_reduce(x)          # eager warmup 1 轮
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    outg = ar.custom_all_reduce(x)
for i in range(200):
    x.copy_(torch.full((n,), float(i + 1), dtype=torch.bfloat16, device=f"cuda:{RANK}") * 0.25)
    g.replay()
torch.cuda.synchronize()
want = (x.float() * 2).bfloat16()
okv = torch.equal(outg, want)
err = ar.ops.state_get(ar.st.data_ptr(), 5)
rnd = ar.ops.state_get(ar.st.data_ptr(), 4)
print(f"r{RANK} graph-200 {'OK' if okv and err == 0 else 'FAIL'} "
      f"err={err} round={rnd} (expect {100 + 1 + 200 + 1})", flush=True)
assert okv and err == 0, f"r{RANK} GRAPH-FAIL"
print(f"r{RANK} MAILBOX-TORCH-ALL-PASS", flush=True)
